/// A Spleeter model's inference: magnitudes in, each stem's raw mask out.
///
/// `magnitude` is `channels × bins × frames`, row-major (`[(channel * bins + bin) * frames + frame]`); each returned mask
/// has the same layout and count, one per stem in `SpleeterStemLayout.stemNames` order. Raw masks are the magnitude
/// each stem's U-Net estimates (`ratio_masks=False` in `Tools/spleeter-pytorch`); the separator turns them into ratio
/// masks itself.
public protocol SpleeterMaskInference: Sendable {
    func masks(magnitude: [Float], channels: Int, bins: Int, frames: Int) throws -> [[Float]]
}

/// What the separator rejects.
public enum SpleeterSeparatorError: Error, Equatable {
    /// The input had a channel count other than 1 or 2.
    case unsupportedChannelCount(Int)

    /// The inference returned a different number of masks than the layout has stems.
    case unexpectedMaskCount(expected: Int, actual: Int)

    /// A mask had a different element count than the magnitude.
    case unexpectedMaskSize(expected: Int, actual: Int)
}

/// Spleeter's ratio masks (`separator.py`): each stem's share of the summed squared raw masks, per element.
public enum SpleeterRatioMask {
    /// Keeps the division finite where every mask is zero.
    public static let epsilon: Float = 1e-10

    /// `(m_i² + ε / n) / (Σ m_j² + ε)` for each of the `n` masks, element by element, in FP32.
    public static func ratioMasks(_ masks: [[Float]]) -> [[Float]] {
        guard let count = masks.first?.count else { return [] }
        let stemShare = epsilon / Float(masks.count)
        var sum = [Float](repeating: epsilon, count: count)
        for mask in masks {
            for index in 0 ..< count {
                sum[index] += mask[index] * mask[index]
            }
        }
        return masks.map { mask in
            (0 ..< count).map { index in (mask[index] * mask[index] + stemShare) / sum[index] }
        }
    }
}

/// One Spleeter window end to end: STFT, the model's masks, ratio masking, and the inverse STFT per stem.
public struct SpleeterSeparator: Sendable {
    public let stems: SpleeterStemLayout
    public let configuration: SpleeterConfiguration
    public let inference: any SpleeterMaskInference
    public let stft: SpleeterSTFT

    /// - Parameters:
    ///   - stems: The stems the model separates.
    ///   - inference: The model.
    ///   - configuration: The geometry the model was converted with.
    ///   - fft: The FFT to use; `PortableRealFFT` when nil.
    public init(
        stems: SpleeterStemLayout,
        inference: any SpleeterMaskInference,
        configuration: SpleeterConfiguration = SpleeterConfiguration(),
        fft: (any SpleeterFFT)? = nil
    ) {
        self.stems = stems
        self.configuration = configuration
        self.inference = inference
        stft = SpleeterSTFT(
            fftSize: configuration.fftSize,
            hopLength: configuration.hopLength,
            frequencyBins: configuration.frequencyBins,
            windowGain: configuration.windowGain,
            fft: fft
        )
    }

    /// The samples one call covers.
    public var windowLength: Int {
        configuration.windowLength
    }

    /// Separates one window.
    ///
    /// The model always sees two channels: a mono input is given as both and comes back as one. Each channel is padded
    /// with zeros, or cut, to `windowLength`; each stem comes back at the input's length (the longer channel's), at most
    /// `windowLength`. For every channel the stems add up to the input.
    ///
    /// - Parameter channels: One or two channels of samples at the model's rate (44.1 kHz).
    /// - Returns: The stems in `stems.stemNames` order, each with the input's channels.
    public func separate(_ channels: [[Float]]) throws -> [[[Float]]] {
        guard channels.count == 1 || channels.count == 2 else {
            throw SpleeterSeparatorError.unsupportedChannelCount(channels.count)
        }
        let length = min(channels.map(\.count).max() ?? 0, windowLength)
        let stereo = channels.count == 2 ? channels : [channels[0], channels[0]]
        let spectrograms = stereo.map { stft.forward($0.paddedOrClamped(to: windowLength)) }
        let bins = configuration.frequencyBins
        let frames = spectrograms[0].frames

        let magnitude = spectrograms.flatMap(\.magnitude)
        let masks = try inference.masks(magnitude: magnitude, channels: 2, bins: bins, frames: frames)
        guard masks.count == stems.stemCount else {
            throw SpleeterSeparatorError.unexpectedMaskCount(expected: stems.stemCount, actual: masks.count)
        }
        for mask in masks where mask.count != magnitude.count {
            throw SpleeterSeparatorError.unexpectedMaskSize(expected: magnitude.count, actual: mask.count)
        }

        let channelSize = bins * frames
        return SpleeterRatioMask.ratioMasks(masks).map { ratio in
            (0 ..< channels.count).map { channel in
                let offset = channel * channelSize
                var masked = spectrograms[channel]
                for index in 0 ..< channelSize {
                    masked.real[index] *= ratio[offset + index]
                    masked.imag[index] *= ratio[offset + index]
                }
                return Array(stft.inverse(masked).prefix(length))
            }
        }
    }
}

extension Array where Element == Float {
    func paddedOrClamped(to targetLength: Int) -> [Float] {
        if count == targetLength {
            self
        } else if count < targetLength {
            self + [Float](repeating: 0, count: targetLength - count)
        } else {
            Array(prefix(targetLength))
        }
    }
}
