#if canImport(Foundation)
import Foundation
#endif

/// One channel's complex spectrogram, laid out bin-major — `bins × frames`, element `[bin * frames + frame]` — the
/// layout the model's `magnitude` input and mask outputs use.
public struct ComplexSpectrogram: Sendable, Equatable {
    /// The number of frequency bins kept, from DC up.
    public let bins: Int

    /// The number of time frames.
    public let frames: Int

    /// The real parts, `bins × frames`.
    public var real: [Float]

    /// The imaginary parts, `bins × frames`.
    public var imag: [Float]

    public init(bins: Int, frames: Int, real: [Float], imag: [Float]) {
        precondition(real.count == bins * frames && imag.count == bins * frames)
        self.bins = bins
        self.frames = frames
        self.real = real
        self.imag = imag
    }

    /// `|real + i·imag|` per element, in the same layout.
    public var magnitude: [Float] {
        zip(real, imag).map { ($0 * $0 + $1 * $1).squareRoot() }
    }
}

/// Spleeter's short-time Fourier transform: a periodic Hann window, frames centered by padding `fftSize / 2` zeros on
/// each side (`torch.stft(center: true, pad_mode: "constant")`), only the lowest `frequencyBins` bins kept.
public struct SpleeterSTFT: Sendable {
    /// The gain of vDSP's `hanningNormalized` window over the plain periodic Hann, `√(8/3)`: the window swift-spleeter
    /// has used through 0.3.1. It scales the magnitudes the model sees; the inverse divides it back out.
    public static let vDSPHannGain = Float((8.0 / 3.0).squareRoot())

    public let fftSize: Int
    public let hopLength: Int
    public let frequencyBins: Int

    /// The periodic Hann window times `windowGain`, `windowGain · (0.5 − 0.5 cos(2πn / fftSize))`.
    public let window: [Float]

    public let fft: any SpleeterFFT

    /// - Parameters:
    ///   - fftSize: The FFT size, a power of two.
    ///   - hopLength: The step between frames.
    ///   - frequencyBins: The bins kept, at most `fftSize / 2 + 1`.
    ///   - windowGain: The window's gain over the plain periodic Hann (`torch.hann_window`, what Spleeter trained
    ///     with). `vDSPHannGain` reproduces the separation swift-spleeter has shipped; 1 is the training's scale.
    ///   - fft: The FFT to use, of `fftSize`; `PortableRealFFT` when nil.
    public init(
        fftSize: Int,
        hopLength: Int,
        frequencyBins: Int,
        windowGain: Float = vDSPHannGain,
        fft: (any SpleeterFFT)? = nil
    ) {
        precondition(frequencyBins <= fftSize / 2 + 1, "frequencyBins must be <= fftSize / 2 + 1")
        precondition(fft.map { $0.size == fftSize } ?? true, "the FFT's size must be fftSize")
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.frequencyBins = frequencyBins
        self.fft = fft ?? PortableRealFFT(size: fftSize)
        let gain = Double(windowGain)
        window = (0 ..< fftSize).map {
            Float(gain * (0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(fftSize))))
        }
    }

    /// The number of frames `forward` makes of `sampleCount` samples.
    public func frameCount(forSampleCount sampleCount: Int) -> Int {
        (sampleCount + 2 * (fftSize / 2) - fftSize) / hopLength + 1
    }

    /// The forward transform of one channel.
    public func forward(_ waveform: [Float]) -> ComplexSpectrogram {
        let pad = fftSize / 2
        let frames = frameCount(forSampleCount: waveform.count)
        let half = fftSize / 2
        var real = [Float](repeating: 0, count: frequencyBins * frames)
        var imag = [Float](repeating: 0, count: frequencyBins * frames)

        waveform.withUnsafeBufferPointer { samples in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: fftSize) { frame in
                withUnsafeTemporaryAllocation(of: Float.self, capacity: half + 1) { binReal in
                    withUnsafeTemporaryAllocation(of: Float.self, capacity: half + 1) { binImag in
                        for t in 0 ..< frames {
                            // Frame t covers padded[t·hop ..< t·hop + fftSize], padded = pad zeros + waveform + pad zeros.
                            let start = t * hopLength - pad
                            for n in 0 ..< fftSize {
                                let index = start + n
                                let sample = index >= 0 && index < samples.count ? samples[index] : 0
                                frame[n] = sample * window[n]
                            }
                            fft.forward(UnsafeBufferPointer(frame), real: binReal, imag: binImag)
                            for f in 0 ..< frequencyBins {
                                real[f * frames + t] = binReal[f]
                                imag[f * frames + t] = binImag[f]
                            }
                        }
                    }
                }
            }
        }
        return ComplexSpectrogram(bins: frequencyBins, frames: frames, real: real, imag: imag)
    }

    /// The inverse transform: each frame's inverse FFT, windowed and overlap-added, divided by the overlapping windows'
    /// squared sum, with the centering pad trimmed. Bins past `spectrogram.bins` count as zero, as Spleeter pads them.
    ///
    /// - Returns: `hopLength × (frames − 1)` samples.
    public func inverse(_ spectrogram: ComplexSpectrogram) -> [Float] {
        let frames = spectrogram.frames
        let bins = min(spectrogram.bins, fftSize / 2 + 1)
        let half = fftSize / 2
        let fullLength = hopLength * (frames - 1) + fftSize
        var output = [Float](repeating: 0, count: fullLength)
        let windowSum = windowSquaredSum(frames: frames)

        output.withUnsafeMutableBufferPointer { output in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: fftSize) { frame in
                withUnsafeTemporaryAllocation(of: Float.self, capacity: half + 1) { binReal in
                    withUnsafeTemporaryAllocation(of: Float.self, capacity: half + 1) { binImag in
                        binReal.initialize(repeating: 0)
                        binImag.initialize(repeating: 0)
                        for t in 0 ..< frames {
                            for f in 0 ..< bins {
                                binReal[f] = spectrogram.real[f * frames + t]
                                binImag[f] = spectrogram.imag[f * frames + t]
                            }
                            fft.inverse(
                                real: UnsafeBufferPointer(binReal),
                                imag: UnsafeBufferPointer(binImag),
                                output: frame
                            )
                            let start = t * hopLength
                            for n in 0 ..< fftSize {
                                output[start + n] += frame[n] * window[n]
                            }
                        }
                    }
                }
            }
            for index in 0 ..< fullLength where windowSum[index] > 1e-8 {
                output[index] /= windowSum[index]
            }
        }

        let pad = fftSize / 2
        return Array(output[pad ..< fullLength - pad])
    }

    /// Each output sample's sum of squared windows over the frames that overlap it.
    private func windowSquaredSum(frames: Int) -> [Float] {
        var sum = [Float](repeating: 0, count: hopLength * (frames - 1) + fftSize)
        for t in 0 ..< frames {
            let start = t * hopLength
            for n in 0 ..< fftSize {
                sum[start + n] += window[n] * window[n]
            }
        }
        return sum
    }
}
