import CoreML

protocol InternalAudioSeparatorProtocol: AudioSeparatorProtocol {
    associatedtype SpleeterModel: SpleeterModelProtocol
    associatedtype AudioFileStreamWriterStems: StemsProtocol
        where AudioFileStreamWriterStems.Value == AudioFileStreamWriter

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    typealias MLTensorStems = SpleeterModel.MLTensorStems

    var model: SpleeterModel { get }
    var fftSize: Int { get }
    var frequencyLimit: Int { get }
    var clampingFrameCount: Int { get }

    init(
        model: SpleeterModel,
        fftSize: Int,
        frequencyLimit: Int,
        clampingFrameCount: Int
    ) throws

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func asyncMapStereoStems(
        _ tensors: MLTensorStems,
        _ transform: (MLTensor) async throws -> StereoValues<[Float]>
    ) async rethrows -> StereoFloatArrayStems

    /// Each stem as the mean of its two channels.
    func monaural(_ stems: StereoFloatArrayStems) -> FloatArrayStems

    func mapStems(
        _ urls: URLStems,
        _ transform: (URL) throws -> AudioFileStreamWriter
    ) rethrows -> AudioFileStreamWriterStems
}

extension InternalAudioSeparatorProtocol {
    var hopLength: Int {
        fftSize / 4
    }

    var clampingLength: Int {
        hopLength * (clampingFrameCount - 1)
    }

    // swiftlint:disable:next missing_docs
    public init(
        modelURL: URL,
        fftSize: Int,
        frequencyLimit: Int,
        clampingFrameCount: Int
    ) throws {
        try self.init(
            model: SpleeterModel(contentsOf: modelURL),
            fftSize: fftSize,
            frequencyLimit: frequencyLimit,
            clampingFrameCount: clampingFrameCount
        )
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    // swiftlint:disable:next missing_docs
    public func separate(
        from inputURL: URL,
        to outputURLs: URLStems
    ) -> AsyncThrowingStream<Progress, any Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let audioFile = try AudioFile(forReading: inputURL)

                    let sampleRate = audioFile.sampleRate

                    let fileWriters = try mapStems(outputURLs) {
                        try AudioFileStreamWriter(
                            to: $0,
                            sampleRate: sampleRate,
                            channelCount: 1
                        )
                    }

                    let stride = stride(from: 0, to: audioFile.length, by: clampingLength)

                    continuation.yield(Progress(total: stride.underestimatedCount, current: 0))

                    for (index, position) in stride.enumerated() {
                        let range = position ..< min(position + clampingLength, audioFile.length)

                        let chunk = try audioFile.readStereoSamples(in: range)

                        let output = try await separate(chunk: chunk)

                        for (writer, samples) in zip(fileWriters.values, output.values) {
                            try writer.append(samples: [samples])
                        }

                        continuation.yield(Progress(total: stride.underestimatedCount, current: index + 1))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    // swiftlint:disable:next missing_docs
    public func separate(
        _ waveform: StereoValues<[Float]>
    ) -> AsyncThrowingStream<(FloatArrayStems?, Progress), any Error> {
        chunkedStream(waveform) { try await separate(chunk: $0) }
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    // swiftlint:disable:next missing_docs
    public func separateStereo(
        _ waveform: StereoValues<[Float]>
    ) -> AsyncThrowingStream<(StereoFloatArrayStems?, Progress), any Error> {
        chunkedStream(waveform) { try await separateStereo(chunk: $0) }
    }

    /// Runs `separateChunk` over `waveform` one model window at a time, streaming each window's stems with the
    /// progress. Ending the stream early (the consumer stops iterating or is cancelled) stops the work after the
    /// window in progress.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func chunkedStream<Output>(
        _ waveform: StereoValues<[Float]>,
        separateChunk: @escaping (StereoValues<[Float]>) async throws -> Output
    ) -> AsyncThrowingStream<(Output?, Progress), any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let length = max(waveform.left.count, waveform.right.count)
                    let stride = stride(from: 0, to: length, by: clampingLength)

                    continuation.yield((
                        nil,
                        Progress(total: stride.underestimatedCount, current: 0)
                    ))

                    for (index, position) in stride.enumerated() {
                        try Task.checkCancellation()

                        let range = position ..< min(position + clampingLength, length)

                        let chunk = waveform.mapChannels { Array($0[range]) }

                        let output = try await separateChunk(chunk)

                        continuation.yield((
                            output,
                            Progress(total: stride.underestimatedCount, current: index + 1)
                        ))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Performs separation on a single chunk of stereo waveform, returning each stem as the mean of its two channels.
    ///
    /// - Parameter chunk: A chunk of stereo waveform samples.
    ///
    /// - Returns: Separated stems as monaural waveforms with lengths clamped to the input length.
    ///
    /// - Throws: An error if the model prediction or inverse STFT fails.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func separate(
        chunk: StereoValues<[Float]>
    ) async throws -> FloatArrayStems {
        try await monaural(separateStereo(chunk: chunk))
    }

    /// Performs separation on a single chunk of stereo waveform using the pretrained Core ML model.
    ///
    /// The model's masks are turned into Spleeter's ratio masks — each stem's share of the summed squared masks, per
    /// bin and channel — so the stems add up to the input, as Spleeter's own separation does.
    ///
    /// - Parameter chunk: A chunk of stereo waveform samples.
    ///
    /// - Returns: Separated stems as stereo waveforms with lengths clamped to the input length.
    ///
    /// - Throws: An error if the model prediction or inverse STFT fails.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func separateStereo(
        chunk: StereoValues<[Float]>
    ) async throws -> StereoFloatArrayStems {
        let stft = try STFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)

        let spectrograms = try chunk.mapChannels {
            let spectrogram = try stft.forward(waveform: $0.paddedOrClamped(to: 1024 * 215).map(\.self))
            return SpectrogramTensor(spectrogram)
        }

        let magnitude = MLTensor(spectrograms.mapChannels(\.magnitude))

        let masks = try await model.prediction(magnitude: magnitude)

        let squaredMasks = masks.values.map { $0 * $0 }
        let squaredSum = squaredMasks.dropFirst().reduce(squaredMasks[0], +) + Self.maskEpsilon
        let stemCount = Float(squaredMasks.count)

        let complex = MLTensor(spectrograms.mapChannels(\.complex))
        let maxLength = max(chunk.left.count, chunk.right.count)

        return try await asyncMapStereoStems(masks) { mask in
            let ratioMask = (mask * mask + Self.maskEpsilon / stemCount) / squaredSum
            let masked = ratioMask * complex

            func waveform(ofChannel channel: Int) async throws -> [Float] {
                let channelMasked = masked[channel]
                let waveform = try await stft.inverse(
                    Spectrogram(
                        realFrames: channelMasked[..., 0].transposed().array2d(),
                        imagFrames: channelMasked[..., 1].transposed().array2d()
                    )
                )
                return Array(waveform.prefix(maxLength))
            }

            return try await StereoValues(left: waveform(ofChannel: 0), right: waveform(ofChannel: 1))
        }
    }
}

extension InternalAudioSeparatorProtocol {
    /// Spleeter's ratio-mask epsilon (`separator.py`): keeps the division finite where every mask is zero.
    fileprivate static var maskEpsilon: Float {
        1e-10
    }
}
