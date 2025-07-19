import CoreML

/// Audio separator that separates a music into 2 stems (vocals and instruments)
/// by using a pretrained Spleeter Core ML model.
public struct AudioSeparator2 {
    private let spleeter2Model: Spleeter2Model

    private let fftSize: Int
    private var hopLength: Int {
        fftSize / 4
    }

    private let frequencyLimit: Int

    private let clampingFrameCount: Int
    private var clampingLength: Int {
        clampingFrameCount * (hopLength - 1)
    }

    /// Initializes the audio separator with a compiled Spleeter2 Core ML model and STFT parameters.
    ///
    /// - Parameters:
    ///   - modelURL: URL pointing to the compiled Core ML model (.mlmodelc).
    ///   - fftSize: FFT size for Short-Time Fourier Transform (STFT). Must match the model's configuration.
    ///   - frequencyLimit: Number of frequency bins retained in the STFT output. Must match the model's configuration.
    ///   - clampingFrameCount: Number of STFT time frames processed per model inference.
    ///                         Must match the model's configuration.
    ///
    /// - Throws: An error if the model fails to load from the provided URL.
    public init(
        modelURL: URL,
        fftSize: Int = 4096,
        frequencyLimit: Int = 1024,
        clampingFrameCount: Int = 216
    ) throws {
        spleeter2Model = try Spleeter2Model(contentsOf: modelURL)
        self.fftSize = fftSize
        self.frequencyLimit = frequencyLimit
        self.clampingFrameCount = clampingFrameCount
    }

    /// Separates vocals and instruments from an audio file located at `inputURL`, and writes
    /// the separated stems to the corresponding URLs in `outputURLs`.
    ///
    /// This method reads the input file in chunks, processes each chunk asynchronously,
    /// and streams progress updates.
    ///
    /// - Parameters:
    ///   - inputURL: The URL of the audio file to separate.
    ///   - outputURLs: Destination URLs for the separated vocals and instruments audio files.
    ///
    /// - Returns: An `AsyncThrowingStream` that emits `Progress` updates and may throw errors.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func separate(
        from inputURL: URL,
        to outputURLs: Stems2<URL>
    ) -> AsyncThrowingStream<Progress, any Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let audioFile = try AudioFile(forReading: inputURL)

                    let sampleRate = audioFile.sampleRate

                    let vocalsWriter = try AudioFileStreamWriter(
                        to: outputURLs.vocals,
                        sampleRate: sampleRate,
                        channelCount: 1
                    )
                    let instrumentsWriter = try AudioFileStreamWriter(
                        to: outputURLs.instruments,
                        sampleRate: sampleRate,
                        channelCount: 1
                    )

                    let stride = stride(from: 0, to: audioFile.length, by: clampingLength)

                    continuation.yield(Progress(total: stride.underestimatedCount, current: 0))

                    for (index, position) in stride.enumerated() {
                        let range = position ..< min(position + clampingLength, audioFile.length)

                        let chunk = try audioFile.readStereoSamples(in: range)

                        let output = try await separate(chunk: chunk)

                        try vocalsWriter.append(samples: [output.vocals])
                        try instrumentsWriter.append(samples: [output.instruments])

                        continuation.yield(Progress(total: stride.underestimatedCount, current: index + 1))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Separates vocals and instruments from a stereo waveform.
    ///
    /// This method asynchronously processes the waveform in chunks and streams progress
    /// updates along with partial separation results.
    ///
    /// - Parameter waveform: The input stereo waveform represented as `StereoValues<[Float]>`.
    ///
    /// - Returns: An `AsyncThrowingStream` that emits tuples of optional separated stems and progress updates.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func separate(
        _ waveform: StereoValues<[Float]>
    ) -> AsyncThrowingStream<(Stems2<[Float]>?, Progress), any Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let length = max(waveform.left.count, waveform.right.count)
                    let stride = stride(from: 0, to: length, by: clampingLength)

                    continuation.yield((
                        nil,
                        Progress(total: stride.underestimatedCount, current: 0)
                    ))

                    for (index, position) in stride.enumerated() {
                        let range = position ..< min(position + clampingLength, length)

                        let chunk = waveform.mapChannels { Array($0[range]) }

                        let output = try await separate(chunk: chunk)

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
        }
    }

    /// Performs separation on a single chunk of stereo waveform using the pretrained Core ML model.
    ///
    /// - Parameter chunk: A chunk of stereo waveform samples.
    ///
    /// - Returns: Separated stems as monaural waveforms with lengths clamped to the input length.
    ///
    /// - Throws: An error if the model prediction or inverse STFT fails.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func separate(
        chunk: StereoValues<[Float]>
    ) async throws -> Stems2<[Float]> {
        let stft = try STFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)

        let spectrograms = try chunk.mapChannels {
            let spectrogram = try stft.forward(waveform: $0.paddedOrClamped(to: 1024 * 215).map(\.self))
            return SpectrogramTensor(spectrogram)
        }

        let magnitude = MLTensor(spectrograms.mapChannels(\.magnitude))

        let result = try await spleeter2Model.prediction(magnitude: magnitude)
        let masks = try Stems2(
            vocals: result.vocalsMaskTensor,
            instruments: result.instrumentsMaskTensor
        )

        return try await masks.asyncMapStems { mask in
            let complex = MLTensor(spectrograms.mapChannels(\.complex))
            let masked = mask * complex
            let monauralMasked = masked.mean(alongAxes: 0)
            let monauralMaskedWaveform = try await stft.inverse(
                Spectrogram(
                    realFrames: monauralMasked[..., 0].transposed().array2d(),
                    imagFrames: monauralMasked[..., 1].transposed().array2d()
                )
            )

            let maxLength = max(chunk.left.count, chunk.right.count)
            let clamped = monauralMaskedWaveform.prefix(maxLength)
            return Array(clamped)
        }
    }
}

extension AudioSeparator2 {
    /// Represents progress updates emitted during asynchronous separation operations.
    public struct Progress: Sendable {
        /// Total number of chunks or steps to be processed.
        public let total: Int

        /// Current completed chunk or step index.
        public let current: Int
    }
}
