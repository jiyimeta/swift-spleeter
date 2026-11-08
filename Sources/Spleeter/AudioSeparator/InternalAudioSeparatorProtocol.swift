import Foundation
import SpleeterCore

protocol InternalAudioSeparatorProtocol: AudioSeparatorProtocol {
    associatedtype AudioFileStreamWriterStems: StemsProtocol
        where AudioFileStreamWriterStems.Value == AudioFileStreamWriter

    /// The stems the model separates, in its output order.
    static var stemLayout: SpleeterStemLayout { get }

    /// One window end to end, on the Core ML model.
    var separator: SpleeterSeparator { get }

    init(separator: SpleeterSeparator)

    /// The stems from the separator's output, in `stemLayout` order.
    func stereoStems(_ values: [StereoValues<[Float]>]) -> StereoFloatArrayStems

    /// Each stem as the mean of its two channels.
    func monaural(_ stems: StereoFloatArrayStems) -> FloatArrayStems

    func mapStems(
        _ urls: URLStems,
        _ transform: (URL) throws -> AudioFileStreamWriter
    ) rethrows -> AudioFileStreamWriterStems
}

extension InternalAudioSeparatorProtocol {
    /// The samples one model prediction covers.
    var clampingLength: Int {
        separator.windowLength
    }

    // swiftlint:disable:next missing_docs
    public var windowLength: Int {
        separator.windowLength
    }

    // swiftlint:disable:next missing_docs
    public init(
        modelURL: URL,
        fftSize: Int,
        frequencyLimit: Int,
        clampingFrameCount: Int?
    ) throws {
        let inference = try CoreMLMaskInference(contentsOf: modelURL, stems: Self.stemLayout)
        try self.init(separator: SpleeterSeparator(
            stems: Self.stemLayout,
            inference: inference,
            configuration: SpleeterConfiguration(
                fftSize: fftSize,
                hopLength: fftSize / 4,
                frequencyBins: frequencyLimit,
                framesPerWindow: clampingFrameCount ?? inference.modelFrames ?? 216
            ),
            fft: AccelerateRealFFT(size: fftSize)
        ))
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
    /// - Throws: An error if the model prediction fails.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    private func separateStereo(
        chunk: StereoValues<[Float]>
    ) async throws -> StereoFloatArrayStems {
        let stems = try separator.separate([chunk.left, chunk.right])
        return stereoStems(stems.map { StereoValues(left: $0[0], right: $0[1]) })
    }
}
