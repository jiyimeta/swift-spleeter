import CoreML

/// Audio separator that separates a music into 2 stems (vocals and accompaniment)
/// by using a pretrained Spleeter Core ML model.
public struct AudioSeparator2: AudioSeparatorProtocol, InternalAudioSeparatorProtocol {
    typealias SpleeterModel = Spleeter2Model

    public typealias FloatArrayStems = Stems2<[Float]>
    public typealias StereoFloatArrayStems = Stems2<StereoValues<[Float]>>
    public typealias URLStems = Stems2<URL>
    typealias AudioFileStreamWriterStems = Stems2<AudioFileStreamWriter>

    let model: Spleeter2Model
    let fftSize: Int
    let frequencyLimit: Int
    let clampingFrameCount: Int

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func asyncMapStereoStems(
        _ tensors: MLTensorStems,
        _ transform: (MLTensor) async throws -> StereoValues<[Float]>
    ) async rethrows -> Stems2<StereoValues<[Float]>> {
        try await tensors.asyncMapStems(transform)
    }

    func monaural(_ stems: Stems2<StereoValues<[Float]>>) -> Stems2<[Float]> {
        stems.mapStems(\.monaural)
    }

    func mapStems(
        _ urls: Stems2<URL>,
        _ transform: (URL) throws -> AudioFileStreamWriter
    ) rethrows -> Stems2<AudioFileStreamWriter> {
        try urls.mapStems(transform)
    }
}
