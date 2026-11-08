import Foundation
import SpleeterCore

/// Audio separator that separates a music into 5 stems (vocals, piano, drums, bass and others)
/// by using a pretrained Spleeter Core ML model.
public struct AudioSeparator5: AudioSeparatorProtocol, InternalAudioSeparatorProtocol {
    public typealias FloatArrayStems = Stems5<[Float]>
    public typealias StereoFloatArrayStems = Stems5<StereoValues<[Float]>>
    public typealias URLStems = Stems5<URL>
    typealias AudioFileStreamWriterStems = Stems5<AudioFileStreamWriter>

    static let stemLayout = SpleeterStemLayout.five

    let separator: SpleeterSeparator

    func stereoStems(_ values: [StereoValues<[Float]>]) -> Stems5<StereoValues<[Float]>> {
        Stems5(values)
    }

    func monaural(_ stems: Stems5<StereoValues<[Float]>>) -> Stems5<[Float]> {
        stems.mapStems(\.monaural)
    }

    func mapStems(
        _ urls: Stems5<URL>,
        _ transform: (URL) throws -> AudioFileStreamWriter
    ) rethrows -> Stems5<AudioFileStreamWriter> {
        try urls.mapStems(transform)
    }
}
