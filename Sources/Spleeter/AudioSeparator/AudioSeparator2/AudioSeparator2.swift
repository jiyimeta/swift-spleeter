import Foundation
import SpleeterCore

/// Audio separator that separates a music into 2 stems (vocals and accompaniment)
/// by using a pretrained Spleeter Core ML model.
public struct AudioSeparator2: AudioSeparatorProtocol, InternalAudioSeparatorProtocol {
    public typealias FloatArrayStems = Stems2<[Float]>
    public typealias StereoFloatArrayStems = Stems2<StereoValues<[Float]>>
    public typealias URLStems = Stems2<URL>
    typealias AudioFileStreamWriterStems = Stems2<AudioFileStreamWriter>

    static let stemLayout = SpleeterStemLayout.two

    let separator: SpleeterSeparator

    func stereoStems(_ values: [StereoValues<[Float]>]) -> Stems2<StereoValues<[Float]>> {
        Stems2(values)
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
