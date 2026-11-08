import Foundation
import SpleeterCore

/// Audio separator that separates a music into 4 stems (vocals, drums, bass and others)
/// by using a pretrained Spleeter Core ML model.
public struct AudioSeparator4: AudioSeparatorProtocol, InternalAudioSeparatorProtocol {
    public typealias FloatArrayStems = Stems4<[Float]>
    public typealias StereoFloatArrayStems = Stems4<StereoValues<[Float]>>
    public typealias URLStems = Stems4<URL>
    typealias AudioFileStreamWriterStems = Stems4<AudioFileStreamWriter>

    static let stemLayout = SpleeterStemLayout.four

    let separator: SpleeterSeparator

    func stereoStems(_ values: [StereoValues<[Float]>]) -> Stems4<StereoValues<[Float]>> {
        Stems4(values)
    }

    func monaural(_ stems: Stems4<StereoValues<[Float]>>) -> Stems4<[Float]> {
        stems.mapStems(\.monaural)
    }

    func mapStems(
        _ urls: Stems4<URL>,
        _ transform: (URL) throws -> AudioFileStreamWriter
    ) rethrows -> Stems4<AudioFileStreamWriter> {
        try urls.mapStems(transform)
    }
}
