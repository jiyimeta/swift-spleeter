import OrtInference
import SpleeterCore

/// What the ONNX backend rejects.
public enum OnnxMaskInferenceError: Error, Equatable {
    /// The model has no `magnitude` input, or it is not `2 × bins × frames`.
    case unexpectedInput([Int])

    /// The model's frame count is fixed and smaller than the window's.
    case windowTooLong(frames: Int, modelFrames: Int)
}

/// `SpleeterMaskInference` on a Spleeter ONNX model (`Tools/spleeter-pytorch/convert-to-onnx.py`), in ONNX Runtime.
///
/// The exported models take a fixed number of frames (512 by default). A shorter window is padded with zero frames to
/// it — what the separator does inside the graph anyway, as it partitions its input into 512-frame blocks — and the
/// masks are cut back to the window.
public struct OnnxMaskInference: SpleeterMaskInference {
    public let session: OrtSession
    public let stems: SpleeterStemLayout

    /// The frames the model takes, or nil when its frames axis is dynamic.
    public let modelFrames: Int?

    /// - Parameters:
    ///   - modelPath: The `.onnx` file.
    ///   - stems: The stems the model separates.
    ///   - options: How ONNX Runtime runs it.
    public init(modelPath: String, stems: SpleeterStemLayout, options: OrtSessionOptions = OrtSessionOptions()) throws {
        try self.init(session: OrtSession(modelPath: modelPath, options: options), stems: stems)
    }

    public init(session: OrtSession, stems: SpleeterStemLayout) throws {
        let shape = session.inputShapes[SpleeterStemLayout.magnitudeInputName] ?? []
        guard shape.count == 3 else { throw OnnxMaskInferenceError.unexpectedInput(shape) }
        self.session = session
        self.stems = stems
        modelFrames = shape[2] > 0 ? shape[2] : nil
    }

    public func masks(magnitude: [Float], channels: Int, bins: Int, frames: Int) throws -> [[Float]] {
        let runFrames = modelFrames ?? frames
        guard runFrames >= frames else {
            throw OnnxMaskInferenceError.windowTooLong(frames: frames, modelFrames: runFrames)
        }
        let input = runFrames == frames
            ? magnitude
            : Self.resizeFrames(magnitude, rows: channels * bins, from: frames, to: runFrames)
        let outputs = try session.run(
            [SpleeterStemLayout.magnitudeInputName: OrtTensor(shape: [channels, bins, runFrames], values: input)],
            outputNames: stems.maskNames
        )
        return stems.maskNames.map { name in
            let mask = outputs[name]!.values
            return runFrames == frames ? mask : Self.resizeFrames(mask, rows: channels * bins, from: runFrames, to: frames)
        }
    }

    /// Each `from`-frame row of `values` cut or zero-padded to `to` frames.
    static func resizeFrames(_ values: [Float], rows: Int, from: Int, to: Int) -> [Float] {
        var resized = [Float](repeating: 0, count: rows * to)
        let kept = min(from, to)
        values.withUnsafeBufferPointer { source in
            resized.withUnsafeMutableBufferPointer { destination in
                for row in 0 ..< rows {
                    for frame in 0 ..< kept {
                        destination[row * to + frame] = source[row * from + frame]
                    }
                }
            }
        }
        return resized
    }
}
