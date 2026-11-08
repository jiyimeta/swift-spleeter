import CoreML
import SpleeterCore

/// Errors thrown by STFT during FFT setup.
public enum STFTError: Error {
    /// Failed to create the FFT setup object.
    case failedToCreateFFT

    /// The base address of the input sample buffer is nil.
    case noBaseAddress
}

/// Errors thrown reading a Spleeter Core ML model's output.
public enum SpleeterModelError: Error {
    case invalidFeatureName(_ featureName: String)
}

/// `SpleeterMaskInference` on a compiled Spleeter Core ML model (`Tools/spleeter-pytorch/convert-to-coreml.py`).
struct CoreMLMaskInference: SpleeterMaskInference, @unchecked Sendable {
    // MLModel's predictions are safe to make from any thread.
    let model: MLModel
    let maskNames: [String]

    init(model: MLModel, maskNames: [String]) {
        self.model = model
        self.maskNames = maskNames
    }

    init(contentsOf modelURL: URL, stems: SpleeterStemLayout) throws {
        try self.init(model: MLModel(contentsOf: modelURL), maskNames: stems.maskNames)
    }

    /// The frames the model's `magnitude` input takes (`2 × bins × frames`), when its shape is fixed.
    var modelFrames: Int? {
        let input = model.modelDescription.inputDescriptionsByName[SpleeterStemLayout.magnitudeInputName]
        guard let shape = input?.multiArrayConstraint?.shape, shape.count == 3 else { return nil }
        return shape[2].intValue
    }

    func masks(magnitude: [Float], channels: Int, bins: Int, frames: Int) throws -> [[Float]] {
        let input = try MLMultiArray(
            shape: [channels, bins, frames].map { NSNumber(value: $0) },
            dataType: .float32
        )
        magnitude.withUnsafeBufferPointer { source in
            let destination = input.dataPointer.bindMemory(to: Float.self, capacity: magnitude.count)
            destination.update(from: source.baseAddress!, count: magnitude.count)
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            SpleeterStemLayout.magnitudeInputName: MLFeatureValue(multiArray: input),
        ])
        let output = try model.prediction(from: features)
        return try maskNames.map { name in
            guard let multiArray = output.featureValue(for: name)?.multiArrayValue else {
                throw SpleeterModelError.invalidFeatureName(name)
            }
            return MLShapedArray<Float>(multiArray).scalars
        }
    }
}
