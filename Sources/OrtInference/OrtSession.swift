import COnnxRuntime
#if canImport(Glibc)
import Glibc
#elseif canImport(Android)
import Android
#elseif canImport(Darwin)
import Darwin
#endif

/// An error ONNX Runtime reported, or a misuse `OrtSession` caught before calling it.
public struct OrtError: Error, CustomStringConvertible, Sendable {
    /// ONNX Runtime's `OrtErrorCode`, or -1 for an error `OrtSession` raised itself.
    public let code: Int
    public let message: String

    public var description: String {
        "ONNX Runtime error \(code): \(message)"
    }
}

/// A float tensor: its shape, and its elements in row-major order.
public struct OrtTensor: Sendable, Equatable {
    public var shape: [Int]
    public var values: [Float]

    public init(shape: [Int], values: [Float]) {
        precondition(shape.reduce(1, *) == values.count, "the shape must cover the values")
        self.shape = shape
        self.values = values
    }
}

/// How a session runs.
public struct OrtSessionOptions: Sendable {
    /// The threads one inference may use; ONNX Runtime picks (one per physical core) when nil.
    public var intraOpThreads: Int?

    public init(intraOpThreads: Int? = nil) {
        self.intraOpThreads = intraOpThreads
    }
}

/// An ONNX model loaded into ONNX Runtime's CPU execution provider, run with named float tensors.
///
/// `run` may be called from several threads at once; ONNX Runtime documents `Run` as thread-safe.
public final class OrtSession: @unchecked Sendable {
    /// The model's inputs, in its order.
    public let inputNames: [String]

    /// The model's outputs, in its order.
    public let outputNames: [String]

    /// Each input's shape as the model declares it; a dynamic dimension is -1.
    public let inputShapes: [String: [Int]]

    private let session: OpaquePointer
    private let memoryInfo: OpaquePointer

    /// Loads the model at `modelPath`.
    public init(modelPath: String, options: OrtSessionOptions = OrtSessionOptions()) throws {
        let api = try OrtRuntime.api()
        let env = try OrtRuntime.env()

        var sessionOptions: OpaquePointer?
        try OrtRuntime.check(api.CreateSessionOptions!(&sessionOptions))
        defer { api.ReleaseSessionOptions!(sessionOptions) }
        if let threads = options.intraOpThreads {
            try OrtRuntime.check(api.SetIntraOpNumThreads!(sessionOptions, Int32(threads)))
        }
        try OrtRuntime.check(api.SetSessionGraphOptimizationLevel!(sessionOptions, ORT_ENABLE_ALL))

        var session: OpaquePointer?
        try OrtRuntime.check(api.CreateSession!(env, modelPath, sessionOptions, &session))
        guard let session else { throw OrtError(code: -1, message: "CreateSession returned no session") }

        var memoryInfo: OpaquePointer?
        do {
            try OrtRuntime.check(api.CreateCpuMemoryInfo!(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo))
        } catch {
            api.ReleaseSession!(session)
            throw error
        }

        self.session = session
        self.memoryInfo = memoryInfo!

        // From here on deinit releases the session and the memory info.
        var allocator: UnsafeMutablePointer<OrtAllocator>?
        try OrtRuntime.check(api.GetAllocatorWithDefaultOptions!(&allocator))
        func names(
            count: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
            name: (OpaquePointer?, Int, UnsafeMutablePointer<OrtAllocator>?,
                   UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> OpaquePointer?
        ) throws -> [String] {
            var total = 0
            try OrtRuntime.check(count(session, &total))
            return try (0 ..< total).map { index in
                var raw: UnsafeMutablePointer<CChar>?
                try OrtRuntime.check(name(session, index, allocator, &raw))
                defer { _ = api.AllocatorFree!(allocator, raw) }
                return String(cString: raw!)
            }
        }
        inputNames = try names(count: api.SessionGetInputCount!, name: api.SessionGetInputName!)
        outputNames = try names(count: api.SessionGetOutputCount!, name: api.SessionGetOutputName!)

        var shapes: [String: [Int]] = [:]
        for (index, name) in inputNames.enumerated() {
            var typeInfo: OpaquePointer?
            try OrtRuntime.check(api.SessionGetInputTypeInfo!(session, index, &typeInfo))
            defer { api.ReleaseTypeInfo!(typeInfo) }
            var tensorInfo: OpaquePointer?
            try OrtRuntime.check(api.CastTypeInfoToTensorInfo!(typeInfo, &tensorInfo))
            // A non-tensor input (a sequence, a map) has no shape to report.
            if let tensorInfo {
                shapes[name] = try OrtRuntime.shape(of: tensorInfo)
            }
        }
        inputShapes = shapes
    }

    deinit {
        if let api = try? OrtRuntime.api() {
            api.ReleaseSession!(session)
            api.ReleaseMemoryInfo!(memoryInfo)
        }
    }

    /// Runs the model.
    ///
    /// - Parameters:
    ///   - inputs: A tensor for each of the model's inputs, by name.
    ///   - outputNames: The outputs to compute; all of them when nil.
    /// - Returns: The outputs, by name.
    public func run(_ inputs: [String: OrtTensor], outputNames: [String]? = nil) throws -> [String: OrtTensor] {
        let api = try OrtRuntime.api()
        let requested = outputNames ?? self.outputNames
        let names = Array(inputs.keys)

        // ONNX Runtime reads the input tensors in place, so their buffers live until Run returns.
        var buffers: [UnsafeMutableBufferPointer<Float>] = []
        var values: [OpaquePointer?] = []
        defer {
            for value in values {
                api.ReleaseValue!(value)
            }
            for buffer in buffers {
                buffer.deallocate()
            }
        }
        for name in names {
            let tensor = inputs[name]!
            let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(tensor.values.count, 1))
            _ = buffer.initialize(from: tensor.values)
            buffers.append(buffer)
            var shape = tensor.shape.map(Int64.init)
            var value: OpaquePointer?
            try OrtRuntime.check(api.CreateTensorWithDataAsOrtValue!(
                memoryInfo, buffer.baseAddress, tensor.values.count * MemoryLayout<Float>.stride,
                &shape, shape.count, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value
            ))
            values.append(value)
        }

        var outputs = [OpaquePointer?](repeating: nil, count: requested.count)
        defer {
            for output in outputs {
                api.ReleaseValue!(output)
            }
        }
        try OrtRuntime.withCStrings(names) { inputNames in
            try OrtRuntime.withCStrings(requested) { outputNames in
                try OrtRuntime.check(api.Run!(
                    session, nil, inputNames, values, values.count, outputNames, requested.count, &outputs
                ))
            }
        }

        var result: [String: OrtTensor] = [:]
        for (name, output) in zip(requested, outputs) {
            guard let output else { throw OrtError(code: -1, message: "no value for output \(name)") }
            var info: OpaquePointer?
            try OrtRuntime.check(api.GetTensorTypeAndShape!(output, &info))
            defer { api.ReleaseTensorTypeAndShapeInfo!(info) }
            var type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
            try OrtRuntime.check(api.GetTensorElementType!(info, &type))
            guard type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT else {
                throw OrtError(code: -1, message: "output \(name) is not a float tensor (type \(type.rawValue))")
            }
            let shape = try OrtRuntime.shape(of: info!)
            let count = shape.reduce(1, *)
            var data: UnsafeMutableRawPointer?
            try OrtRuntime.check(api.GetTensorMutableData!(output, &data))
            let elements = UnsafeBufferPointer(start: data?.assumingMemoryBound(to: Float.self), count: count)
            result[name] = OrtTensor(shape: shape, values: Array(elements))
        }
        return result
    }
}

/// The process-wide ONNX Runtime API table and environment.
enum OrtRuntime {
    private static let apiTable: UnsafePointer<OrtApi>? = OrtGetApiBase()?.pointee.GetApi?(UInt32(ORT_API_VERSION))

    private static let environment: Result<OpaquePointer, OrtError> = {
        do {
            let api = try api()
            var env: OpaquePointer?
            try check(api.CreateEnv!(ORT_LOGGING_LEVEL_WARNING, "OrtInference", &env))
            return .success(env!)
        } catch let error as OrtError {
            return .failure(error)
        } catch {
            return .failure(OrtError(code: -1, message: "\(error)"))
        }
    }()

    /// The API table of the ONNX Runtime this was linked against, at the headers' API version.
    static func api() throws -> OrtApi {
        guard let apiTable else {
            throw OrtError(
                code: -1,
                message: "the linked ONNX Runtime does not provide API version \(ORT_API_VERSION) (older than 1.31?)"
            )
        }
        return apiTable.pointee
    }

    static func env() throws -> OpaquePointer {
        try environment.get()
    }

    static func check(_ status: OpaquePointer?) throws {
        guard let status else { return }
        let api = apiTable!.pointee
        let message = api.GetErrorMessage!(status).map { String(cString: $0) } ?? "(no message)"
        let code = Int(api.GetErrorCode!(status).rawValue)
        api.ReleaseStatus!(status)
        throw OrtError(code: code, message: message)
    }

    static func shape(of info: OpaquePointer) throws -> [Int] {
        let api = try api()
        var count = 0
        try check(api.GetDimensionsCount!(info, &count))
        var dimensions = [Int64](repeating: 0, count: count)
        try check(api.GetDimensions!(info, &dimensions, count))
        return dimensions.map(Int.init)
    }

    /// `body` with `strings` as an array of C strings.
    static func withCStrings<Result>(
        _ strings: [String],
        _ body: ([UnsafePointer<CChar>?]) throws -> Result
    ) rethrows -> Result {
        let copies = strings.map { strdup($0) }
        defer {
            for copy in copies {
                free(copy)
            }
        }
        return try body(copies.map { UnsafePointer($0) })
    }
}
