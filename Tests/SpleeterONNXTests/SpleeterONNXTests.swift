import Foundation
@testable import OrtInference
import SpleeterCore
@testable import SpleeterONNX
import Testing
#if canImport(CoreML)
@testable import Spleeter
#endif

/// Where the exported models live: `SPLEETER_TEST_MODELS`, or `~/Library/Caches/swift-spleeter-ort/models`, holding
/// `Spleeter4Model-512.onnx`, `Spleeter4Model-512-fp16w.onnx`, `Spleeter4Model-216.onnx` (`convert-to-onnx.py -n 4
/// [-w float16] [-f 216]`) and `coreml-512/4stems/SpleeterModel.mlmodelc` (`convert-to-coreml.py -n 4 -f 512`).
enum TestModels {
    static let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["SPLEETER_TEST_MODELS"]
        ?? NSHomeDirectory() + "/Library/Caches/swift-spleeter-ort/models")

    static let onnx512 = directory.appendingPathComponent("Spleeter4Model-512.onnx")
    static let onnx512Float16Weights = directory.appendingPathComponent("Spleeter4Model-512-fp16w.onnx")
    static let onnx216 = directory.appendingPathComponent("Spleeter4Model-216.onnx")
    static let coreML512 = directory.appendingPathComponent("coreml-512/4stems/SpleeterModel.mlmodelc")
    /// The 0.3.1 release's 216-frame 4-stem Core ML model.
    static let coreML216 = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/Caches/swift-spleeter-test/Spleeter4Model.mlmodelc")

    static func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    /// The spleeter-pytorch submodule's example clip, 10.9 s of stereo music at 44.1 kHz.
    static let clip = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tools/spleeter-pytorch/audio-example.mp3")
}

/// `10 log10(‖reference‖² / ‖reference − estimate‖²)`.
func snr(_ estimate: [Float], _ reference: [Float]) -> Double {
    var signal = 0.0
    var noise = 0.0
    for index in 0 ..< min(estimate.count, reference.count) {
        signal += Double(reference[index]) * Double(reference[index])
        let difference = Double(reference[index]) - Double(estimate[index])
        noise += difference * difference
    }
    return 10 * log10(signal / max(noise, 1e-30))
}

/// Each stem's SNR of `estimate` against `reference`, both channels together.
func stemSNRs(_ estimate: [[[Float]]], _ reference: [[[Float]]]) -> [Double] {
    zip(estimate, reference).map { snr($0.0.flatMap(\.self), $0.1.flatMap(\.self)) }
}

func format(_ snrs: [Double]) -> String {
    zip(SpleeterStemLayout.four.stemNames, snrs).map { "\($0.0) \(String(format: "%.1f", $0.1)) dB" }
        .joined(separator: ", ")
}

/// The mean time of one `separate` call over `runs`, after one warm-up call.
func timePerWindow(_ separator: SpleeterSeparator, _ input: [[Float]], runs: Int = 3) throws -> Double {
    _ = try separator.separate(input)
    let start = Date()
    for _ in 0 ..< runs {
        _ = try separator.separate(input)
    }
    return Date().timeIntervalSince(start) / Double(runs)
}

@Suite(.enabled(if: TestModels.exists(TestModels.onnx512)))
struct OrtSessionTests {
    @Test
    func describesTheModel() throws {
        let session = try OrtSession(modelPath: TestModels.onnx512.path)
        #expect(session.inputNames == ["magnitude"])
        #expect(session.outputNames == SpleeterStemLayout.four.maskNames)
        #expect(session.inputShapes["magnitude"] == [2, 1024, 512])
    }

    @Test
    func runsWithNamedTensors() throws {
        let session = try OrtSession(modelPath: TestModels.onnx512.path)
        let input = OrtTensor(shape: [2, 1024, 512], values: [Float](repeating: 1, count: 2 * 1024 * 512))
        let outputs = try session.run(["magnitude": input], outputNames: ["bassMask"])
        #expect(Array(outputs.keys) == ["bassMask"])
        #expect(outputs["bassMask"]?.shape == [2, 1024, 512, 1])
        #expect(outputs["bassMask"]?.values.allSatisfy(\.isFinite) == true)
    }

    /// The bare inference time, without the STFTs, at a few thread counts.
    @Test
    func timing() throws {
        let input = OrtTensor(
            shape: [2, 1024, 512],
            values: (0 ..< 2 * 1024 * 512).map { Float($0 % 977) / 97.7 }
        )
        for threads in [nil, 1, 4, 8] as [Int?] {
            let session = try OrtSession(modelPath: TestModels.onnx512.path, options: OrtSessionOptions(intraOpThreads: threads))
            _ = try session.run(["magnitude": input])
            let start = Date()
            for _ in 0 ..< 3 {
                _ = try session.run(["magnitude": input])
            }
            let elapsed = Date().timeIntervalSince(start) / 3
            print("OrtSession.run, 512 frames, threads \(threads.map(String.init) ?? "default"): \(Int(elapsed * 1000)) ms")
        }
    }

    /// A provider the build lacks is skipped, not fatal; the CPU provider is always there.
    @Test
    func fallsBackFromAnUnavailableProvider() throws {
        let available = try OrtSession.availableProviders()
        print("ONNX Runtime providers in this build: \(available)")
        #expect(available.contains("CPUExecutionProvider"))
        let session = try OrtSession(
            modelPath: TestModels.onnx512.path,
            options: OrtSessionOptions(executionProviders: [.named("NoSuchProvider"), .xnnpack(threads: 4)])
        )
        print("used \(session.executionProviders), skipped \(session.skippedExecutionProviders)")
        #expect(session.executionProviders.last == "CPUExecutionProvider")
        #expect(session.skippedExecutionProviders.contains { $0.provider == "NoSuchProviderExecutionProvider" })
        #expect(session.executionProviders.contains("XnnpackExecutionProvider")
            == available.contains("XnnpackExecutionProvider"))
        let input = OrtTensor(shape: [2, 1024, 512], values: [Float](repeating: 1, count: 2 * 1024 * 512))
        #expect(try session.run(["magnitude": input], outputNames: ["vocalsMask"])["vocalsMask"]?.values.count
            == 2 * 1024 * 512)
    }

    /// XNNPACK against the CPU provider, where this build has it (XNNPACK wants the session pool at 1 thread and its
    /// own pool sized instead).
    @Test(.enabled(if: (try? OrtSession.availableProviders().contains("XnnpackExecutionProvider")) == true))
    func xnnpackTiming() throws {
        let input = OrtTensor(
            shape: [2, 1024, 512],
            values: (0 ..< 2 * 1024 * 512).map { Float($0 % 977) / 97.7 }
        )
        let cpu = try OrtSession(modelPath: TestModels.onnx512.path)
        let reference = try cpu.run(["magnitude": input])
        for threads in [4, 8] {
            let session = try OrtSession(
                modelPath: TestModels.onnx512.path,
                options: OrtSessionOptions(intraOpThreads: 1, executionProviders: [.xnnpack(threads: threads)])
            )
            #expect(session.executionProviders.first == "XnnpackExecutionProvider")
            let outputs = try session.run(["magnitude": input])
            let start = Date()
            for _ in 0 ..< 3 {
                _ = try session.run(["magnitude": input])
            }
            let elapsed = Date().timeIntervalSince(start) / 3
            let snrs = SpleeterStemLayout.four.maskNames.map { snr(outputs[$0]!.values, reference[$0]!.values) }
            print("XNNPACK \(threads) threads: \(Int(elapsed * 1000)) ms per 512 frames, masks vs CPU \(snrs.map { Int($0) }) dB")
        }
    }

    /// A provider the build has is appended and runs: Core ML in the macOS release, which lacks XNNPACK.
    @Test(.enabled(if: (try? OrtSession.availableProviders().contains("CoreMLExecutionProvider")) == true))
    func appendsAnAvailableProvider() throws {
        let input = OrtTensor(
            shape: [2, 1024, 512],
            values: (0 ..< 2 * 1024 * 512).map { Float($0 % 977) / 97.7 }
        )
        let reference = try OrtSession(modelPath: TestModels.onnx512.path).run(["magnitude": input])
        let session = try OrtSession(
            modelPath: TestModels.onnx512.path,
            options: OrtSessionOptions(executionProviders: [.named("CoreML", options: ["ModelFormat": "MLProgram"])])
        )
        #expect(session.executionProviders == ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        let outputs = try session.run(["magnitude": input])
        let start = Date()
        _ = try session.run(["magnitude": input])
        let elapsed = Date().timeIntervalSince(start)
        let snrs = SpleeterStemLayout.four.maskNames.map { snr(outputs[$0]!.values, reference[$0]!.values) }
        print("ORT CoreML EP: \(Int(elapsed * 1000)) ms per 512 frames, masks vs CPU \(snrs.map { Int($0) }) dB")
        #expect(snrs.allSatisfy { $0 > 30 })
    }

    /// A failure after CreateSession — reading the inputs and outputs — releases the session and the memory info: the
    /// half-built session never reaches deinit. (Only this test fails an init past CreateSession, so the count is its.)
    @Test
    func failingInitReleasesItsHandles() throws {
        struct Injected: Error {}
        let before = OrtSession.handlesReleasedByFailedInit
        #expect(throws: Injected.self) {
            try OrtSession(
                modelPath: TestModels.onnx512.path,
                options: OrtSessionOptions(),
                afterCreatingSession: { throw Injected() }
            )
        }
        #expect(OrtSession.handlesReleasedByFailedInit == before + 2)
    }

    @Test
    func reportsAMissingModel() {
        #expect(throws: OrtError.self) {
            try OrtSession(modelPath: "/nonexistent/model.onnx")
        }
    }

    @Test
    func rejectsAWrongShape() throws {
        let session = try OrtSession(modelPath: TestModels.onnx512.path)
        #expect(throws: OrtError.self) {
            try session.run(["magnitude": OrtTensor(shape: [2, 1024, 3], values: [Float](repeating: 0, count: 6144))])
        }
    }
}

@Suite(.enabled(if: TestModels.exists(TestModels.onnx512)))
struct OnnxSeparatorTests {
    @Test
    func stemsAddUpToTheMix() throws {
        let inference = try OnnxMaskInference(modelPath: TestModels.onnx512.path, stems: .four)
        #expect(inference.modelFrames == 512)
        let separator = SpleeterSeparator(stems: .four, inference: inference)
        let left = (0 ..< 300_000).map { Float(0.3 * sin(Double($0) * 0.031) + 0.2 * sin(Double($0) * 0.4)) }
        let right = left.map { $0 * 0.5 }
        let stems = try separator.separate([left, right])
        for (channel, input) in [left, right].enumerated() {
            let sum = (0 ..< input.count).map { index in stems.reduce(Float(0)) { $0 + $1[channel][index] } }
            #expect(snr(sum, input) > 30)
        }
    }

    /// A 216-frame window on the 512-frame model is padded to 512 frames, as the graph pads it itself.
    @Test(.enabled(if: TestModels.exists(TestModels.onnx216)))
    func shortWindowPadsToTheModelsFrames() throws {
        let input = (0 ..< 1024 * 215).map { Float(0.3 * sin(Double($0) * 0.02) + 0.1 * sin(Double($0) * 0.7)) }
        let fixed = try SpleeterSeparator(
            stems: .four,
            inference: OnnxMaskInference(modelPath: TestModels.onnx512.path, stems: .four),
            configuration: SpleeterConfiguration(framesPerWindow: 216)
        ).separate([input, input])
        let native = try SpleeterSeparator(
            stems: .four,
            inference: OnnxMaskInference(modelPath: TestModels.onnx216.path, stems: .four),
            configuration: SpleeterConfiguration(framesPerWindow: 216)
        ).separate([input, input])
        let snrs = stemSNRs(fixed, native)
        print("ONNX 512-frame model on a 216-frame window vs the 216-frame model: \(format(snrs))")
        #expect(snrs.allSatisfy { $0 > 60 })
    }
}

#if canImport(CoreML)
/// The real clip through each backend: the stems' agreement, and the time per window.
@Suite(.enabled(if: TestModels.exists(TestModels.onnx512) && TestModels.exists(TestModels.coreML512)))
struct BackendAgreementTests {
    static func clip() throws -> [[Float]] {
        let file = try AudioFile(forReading: TestModels.clip)
        #expect(file.sampleRate == 44100)
        let stereo = try file.readStereoSamples()
        return [stereo.left, stereo.right]
    }

    static func coreML(_ url: URL, frames: Int) throws -> SpleeterSeparator {
        try SpleeterSeparator(
            stems: .four,
            inference: CoreMLMaskInference(contentsOf: url, stems: .four),
            configuration: SpleeterConfiguration(framesPerWindow: frames),
            fft: AccelerateRealFFT(size: 4096)
        )
    }

    static func onnx(_ url: URL, frames: Int) throws -> SpleeterSeparator {
        try SpleeterSeparator(
            stems: .four,
            inference: OnnxMaskInference(modelPath: url.path, stems: .four),
            configuration: SpleeterConfiguration(framesPerWindow: frames)
        )
    }

    @Test
    func coreMLAndONNXAgreeAt512Frames() throws {
        let input = try Self.clip()
        let coreML = try Self.coreML(TestModels.coreML512, frames: 512)
        let onnx = try Self.onnx(TestModels.onnx512, frames: 512)
        #expect(coreML.windowLength == 1024 * 511 && input[0].count <= coreML.windowLength)
        let coreMLStems = try coreML.separate(input)
        let onnxStems = try onnx.separate(input)
        let snrs = stemSNRs(coreMLStems, onnxStems)
        print("512 frames, Core ML FP16 vs ONNX FP32: \(format(snrs))")
        #expect(snrs.allSatisfy { $0 > 25 })

        let seconds = Double(coreML.windowLength) / 44100
        let coreMLTime = try timePerWindow(coreML, input)
        let onnxTime = try timePerWindow(onnx, input)
        print(String(
            format: "512 frames (%.2f s of audio): Core ML %.0f ms, ONNX Runtime CPU %.0f ms per window",
            seconds, coreMLTime * 1000, onnxTime * 1000
        ))
    }

    @Test(.enabled(if: TestModels.exists(TestModels.onnx512Float16Weights)))
    func float16WeightsStayCloseToFloat32() throws {
        let input = try Self.clip()
        let float32 = try Self.onnx(TestModels.onnx512, frames: 512).separate(input)
        let float16Weights = try Self.onnx(TestModels.onnx512Float16Weights, frames: 512).separate(input)
        let snrs = stemSNRs(float16Weights, float32)
        print("512 frames, ONNX FP16-stored weights vs FP32: \(format(snrs))")
        #expect(snrs.allSatisfy { $0 > 40 })
    }

    @Test(.enabled(if: TestModels.exists(TestModels.onnx216) && TestModels.exists(TestModels.coreML216)))
    func coreMLAndONNXAgreeAt216Frames() throws {
        let input = try Self.clip().map { Array($0.prefix(1024 * 215)) }
        let coreML = try Self.coreML(TestModels.coreML216, frames: 216)
        let onnx = try Self.onnx(TestModels.onnx216, frames: 216)
        let snrs = stemSNRs(try coreML.separate(input), try onnx.separate(input))
        print("216 frames, Core ML FP16 (0.3.1) vs ONNX FP32: \(format(snrs))")
        #expect(snrs.allSatisfy { $0 > 25 })

        let coreMLTime = try timePerWindow(coreML, input)
        let onnxTime = try timePerWindow(onnx, input)
        print(String(
            format: "216 frames (%.2f s of audio): Core ML %.0f ms, ONNX Runtime CPU %.0f ms per window",
            Double(coreML.windowLength) / 44100, coreMLTime * 1000, onnxTime * 1000
        ))
    }
}
#endif
