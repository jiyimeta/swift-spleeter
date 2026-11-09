import CoreML
import Foundation
@testable import Spleeter
import Testing

/// What the separators return: stems that add up to the input — Spleeter's ratio masks share every bin between the
/// stems — at the input's level, in stereo when asked.
struct AudioSeparatorTests {
    /// The 2-stem model the example app bundles.
    static let twoStemModel = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Example/SpleeterExample/Spleeter2Model.mlmodelc")

    /// The 4-stem model from the 0.3.1 release, when a copy sits in `~/Library/Caches/swift-spleeter-test/`.
    static let fourStemModel = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/Caches/swift-spleeter-test/Spleeter4Model.mlmodelc")

    /// A few seconds of something music-like: two tones and a decaying click train per channel.
    static func signal(seconds: Double, channelSeed: Double) -> [Float] {
        let count = Int(seconds * 44100)
        return (0 ..< count).map { index in
            let time = Double(index) / 44100
            let tone = 0.3 * sin(2 * .pi * (220 + channelSeed) * time) + 0.2 * sin(2 * .pi * (331 * channelSeed) * time)
            let click = exp(-Double(index % 11025) / 400) * 0.3
            return Float(tone + click)
        }
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    static func collect<Stems>(
        _ stream: AsyncThrowingStream<(Stems?, Spleeter.Progress), any Error>
    ) async throws -> [Stems] {
        var chunks: [Stems] = []
        for try await (stems, _) in stream {
            if let stems {
                chunks.append(stems)
            }
        }
        return chunks
    }

    static func relativeError(_ sum: [Float], _ reference: [Float]) -> Double {
        let count = min(sum.count, reference.count)
        var error = 0.0
        var energy = 0.0
        for index in 0 ..< count {
            error += Double((sum[index] - reference[index]) * (sum[index] - reference[index]))
            energy += Double(reference[index] * reference[index])
        }
        return (error / energy).squareRoot()
    }

    @Test
    func twoStemsAddUpToTheMonoMix() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let left = Self.signal(seconds: 4, channelSeed: 1)
        let right = Self.signal(seconds: 4, channelSeed: 1.5)
        let separator = try AudioSeparator2(modelURL: Self.twoStemModel)
        let chunks = try await Self.collect(separator.separate(StereoValues(left: left, right: right)))
        let vocals = chunks.flatMap(\.vocals)
        let accompaniment = chunks.flatMap(\.accompaniment)
        let sum = zip(vocals, accompaniment).map { $0 + $1 }
        let mono = zip(left, right).map { ($0 + $1) / 2 }
        #expect(vocals.count == left.count)
        #expect(Self.relativeError(sum, mono) < 0.02)
    }

    @Test
    func stereoStemsKeepEachChannel() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let left = Self.signal(seconds: 4, channelSeed: 1)
        let silent = [Float](repeating: 0, count: left.count)
        let separator = try AudioSeparator2(modelURL: Self.twoStemModel)
        let chunks = try await Self.collect(separator.separateStereo(StereoValues(left: left, right: silent)))
        let vocals = chunks.flatMap(\.vocals.left)
        let accompaniment = chunks.flatMap(\.accompaniment.left)
        let rightPeak = chunks.flatMap { $0.values.flatMap(\.right) }.map(abs).max() ?? 0
        #expect(vocals.count == left.count)
        #expect(Self.relativeError(zip(vocals, accompaniment).map { $0 + $1 }, left) < 0.02)
        #expect(rightPeak < 1e-4)
    }

    /// The models emit each stem's raw mask — the magnitude it estimates for that stem — and the separator turns them
    /// into ratio masks in FP32. A model that already emitted ratio masks (the 0.2.0 release's) was normalized twice,
    /// and squared magnitudes in FP16, where anything above 256 overflows.
    @Test
    func modelEmitsRawMasks() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let model = try CoreMLMaskInference(contentsOf: Self.twoStemModel, stems: .two)
        // 100 squared stays within FP16, so a ratio-mask model would answer at most 1 here rather than overflow.
        let magnitude = [Float](repeating: 100, count: 2 * 1024 * 216)
        let masks = try model.masks(magnitude: magnitude, channels: 2, bins: 1024, frames: 216)
        let vocals = masks[0]
        let accompaniment = masks[1]
        #expect(vocals.count == magnitude.count && accompaniment.count == magnitude.count)
        let peak = max(vocals.max() ?? 0, accompaniment.max() ?? 0)
        // A ratio mask tops out at 1 (1.001 after FP16 rounding); a raw one here is on the magnitude's scale.
        #expect(peak > 10)
        #expect(peak <= 100 * 1.01)
    }

    /// A loud bass tone — every bin it fills is far above 256 — stays in the accompaniment instead of being thrown out
    /// of it by an FP16 overflow.
    @Test
    func loudBassToneStaysInTheAccompaniment() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let tone = (0 ..< 4 * 44100).map { Float(0.9 * sin(2 * .pi * 80 * Double($0) / 44100)) }
        let separator = try AudioSeparator2(modelURL: Self.twoStemModel)
        let chunks = try await Self.collect(separator.separate(StereoValues(left: tone, right: tone)))
        let accompaniment = chunks.flatMap(\.accompaniment)
        // `#expect` treats a key path argument as a throwing call, so the closure form stays.
        // swiftformat:disable:next preferKeyPath
        #expect(accompaniment.allSatisfy { $0.isFinite })
        #expect(Self.relativeError(accompaniment, tone) < 0.05)
    }

    @Test(.enabled(if: FileManager.default.fileExists(atPath: AudioSeparatorTests.fourStemModel.path)))
    func fourStemsAddUpToTheMix() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let left = Self.signal(seconds: 4, channelSeed: 1)
        let right = Self.signal(seconds: 4, channelSeed: 1.5)
        let separator = try AudioSeparator4(modelURL: Self.fourStemModel)
        let chunks = try await Self.collect(separator.separateStereo(StereoValues(left: left, right: right)))
        for (channel, input) in [(\StereoValues<[Float]>.left, left), (\.right, right)] {
            let stems = (0 ..< 4).map { stem in chunks.flatMap { $0.values[stem][keyPath: channel] } }
            let sum = (0 ..< input.count).map { index in stems.reduce(Float(0)) { $0 + $1[index] } }
            #expect(Self.relativeError(sum, input) < 0.02)
        }
    }

    /// The 0.4.0 release's 512-frame models, `convert-to-coreml.py -f 512`, compiled under
    /// `~/Library/Caches/swift-spleeter-ort/models/coreml-512/{2,4,5}stems/`.
    static func fiveTwelveFrameModel(stems: Int) -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Caches/swift-spleeter-ort/models/coreml-512/\(stems)stems/SpleeterModel.mlmodelc")
    }

    /// A 512-frame model gives the separator its window without being told, and its stems add up to the mix across
    /// windows: 14 s is one full 11.9 s window and a short one.
    @Test(.enabled(if: [2, 4, 5].allSatisfy {
        FileManager.default.fileExists(atPath: AudioSeparatorTests.fiveTwelveFrameModel(stems: $0).path)
    }))
    func fiveTwelveFrameModelsSetTheirWindow() async throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) else { return }
        let left = Self.signal(seconds: 14, channelSeed: 1)
        let right = Self.signal(seconds: 14, channelSeed: 1.5)
        let input = StereoValues(left: left, right: right)

        func check<Separator: AudioSeparatorProtocol>(_ separator: Separator, stems: Int) async throws {
            #expect(separator.windowLength == 1024 * 511)
            let chunks = try await Self.collect(separator.separateStereo(input))
            #expect(chunks.count == 2)
            for (channel, reference) in [(\StereoValues<[Float]>.left, left), (\.right, right)] {
                let stemSamples = (0 ..< stems).map { stem in chunks.flatMap { $0.values[stem][keyPath: channel] } }
                #expect(stemSamples.allSatisfy { $0.count == reference.count })
                let sum = (0 ..< reference.count).map { index in stemSamples.reduce(Float(0)) { $0 + $1[index] } }
                #expect(Self.relativeError(sum, reference) < 0.02, "\(stems) stems")
            }
        }
        try await check(AudioSeparator2(modelURL: Self.fiveTwelveFrameModel(stems: 2)), stems: 2)
        try await check(AudioSeparator4(modelURL: Self.fiveTwelveFrameModel(stems: 4)), stems: 4)
        try await check(AudioSeparator5(modelURL: Self.fiveTwelveFrameModel(stems: 5)), stems: 5)
    }
}
