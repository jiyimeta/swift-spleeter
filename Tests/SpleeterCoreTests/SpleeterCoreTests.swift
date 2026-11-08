import Foundation
@testable import SpleeterCore
import Testing

/// A deterministic pseudo-random signal in [-0.5, 0.5).
func noise(count: Int, seed: UInt64) -> [Float] {
    var state = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return (0 ..< count).map { _ in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(state >> 40) / Float(1 << 24) - 0.5
    }
}

/// A few tones below the 1024 bins' 11 kHz, so the kept bins hold the whole signal.
func tones(count: Int, seed: Double) -> [Float] {
    (0 ..< count).map { index in
        let time = Double(index) / 44100
        return Float(0.3 * sin(2 * .pi * (220 + seed) * time) + 0.2 * sin(2 * .pi * 1830 * seed * time)
            + 0.1 * sin(2 * .pi * 7000 * time))
    }
}

/// `‖a − b‖ / ‖b‖`.
func relativeError(_ a: [Float], _ b: [Float]) -> Double {
    precondition(a.count == b.count)
    var error = 0.0
    var energy = 0.0
    for index in a.indices {
        let difference = Double(a[index]) - Double(b[index])
        error += difference * difference
        energy += Double(b[index]) * Double(b[index])
    }
    return (error / max(energy, .leastNonzeroMagnitude)).squareRoot()
}

struct PortableRealFFTTests {
    /// The forward transform is the plain DFT, bins 0 … size / 2.
    @Test(arguments: [4, 16, 256])
    func forwardMatchesTheDFT(size: Int) {
        let fft = PortableRealFFT(size: size)
        let input = noise(count: size, seed: UInt64(size))
        var real = [Float](repeating: 0, count: size / 2 + 1)
        var imag = [Float](repeating: 0, count: size / 2 + 1)
        input.withUnsafeBufferPointer { input in
            real.withUnsafeMutableBufferPointer { real in
                imag.withUnsafeMutableBufferPointer { imag in
                    fft.forward(input, real: real, imag: imag)
                }
            }
        }
        for bin in 0 ... size / 2 {
            var expectedReal = 0.0
            var expectedImag = 0.0
            for (n, sample) in input.enumerated() {
                let angle = -2 * Double.pi * Double(bin * n) / Double(size)
                expectedReal += Double(sample) * cos(angle)
                expectedImag += Double(sample) * sin(angle)
            }
            #expect(abs(Double(real[bin]) - expectedReal) < 1e-4, "real, bin \(bin)")
            #expect(abs(Double(imag[bin]) - expectedImag) < 1e-4, "imag, bin \(bin)")
        }
    }

    @Test(arguments: [4, 64, 4096])
    func inverseUndoesForward(size: Int) {
        let fft = PortableRealFFT(size: size)
        let input = noise(count: size, seed: 7)
        var real = [Float](repeating: 0, count: size / 2 + 1)
        var imag = [Float](repeating: 0, count: size / 2 + 1)
        var output = [Float](repeating: 0, count: size)
        input.withUnsafeBufferPointer { input in
            real.withUnsafeMutableBufferPointer { real in
                imag.withUnsafeMutableBufferPointer { imag in
                    fft.forward(input, real: real, imag: imag)
                }
            }
        }
        real.withUnsafeBufferPointer { real in
            imag.withUnsafeBufferPointer { imag in
                output.withUnsafeMutableBufferPointer { output in
                    fft.inverse(real: real, imag: imag, output: output)
                }
            }
        }
        #expect(relativeError(output, input) < 1e-5)
    }
}

struct SpleeterSTFTTests {
    @Test
    func windowFramesAndLength() {
        let stft = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 1024)
        let spectrogram = stft.forward([Float](repeating: 0, count: 1024 * 215))
        #expect(spectrogram.frames == 216)
        #expect(spectrogram.bins == 1024)
        #expect(stft.inverse(spectrogram).count == 1024 * 215)
    }

    /// Every bin kept, the inverse gives the signal back at its own level.
    @Test
    func inverseReturnsTheSignal() {
        let stft = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 2049)
        let signal = (0 ..< 1024 * 40).map { Float(sin(Double($0) * 0.01) * 0.5 + sin(Double($0) * 0.3) * 0.2) }
        let reconstructed = stft.inverse(stft.forward(signal))
        #expect(reconstructed.count == signal.count)
        let worst = zip(signal, reconstructed).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 1e-4)
    }
}

/// Masks every stem alike: each stem gets `1 / n` of every bin.
struct UniformInference: SpleeterMaskInference {
    let stems: Int

    func masks(magnitude: [Float], channels: Int, bins: Int, frames: Int) throws -> [[Float]] {
        #expect(magnitude.count == channels * bins * frames)
        return Array(repeating: magnitude, count: stems)
    }
}

/// Gives the first stem everything.
struct FirstStemInference: SpleeterMaskInference {
    let stems: Int

    func masks(magnitude: [Float], channels: Int, bins: Int, frames: Int) throws -> [[Float]] {
        [magnitude] + Array(repeating: [Float](repeating: 0, count: magnitude.count), count: stems - 1)
    }
}

struct SpleeterSeparatorTests {
    @Test
    func ratioMasksShareEveryElement() {
        let masks = [[3, 0, 1e-3], [4, 0, 0], [0, 0, 2e-3]].map { $0.map(Float.init) }
        let ratios = SpleeterRatioMask.ratioMasks(masks)
        for index in 0 ..< 3 {
            let sum = ratios.reduce(Float(0)) { $0 + $1[index] }
            #expect(abs(sum - 1) < 1e-5)
        }
        #expect(abs(ratios[0][0] - 9.0 / 25) < 1e-6)
        #expect(abs(ratios[1][1] - 1.0 / 3) < 1e-6)
    }

    @Test
    func uniformMasksSplitTheInputEvenly() throws {
        let separator = SpleeterSeparator(stems: .four, inference: UniformInference(stems: 4))
        let left = tones(count: 100_000, seed: 1)
        let right = tones(count: 100_000, seed: 2)
        let stems = try separator.separate([left, right])
        #expect(stems.count == 4)
        for stem in stems {
            #expect(stem.count == 2)
            #expect(stem[0].count == left.count)
            // The bins past 11 kHz are dropped, and the input's hard stop at its end spreads into them.
            #expect(relativeError(stem[0], left.map { $0 / 4 }) < 0.01)
            #expect(relativeError(stem[1], right.map { $0 / 4 }) < 0.01)
        }
    }

    @Test
    func monoComesBackMono() throws {
        let separator = SpleeterSeparator(stems: .two, inference: FirstStemInference(stems: 2))
        let mono = tones(count: 50000, seed: 3)
        let stems = try separator.separate([mono])
        #expect(stems.map(\.count) == [1, 1])
        #expect(relativeError(stems[0][0], mono) < 0.01)
        #expect((stems[1][0].map(abs).max() ?? 0) < 1e-4)
    }

    @Test
    func longerInputIsCutToTheWindow() throws {
        let configuration = SpleeterConfiguration(framesPerWindow: 512)
        let separator = SpleeterSeparator(
            stems: .five, inference: UniformInference(stems: 5), configuration: configuration
        )
        #expect(separator.windowLength == 1024 * 511)
        let stems = try separator.separate([noise(count: 600_000, seed: 4)])
        #expect(stems.allSatisfy { $0[0].count == 1024 * 511 })
    }

    @Test
    func rejectsAWrongMaskCount() {
        let separator = SpleeterSeparator(stems: .four, inference: UniformInference(stems: 2))
        #expect(throws: SpleeterSeparatorError.unexpectedMaskCount(expected: 4, actual: 2)) {
            try separator.separate([[0, 0]])
        }
        #expect(throws: SpleeterSeparatorError.unsupportedChannelCount(3)) {
            try separator.separate([[0], [0], [0]])
        }
    }

    @Test
    func layoutNamesMatchTheExports() {
        #expect(SpleeterStemLayout.two.maskNames == ["vocalsMask", "accompanimentMask"])
        #expect(SpleeterStemLayout.four.maskNames == ["vocalsMask", "drumsMask", "bassMask", "otherMask"])
        #expect(SpleeterStemLayout.five.maskNames == ["vocalsMask", "pianoMask", "drumsMask", "bassMask", "otherMask"])
    }
}
