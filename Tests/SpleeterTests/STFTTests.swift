import Accelerate
import Foundation
@testable import Spleeter
import SpleeterCore
import Testing

/// `SpleeterSTFT` — on vDSP and on the portable FFT — reproduces the vDSP STFT the library shipped through 0.3.1.
struct STFTTests {
    static func noise(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0 ..< count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(state >> 40) / Float(1 << 24) - 0.5
        }
    }

    static func relativeError(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count)
        var error = 0.0
        var energy = 0.0
        for index in a.indices {
            let difference = Double(a[index]) - Double(b[index])
            error += difference * difference
            energy += Double(b[index]) * Double(b[index])
        }
        return (error / energy).squareRoot()
    }

    /// The legacy spectrogram in `ComplexSpectrogram`'s bin-major layout.
    static func binMajor(_ legacy: Spectrogram) -> ComplexSpectrogram {
        let frames = legacy.realFrames.count
        let bins = legacy.realFrames[0].count
        var real = [Float](repeating: 0, count: bins * frames)
        var imag = [Float](repeating: 0, count: bins * frames)
        for t in 0 ..< frames {
            for f in 0 ..< bins {
                real[f * frames + t] = legacy.realFrames[t][f]
                imag[f * frames + t] = legacy.imagFrames[t][f]
            }
        }
        return ComplexSpectrogram(bins: bins, frames: frames, real: real, imag: imag)
    }

    static func ffts() throws -> [(String, any SpleeterFFT)] {
        try [("vDSP", AccelerateRealFFT(size: 4096)), ("portable", PortableRealFFT(size: 4096))]
    }

    @Test
    func windowIsVDSPsHann() throws {
        let legacy = try LegacySTFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)
        let core = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 1024)
        #expect(Self.relativeError(core.window, legacy.window) < 1e-6)
    }

    @Test
    func forwardMatchesTheLegacySTFT() throws {
        let signal = Self.noise(count: 1024 * 215, seed: 11)
        let legacy = try Self.binMajor(LegacySTFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)
            .forward(waveform: signal))
        for (name, fft) in try Self.ffts() {
            let core = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 1024, fft: fft).forward(signal)
            #expect(core.frames == legacy.frames && core.bins == legacy.bins)
            let realError = Self.relativeError(core.real, legacy.real)
            let imagError = Self.relativeError(core.imag, legacy.imag)
            print("STFT forward, \(name) vs legacy vDSP: real \(realError), imag \(imagError)")
            #expect(realError < 1e-5, "\(name)")
            #expect(imagError < 1e-5, "\(name)")
        }
    }

    @Test
    func inverseMatchesTheLegacySTFT() throws {
        let signal = Self.noise(count: 1024 * 215, seed: 12)
        let legacySTFT = try LegacySTFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)
        let legacySpectrogram = try legacySTFT.forward(waveform: signal)
        // A masked spectrogram, as the separator inverts: halve every other bin.
        let masked = Spectrogram(
            realFrames: legacySpectrogram.realFrames.map { $0.enumerated().map { $0.offset % 2 == 0 ? $0.element : $0.element / 2 } },
            imagFrames: legacySpectrogram.imagFrames.map { $0.enumerated().map { $0.offset % 2 == 0 ? $0.element : $0.element / 2 } }
        )
        let legacy = try legacySTFT.inverse(masked)
        for (name, fft) in try Self.ffts() {
            let core = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 1024, fft: fft)
                .inverse(Self.binMajor(masked))
            #expect(core.count == legacy.count)
            let error = Self.relativeError(core, legacy)
            print("STFT inverse, \(name) vs legacy vDSP: \(error)")
            #expect(error < 1e-5, "\(name)")
        }
    }

    /// The forward transform is the plain DFT of each windowed frame, as `torch.stft` and `tf.signal.stft` compute
    /// it — the scale the model was trained on. vDSP's real FFT returns twice that.
    @Test
    func forwardMatchesTheDFT() throws {
        let fftSize = 16
        for fft in try [AccelerateRealFFT(size: fftSize), PortableRealFFT(size: fftSize)] as [any SpleeterFFT] {
            let stft = SpleeterSTFT(fftSize: fftSize, hopLength: 4, frequencyBins: 8, fft: fft)
            let signal = (0 ..< 64).map { Float(sin(Double($0) * 0.7) + 0.3 * cos(Double($0) * 2.1)) }
            let spectrogram = stft.forward(signal)

            // Frame 3 starts at 3 × hop in the signal padded by fftSize / 2 on the left.
            let padded = [Float](repeating: 0, count: fftSize / 2) + signal + [Float](repeating: 0, count: fftSize / 2)
            let frame = (0 ..< fftSize).map { padded[3 * 4 + $0] * stft.window[$0] }
            for bin in 0 ..< 8 {
                var real = 0.0
                var imag = 0.0
                for (n, sample) in frame.enumerated() {
                    let angle = -2 * Double.pi * Double(bin * n) / Double(fftSize)
                    real += Double(sample) * cos(angle)
                    imag += Double(sample) * sin(angle)
                }
                let index = bin * spectrogram.frames + 3
                #expect(abs(Double(spectrogram.real[index]) - real) < 1e-4, "real, bin \(bin)")
                #expect(abs(Double(spectrogram.imag[index]) - imag) < 1e-4, "imag, bin \(bin)")
            }
        }
    }

    /// The portable FFT costs this much more than vDSP over one window's transforms (2 forward, 8 inverse STFTs).
    @Test
    func timing() throws {
        let signal = Self.noise(count: 1024 * 215, seed: 13)
        for (name, fft) in try Self.ffts() {
            let stft = SpleeterSTFT(fftSize: 4096, hopLength: 1024, frequencyBins: 1024, fft: fft)
            var spectrogram = stft.forward(signal)
            let start = Date()
            for _ in 0 ..< 2 {
                spectrogram = stft.forward(signal)
            }
            for _ in 0 ..< 8 {
                _ = stft.inverse(spectrogram)
            }
            let elapsed = Date().timeIntervalSince(start)
            print("STFT timing, \(name): \(Int(elapsed * 1000)) ms for 2 forward + 8 inverse")
        }
        let legacy = try LegacySTFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)
        var spectrogram = try legacy.forward(waveform: signal)
        let start = Date()
        for _ in 0 ..< 2 {
            spectrogram = try legacy.forward(waveform: signal)
        }
        for _ in 0 ..< 8 {
            _ = try legacy.inverse(spectrogram)
        }
        print("STFT timing, legacy: \(Int(Date().timeIntervalSince(start) * 1000)) ms for 2 forward + 8 inverse")
    }
}
