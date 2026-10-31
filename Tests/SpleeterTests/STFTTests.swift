import Foundation
@testable import Spleeter
import Testing

struct STFTTests {
    @Test
    func stft() throws {
        // fftSize == hopLength * 4 && signal.count % hopLength == 0
        let signal = [Float](repeating: 0, count: 220_160)
        let stft = try STFT(fftSize: 4096, hopLength: 1024, frequencyLimit: 1024)

        let spectrogram = try stft.forward(waveform: signal)
        let reconstructed = try stft.inverse(spectrogram)

        #expect(spectrogram.realFrames.count == 216)
        #expect(spectrogram.realFrames.map(\.count).max() == 1024)
        #expect(spectrogram.realFrames.map(\.count).min() == 1024)
        #expect(reconstructed.count == signal.count)
    }

    /// The forward transform is the plain DFT of each windowed frame, as `torch.stft` and `tf.signal.stft` compute
    /// it — the scale the model was trained on. vDSP's real FFT returns twice that.
    @Test
    func forwardMatchesTheDFT() throws {
        let fftSize = 16
        let stft = try STFT(fftSize: fftSize, hopLength: 4, frequencyLimit: 8)
        let signal = (0 ..< 64).map { Float(sin(Double($0) * 0.7) + 0.3 * cos(Double($0) * 2.1)) }
        let spectrogram = try stft.forward(waveform: signal)

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
            #expect(abs(Double(spectrogram.realFrames[3][bin]) - real) < 1e-4, "real, bin \(bin)")
            #expect(abs(Double(spectrogram.imagFrames[3][bin]) - imag) < 1e-4, "imag, bin \(bin)")
        }
    }

    /// Every bin kept, the inverse gives the signal back at its own level.
    @Test
    func inverseReturnsTheSignal() throws {
        let fftSize = 4096
        let stft = try STFT(fftSize: fftSize, hopLength: 1024, frequencyLimit: fftSize / 2)
        let signal = (0 ..< 1024 * 40).map { Float(sin(Double($0) * 0.01) * 0.5 + sin(Double($0) * 0.3) * 0.2) }
        let reconstructed = try stft.inverse(stft.forward(waveform: signal))
        #expect(reconstructed.count == signal.count)
        let worst = zip(signal, reconstructed).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 1e-3)
    }
}
