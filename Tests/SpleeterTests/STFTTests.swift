@testable import Spleeter

import Foundation
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
}
