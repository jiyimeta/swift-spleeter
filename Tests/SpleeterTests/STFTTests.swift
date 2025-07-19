@testable import Spleeter

import Foundation
import Testing

struct STFTTests {
    static var sineWave: [Float] {
        let length = 32
        let frequency: Float = 2
        let signal = (0 ..< length).map { sin(2.0 * .pi * frequency * Float($0) / Float(length)) }

        return signal
    }

    @Test(arguments: [sineWave])
    func testSTFT(signal: [Float]) {
        let stft = STFT(fftSize: 16, hopLength: 4, frequencyLimit: 4)
        let spectrogram = stft.forward(waveform: signal)
        let reconstructed = stft.inverse(spectrogram)

        #expect(reconstructed.count == signal.count)
    }
}
