/// The stem sets Deezer's pretrained Spleeter models separate into.
public enum SpleeterStemLayout: Int, Sendable, CaseIterable {
    /// Vocals and accompaniment.
    case two = 2

    /// Vocals, drums, bass and other.
    case four = 4

    /// Vocals, piano, drums, bass and other.
    case five = 5

    /// The number of stems.
    public var stemCount: Int {
        rawValue
    }

    /// The stems' names, in the model's output order.
    public var stemNames: [String] {
        switch self {
        case .two: ["vocals", "accompaniment"]
        case .four: ["vocals", "drums", "bass", "other"]
        case .five: ["vocals", "piano", "drums", "bass", "other"]
        }
    }

    /// The exported models' mask output names, in `stemNames` order — the Core ML and ONNX conversions in
    /// `Tools/spleeter-pytorch` name them alike.
    public var maskNames: [String] {
        stemNames.map { $0 + "Mask" }
    }

    /// The model input's name.
    public static let magnitudeInputName = "magnitude"
}

/// The STFT and window geometry a Spleeter model was converted with.
public struct SpleeterConfiguration: Sendable, Equatable {
    /// The FFT size, a power of two.
    public var fftSize: Int

    /// The step between frames.
    public var hopLength: Int

    /// The bins the model sees, from DC up.
    public var frequencyBins: Int

    /// The STFT frames one model call covers.
    public var framesPerWindow: Int

    /// The analysis window's gain over the plain periodic Hann; see `SpleeterSTFT.init`.
    public var windowGain: Float

    /// Spleeter's geometry: a 4096-point FFT, a hop of 1024, 1024 bins, and 512 frames (1024 × 511 samples, about
    /// 11.9 s at 44.1 kHz) per call — the separator's own partition size, which a shorter window is padded to inside
    /// the model anyway. The Core ML models of 0.3.x take 216 frames (about 5 s).
    public init(
        fftSize: Int = 4096,
        hopLength: Int = 1024,
        frequencyBins: Int = 1024,
        framesPerWindow: Int = 512,
        windowGain: Float = SpleeterSTFT.vDSPHannGain
    ) {
        self.fftSize = fftSize
        self.hopLength = hopLength
        self.frequencyBins = frequencyBins
        self.framesPerWindow = framesPerWindow
        self.windowGain = windowGain
    }

    /// The samples one window covers, `hopLength × (framesPerWindow − 1)`: centered framing of exactly that many
    /// samples makes `framesPerWindow` frames.
    public var windowLength: Int {
        hopLength * (framesPerWindow - 1)
    }
}
