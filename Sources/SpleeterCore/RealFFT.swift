#if canImport(Foundation)
import Foundation
#endif

/// A real-input FFT of one fixed size, as the STFT uses it.
///
/// `forward` is the plain DFT — what `torch.stft` and `tf.signal.stft`, the transforms Spleeter was trained on, compute
/// — and `inverse` undoes it exactly, `1 / size` included. Conformers hold no per-call state, so one instance can serve
/// several threads at once.
public protocol SpleeterFFT: Sendable {
    /// The number of real samples per transform; a power of two.
    var size: Int { get }

    /// The DFT of `input`'s `size` samples, as bins `0 ... size / 2` (`size / 2 + 1` of them). The DC and Nyquist bins'
    /// imaginary parts are zero.
    func forward(
        _ input: UnsafeBufferPointer<Float>,
        real: UnsafeMutableBufferPointer<Float>,
        imag: UnsafeMutableBufferPointer<Float>
    )

    /// The `size` real samples whose DFT is bins `0 ... size / 2`. The DC and Nyquist bins' imaginary parts are
    /// ignored, as a real signal has none.
    func inverse(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        output: UnsafeMutableBufferPointer<Float>
    )
}

/// A radix-2 real FFT in plain Swift, for platforms without Accelerate.
///
/// The `size` real samples go through one complex FFT of `size / 2` points — even samples as the real part, odd as the
/// imaginary — and are split into the real signal's spectrum afterwards.
public struct PortableRealFFT: SpleeterFFT {
    public let size: Int

    /// `size / 2`, the complex FFT's length.
    private let half: Int
    private let bitReversed: [Int]
    /// `cos` / `sin` of `2πk / half`, for `k < half / 2`: the complex FFT's twiddles.
    private let complexCos: [Float]
    private let complexSin: [Float]
    /// `cos` / `sin` of `2πk / size`, for `k ≤ half`: the twiddles that split the packed spectrum.
    private let splitCos: [Float]
    private let splitSin: [Float]

    /// - Parameter size: The transform size, a power of two of at least 4.
    public init(size: Int) {
        precondition(size >= 4 && size & (size - 1) == 0, "size must be a power of two of at least 4")
        self.size = size
        half = size / 2

        var bits = 0
        while 1 << bits < half {
            bits += 1
        }
        bitReversed = (0 ..< size / 2).map { index in
            var reversed = 0
            for bit in 0 ..< bits where index & (1 << bit) != 0 {
                reversed |= 1 << (bits - 1 - bit)
            }
            return reversed
        }
        let half = size / 2
        complexCos = (0 ..< max(half / 2, 1)).map { Float(cos(2 * Double.pi * Double($0) / Double(half))) }
        complexSin = (0 ..< max(half / 2, 1)).map { Float(sin(2 * Double.pi * Double($0) / Double(half))) }
        splitCos = (0 ... half).map { Float(cos(2 * Double.pi * Double($0) / Double(size))) }
        splitSin = (0 ... half).map { Float(sin(2 * Double.pi * Double($0) / Double(size))) }
    }

    public func forward(
        _ input: UnsafeBufferPointer<Float>,
        real: UnsafeMutableBufferPointer<Float>,
        imag: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(input.count >= size && real.count > half && imag.count > half)
        withScratch { zr, zi in
            for n in 0 ..< half {
                let target = bitReversed[n]
                zr[target] = input[2 * n]
                zi[target] = input[2 * n + 1]
            }
            butterflies(zr, zi)

            for k in 0 ... half {
                let ak = k % half
                let bk = (half - k) % half
                // Z[k] and conj(Z[half - k]).
                let ar = zr[ak], ai = zi[ak]
                let br = zr[bk], bi = -zi[bk]
                // E = (Z[k] + conj(Z[half - k])) / 2, the even samples' DFT.
                let er = (ar + br) * 0.5
                let ei = (ai + bi) * 0.5
                // O = (Z[k] - conj(Z[half - k])) / 2i, the odd samples' DFT.
                let or = (ai - bi) * 0.5
                let oi = -(ar - br) * 0.5
                // X[k] = E + e^(-2πik / size) O.
                let c = splitCos[k], s = splitSin[k]
                real[k] = er + c * or + s * oi
                imag[k] = ei + c * oi - s * or
            }
            imag[0] = 0
            imag[half] = 0
        }
    }

    public func inverse(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(real.count > half && imag.count > half && output.count >= size)
        withScratch { zr, zi in
            for k in 0 ..< half {
                let xr = real[k]
                let xi = k == 0 ? 0 : imag[k]
                let yr = real[half - k]
                let yi = k == 0 ? 0 : -imag[half - k] // conj(X[half - k]); the Nyquist bin is real
                // E = (X[k] + conj(X[half - k])) / 2.
                let er = (xr + yr) * 0.5
                let ei = (xi + yi) * 0.5
                // O = (X[k] - conj(X[half - k])) / 2 · e^(2πik / size).
                let dr = (xr - yr) * 0.5
                let di = (xi - yi) * 0.5
                let c = splitCos[k], s = splitSin[k]
                let or = dr * c - di * s
                let oi = dr * s + di * c
                // Z[k] = E + iO, conjugated so the forward butterflies compute the inverse.
                let target = bitReversed[k]
                zr[target] = er - oi
                zi[target] = -(ei + or)
            }
            butterflies(zr, zi)
            let scale = 1 / Float(half)
            for n in 0 ..< half {
                output[2 * n] = zr[n] * scale
                output[2 * n + 1] = -zi[n] * scale
            }
        }
    }

    private func withScratch(_ body: (UnsafeMutableBufferPointer<Float>, UnsafeMutableBufferPointer<Float>) -> Void) {
        withUnsafeTemporaryAllocation(of: Float.self, capacity: half) { zr in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: half) { zi in
                body(zr, zi)
            }
        }
    }

    /// The forward complex FFT, in place, of a bit-reversed `half`-point sequence.
    private func butterflies(_ re: UnsafeMutableBufferPointer<Float>, _ im: UnsafeMutableBufferPointer<Float>) {
        complexCos.withUnsafeBufferPointer { cosTable in
            complexSin.withUnsafeBufferPointer { sinTable in
                var length = 2
                while length <= half {
                    let span = length / 2
                    let step = half / length
                    var start = 0
                    while start < half {
                        for j in 0 ..< span {
                            let wr = cosTable[j * step]
                            let wi = -sinTable[j * step]
                            let a = start + j
                            let b = a + span
                            let tr = wr * re[b] - wi * im[b]
                            let ti = wr * im[b] + wi * re[b]
                            re[b] = re[a] - tr
                            im[b] = im[a] - ti
                            re[a] += tr
                            im[a] += ti
                        }
                        start += length
                    }
                    length *= 2
                }
            }
        }
    }
}
