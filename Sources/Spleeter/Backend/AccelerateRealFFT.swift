import Accelerate
import SpleeterCore

/// `SpleeterFFT` on vDSP's packed real FFT.
///
/// vDSP returns twice the DFT and packs the Nyquist bin's real part where DC's imaginary part would be; this unpacks
/// it into the plain DFT `SpleeterFFT` promises.
struct AccelerateRealFFT: SpleeterFFT, @unchecked Sendable {
    let size: Int

    // A vDSP FFT setup is read-only once made, and vDSP documents it as safe to share between threads that use their
    // own buffers.
    private let fft: vDSP.FFT<DSPSplitComplex>

    init(size: Int) throws {
        precondition(size >= 4 && size & (size - 1) == 0, "size must be a power of two of at least 4")
        guard let fft = vDSP.FFT(
            log2n: vDSP_Length(log2(Double(size))),
            radix: .radix2,
            ofType: DSPSplitComplex.self
        ) else {
            throw STFTError.failedToCreateFFT
        }
        self.size = size
        self.fft = fft
    }

    func forward(
        _ input: UnsafeBufferPointer<Float>,
        real: UnsafeMutableBufferPointer<Float>,
        imag: UnsafeMutableBufferPointer<Float>
    ) {
        let half = size / 2
        withSplit { split in
            input.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { complex in
                vDSP_ctoz(complex, 2, &split, 1, vDSP_Length(half))
            }
            fft.forward(input: split, output: &split)
            real[0] = split.realp[0] * 0.5
            imag[0] = 0
            real[half] = split.imagp[0] * 0.5
            imag[half] = 0
            for k in 1 ..< half {
                real[k] = split.realp[k] * 0.5
                imag[k] = split.imagp[k] * 0.5
            }
        }
    }

    func inverse(
        real: UnsafeBufferPointer<Float>,
        imag: UnsafeBufferPointer<Float>,
        output: UnsafeMutableBufferPointer<Float>
    ) {
        let half = size / 2
        withSplit { split in
            split.realp[0] = real[0]
            split.imagp[0] = real[half]
            for k in 1 ..< half {
                split.realp[k] = real[k]
                split.imagp[k] = imag[k]
            }
            fft.inverse(input: split, output: &split)
            output.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { complex in
                vDSP_ztoc(&split, 1, complex, 2, vDSP_Length(half))
            }
            var scale = 1 / Float(size)
            vDSP_vsmul(output.baseAddress!, 1, &scale, output.baseAddress!, 1, vDSP_Length(size))
        }
    }

    private func withSplit(_ body: (inout DSPSplitComplex) -> Void) {
        let half = size / 2
        withUnsafeTemporaryAllocation(of: Float.self, capacity: half) { realp in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: half) { imagp in
                var split = DSPSplitComplex(realp: realp.baseAddress!, imagp: imagp.baseAddress!)
                body(&split)
            }
        }
    }
}
