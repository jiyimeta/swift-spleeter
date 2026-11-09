# swift-spleeter

A Swift library for separating vocals and instruments from music using a pretrained [Spleeter](https://github.com/deezer/spleeter) model, powered by Core ML — and, through `SpleeterCore` and ONNX Runtime, on platforms without it, such as Android.

---

## Features

- Uses Deezer's **Spleeter model** compiled as a Core ML model
- Supports streaming audio processing for long files
- Exposes separation results as monaural audio, or per channel with `separateStereo(_:)`
- Applies Spleeter's ratio masks, so the stems add up to the input
- Works with stereo input (or mono, automatically handled)
- Async/Await and progress-reporting support (macOS 15+, iOS 18+)

---

## Installation

To use the `Spleeter` library in a SwiftPM project,
add the following line to the dependencies in your `Package.swift` file:

```swift
.package(url: "https://github.com/jiyimeta/swift-spleeter", from: "0.4.0"),
```

Include `"Spleeter"` as a dependency for your executable target:

```swift
.target(name: "<target>", dependencies: [
    .product(name: "Spleeter", package: "swift-spleeter"),
]),
```

Finally, add `import Spleeter` to your source code.

---

## Usage

### 1. Separate from file to file

```swift
import SwiftSpleeter

let modelURL = URL(fileURLWithPath: "path/to/Spleeter2.mlmodelc")
let inputURL = URL(fileURLWithPath: "path/to/mixture.wav")
let outputURLs = Stems2(
    vocals: FileManager.default.temporaryDirectory.appendingPathComponent("vocals.wav"),
    instruments: FileManager.default.temporaryDirectory.appendingPathComponent("instruments.wav")
)

let separator = try AudioSeparator2(modelURL: modelURL)

for try await progress in separator.separate(from: inputURL, to: outputURLs) {
    print("Progress: \(progress.current)/\(progress.total)")
}
```

### 2. Separate in-memory and handle results

```swift
import SwiftSpleeter

let modelURL = URL(fileURLWithPath: "path/to/Spleeter2.mlmodelc")
let inputURL = URL(fileURLWithPath: "path/to/mixture.wav")

let separator = try AudioSeparator2(modelURL: modelURL)
let file = try AudioFile(forReading: inputURL)
let samples = try file.readStereoSamples()

for try await (result, progress) in separator.separate(samples) {
    if let result = result {
        // result.vocals: [Float], result.instruments: [Float]
        print("Received chunk: vocals.count = \(result.vocals.count)")
    }
    print("Progress: \(progress.current)/\(progress.total)")
}
```

### 3. Keep both channels

```swift
let separator = try AudioSeparator4(modelURL: modelURL)
let samples = StereoValues(left: leftSamples, right: rightSamples)

for try await (result, _) in separator.separateStereo(samples) {
    if let result {
        // result.vocals.left / result.vocals.right: [Float], and the same for drums, bass and other
    }
}
```

Ending the stream early — breaking out of the loop, or cancelling the task that iterates it — stops the separation
after the window in progress.

---

## Requirements

- A compiled `.mlmodelc` version of the 2-, 4- or 5-stem Spleeter model
- Designed to work on macOS 15 / iOS 18 / tvOS 18 / watchOS 11 / visionOS 2 or later
- Library can be imported on earlier OS versions, but usage requires runtime availability checks via `if #available(...)`
- Elsewhere (Android, Linux): `SpleeterCore` with the ONNX Runtime backend, an `.onnx` model, and libonnxruntime 1.31 or
  later — see [Other platforms](#other-platforms-spleetercore-and-onnx-runtime)

---

## Models

- A compiled model (`Spleeter2Model.mlmodelc`) is included in [Demo App](#demo-app) and can be used directly. It is a
  216-frame model, as the 0.3.1 release's are.
- The 2-, 4- and 5-stem models are attached to the [releases](https://github.com/jiyimeta/swift-spleeter/releases):
  - 0.4.0: `Spleeter{2,4,5}Model.mlmodelc.zip`, 512-frame Core ML models (about 11.9 s per window), and
    `Spleeter4Model-512-fp16w.onnx.zip`, the 4-stem ONNX model for the ONNX Runtime backend (weights stored as FP16,
    run in FP32).
  - 0.3.1: the same models at 216 frames (about 5 s). They keep working with 0.4.0; the separator reads the window from
    the model. Models before 0.3.1 do not (see below).
- To customize the model, use the provided [spleeter-pytorch](https://github.com/jiyimeta/spleeter-pytorch) submodule under [Tools/spleeter-pytorch/](Tools/spleeter-pytorch/) to export and compile `.mlmodelc` files, or `.onnx` files with `convert-to-onnx.py`.

The models emit each stem's raw mask, and the library turns them into Spleeter's ratio masks
(m_i² / Σ m_j²) in FP32. Models from the 0.2.0 release computed the ratio masks themselves in FP16, where squaring a
magnitude above 256 overflows: loud bins such as a kick drum's dropped out of their stem, and this library then
normalized the masks a second time. The 2- and 4-stem models run in FP16; the 5-stem model is converted with
`--precision float32`, because its FP16 network output drifts from Spleeter's (about 20 dB SNR on the piano stem).

The window a separator streams is the model's: `clampingFrameCount` defaults to the frames its `magnitude` input
takes, and `windowLength` gives the samples per window — `1024 × 215` (about 5 s) for the 216-frame 0.3.x models,
`1024 × 511` (about 11.9 s) for models converted with `convert-to-coreml.py -f 512`. 512 frames is the separator's own
partition size, which it pads a shorter input to anyway, so a 512-frame model separates 2.4 times the audio per call
for about the same work.

### Moving to 0.4.0

- The public API is source-compatible. `StereoValues` and the `Stems2/4/5` types moved to `SpleeterCore`, which
  `import Spleeter` re-exports.
- `clampingFrameCount` is now `Int?`, defaulting to nil: the separator reads the frame count from the model, so the 0.3.1
  216-frame models and the 0.4.0 512-frame ones both work without it. Passing a number still overrides it.
- Code that assumed 5 s windows (`1024 × 215` samples) should read `windowLength` instead: with a 0.4.0 model each window
  is `1024 × 511` samples.
- `SpleeterSeparator` (one window end to end) and `SpleeterConfiguration` default to 512 frames per window.

---

## Other platforms: SpleeterCore and ONNX Runtime

`SpleeterCore` holds everything but the model, with no Apple framework: Spleeter's STFT on a portable FFT,
ratio masks, and `SpleeterSeparator`, which runs one window end to end on any `SpleeterMaskInference`. The `Spleeter`
product is SpleeterCore with a Core ML backend and vDSP's FFT.

### ONNX Runtime backend

`SpleeterONNX` (`OnnxMaskInference`) runs a model from `Tools/spleeter-pytorch/convert-to-onnx.py` in ONNX Runtime
through its C API; `OrtInference` is the small session wrapper under it (load a model file, run it with named float
tensors), usable for other ONNX models. Both link `libonnxruntime`, which this package does not ship, so they are
declared only when the environment asks for them:

| Variable | Effect |
| --- | --- |
| `SPLEETER_ONNX=1` | Declares `SpleeterONNX` and `OrtInference`. The consumer puts libonnxruntime on the linker search path, e.g. `-Xlinker -L<aar>/jni/arm64-v8a`. |
| `SPLEETER_ORT_LIB=<dir>` | Also adds `<dir>` to the search and runtime paths (unsafe flags, so only for this package as the root — its tests). |

The headers in `Sources/COnnxRuntime/include` are ONNX Runtime 1.31.0's C API (MIT); link a 1.31 or newer library.
On Android it comes from the Maven AAR `com.microsoft.onnxruntime:onnxruntime-android` (`jni/<abi>/libonnxruntime.so`),
which the APK also has to package. For the host tests, the desktop release:

```sh
# onnxruntime-osx-arm64-1.31.0.tgz from https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0
SPLEETER_ORT_LIB=~/Library/Caches/swift-spleeter-ort/download-osx/onnxruntime-osx-arm64-1.31.0/lib swift test
```

`OrtSessionOptions(intraOpThreads:executionProviders:)` takes providers to try before the CPU one — `.xnnpack(threads:)`
(in the Android AAR, not in the macOS release) or `.named("CoreML", options: […])` and the like. A provider the
linked build lacks, or refuses, is skipped; `session.executionProviders` and `skippedExecutionProviders` say what was
used. `OnnxMaskInference(modelPath:stems:options:)` passes them through.

To check an Android cross-compile links (swift.org 6.3.3 toolchain with its Android SDK bundle; the
`SpleeterONNXLinkCheck` product is a shared library of the backend, declared with `SPLEETER_ONNX_LINK_CHECK=1`):

```sh
SPLEETER_ONNX=1 SPLEETER_ONNX_LINK_CHECK=1 swift build --swift-sdk aarch64-unknown-linux-android28 \
    --product SpleeterONNXLinkCheck --scratch-path .build-android \
    -Xlinker -L<onnxruntime-android-1.31.0.aar unzipped>/jni/arm64-v8a
```

The model tests look for the exported models in `SPLEETER_TEST_MODELS` (default
`~/Library/Caches/swift-spleeter-ort/models`) and skip when they are missing.

---

## Demo App

To try the library in action:

```sh
cd Example
xcodegen
open SpleeterExample.xcodeproj
```

This will generate and open a demo Xcode project that demonstrates the separation functionality.

---

## License

This library is released under the MIT license. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project uses the 2-stem [Spleeter](https://github.com/deezer/spleeter) model by [Deezer Research](https://deezer.io/research).
