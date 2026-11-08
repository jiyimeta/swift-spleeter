# swift-spleeter

A Swift library for separating vocals and instruments from music using a pretrained [Spleeter](https://github.com/deezer/spleeter) model, powered by Core ML.

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
.package(url: "https://github.com/jiyimeta/swift-spleeter", from: "0.1.0"),
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

---

## Models

- A compiled model (`Spleeter2Model.mlmodelc`) is included in [Demo App](#demo-app) and can be used directly.
- The 2-, 4- and 5-stem models are attached to the [releases](https://github.com/jiyimeta/swift-spleeter/releases). Use the
  ones from 0.3.1 or later with this version of the library.
- To customize the model, use the provided [spleeter-pytorch](https://github.com/jiyimeta/spleeter-pytorch) submodule under [Tools/spleeter-pytorch/](Tools/spleeter-pytorch/) to export and compile `.mlmodelc` files.

The models emit each stem's raw mask, and the library turns them into Spleeter's ratio masks
(m_i² / Σ m_j²) in FP32. Models from the 0.2.0 release computed the ratio masks themselves in FP16, where squaring a
magnitude above 256 overflows: loud bins such as a kick drum's dropped out of their stem, and this library then
normalized the masks a second time. The 2- and 4-stem models run in FP16; the 5-stem model is converted with
`--precision float32`, because its FP16 network output drifts from Spleeter's (about 20 dB SNR on the piano stem).

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
