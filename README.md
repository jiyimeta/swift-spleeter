# swift-spleeter

A Swift library for separating vocals and instruments from music using a pretrained [Spleeter](https://github.com/deezer/spleeter) model, powered by Core ML.

---

## Features

- Uses Deezer's **Spleeter model** compiled as a Core ML model
- Supports streaming audio processing for long files
- Exposes separation results as monaural audio
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

---

## Requirements

- A compiled `.mlmodelc` version of the 2-, 4- or 5-stem Spleeter model
- Designed to work on macOS 15 / iOS 18 / tvOS 18 / watchOS 11 / visionOS 2 or later
- Library can be imported on earlier OS versions, but usage requires runtime availability checks via `if #available(...)`

---

## Models

- A compiled model (`Spleeter2Model.mlmodelc`) is included in [Demo App](#demo-app) and can be used directly.
- To customize the model, use the provided [spleeter-pytorch](https://github.com/jiyimeta/spleeter-pytorch) submodule under [Tools/spleeter-pytorch/](Tools/spleeter-pytorch/) to export and compile `.mlmodelc` files.

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
