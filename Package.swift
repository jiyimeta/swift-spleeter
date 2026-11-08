// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

// The ONNX Runtime backend (`SpleeterONNX`, `OrtInference`) links libonnxruntime, which this package does not ship, so
// it is declared only when asked for — Apple consumers that use the Core ML `Spleeter` product never see it:
//
//   SPLEETER_ONNX=1         declare the ONNX targets; the consumer puts libonnxruntime on the linker search path
//                           itself (`-Xlinker -L<dir>`), e.g. the AAR's jni/<abi>/ for Android.
//   SPLEETER_ORT_LIB=<dir>  also add <dir> to the search path and the runtime path — for building this package as the
//                           root (its tests on a Mac, against the desktop release's lib/). The flags are unsafe ones,
//                           which SwiftPM refuses in a dependency, so a consumer uses SPLEETER_ONNX=1 instead.
//
// See README.md, "ONNX Runtime backend".
let environment = ProcessInfo.processInfo.environment
let ortLibraryDirectory = environment["SPLEETER_ORT_LIB"].flatMap { $0.isEmpty ? nil : $0 }
let includesONNX = environment["SPLEETER_ONNX"] == "1" || ortLibraryDirectory != nil
let ortLinkerSettings: [LinkerSetting] = [.linkedLibrary("onnxruntime")] + (ortLibraryDirectory.map {
    [.unsafeFlags(["-L\($0)", "-Xlinker", "-rpath", "-Xlinker", $0])]
} ?? [])

let package = Package(
    name: "swift-spleeter",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .tvOS(.v15),
        .watchOS(.v8),
        .visionOS(.v1),
    ],
    products: [
        // Core ML separation for Apple platforms.
        .library(
            name: "Spleeter",
            targets: ["Spleeter"]
        ),
        // The separation's signal processing — STFT, ratio masks, one window end to end — on any platform, with the
        // model behind `SpleeterMaskInference`.
        .library(
            name: "SpleeterCore",
            targets: ["SpleeterCore"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "SpleeterCore"
        ),
        .target(
            name: "Spleeter",
            dependencies: ["SpleeterCore"]
        ),
        .testTarget(
            name: "SpleeterCoreTests",
            dependencies: ["SpleeterCore"]
        ),
        .testTarget(
            name: "SpleeterTests",
            dependencies: [
                "Spleeter",
            ]
        ),
    ]
)

if includesONNX {
    package.products += [
        // A Spleeter backend on ONNX Runtime, for platforms without Core ML.
        .library(name: "SpleeterONNX", targets: ["SpleeterONNX"]),
        // A small ONNX Runtime session wrapper: load a model file, run it with named float tensors.
        .library(name: "OrtInference", targets: ["OrtInference"]),
    ]
    // A shared library of the backend alone, so a cross-compile can check that it links against libonnxruntime
    // (a static product only compiles): `swift build --product SpleeterONNXLinkCheck`.
    if environment["SPLEETER_ONNX_LINK_CHECK"] == "1" {
        package.products.append(.library(name: "SpleeterONNXLinkCheck", type: .dynamic, targets: ["SpleeterONNX"]))
    }
    package.targets += [
        .target(
            name: "COnnxRuntime",
            exclude: ["LICENSE-onnxruntime"],
            linkerSettings: ortLinkerSettings
        ),
        .target(
            name: "OrtInference",
            dependencies: ["COnnxRuntime"]
        ),
        .target(
            name: "SpleeterONNX",
            dependencies: ["SpleeterCore", "OrtInference"]
        ),
        .testTarget(
            name: "SpleeterONNXTests",
            dependencies: [
                "SpleeterONNX",
                .target(name: "Spleeter", condition: .when(platforms: [.macOS, .iOS])),
            ]
        ),
    ]
}
