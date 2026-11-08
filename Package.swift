// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

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
