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
        .library(
            name: "Spleeter",
            targets: ["Spleeter"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "Spleeter"
        ),
        .testTarget(
            name: "SpleeterTests",
            dependencies: [
                "Spleeter",
            ]
        ),
    ]
)
