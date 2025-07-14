// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-spleeter",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
        .tvOS(.v13),
        .watchOS(.v6),
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
