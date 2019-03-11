// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Netlib",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
        name: "Netlib",
        targets: ["Netlib"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/danger/swift.git", from: "1.0.0"),
        .package(url: "https://github.com/nicklockwood/SwiftFormat", from: "0.35.8"),
        .package(url: "https://github.com/Realm/SwiftLint", from: "0.28.1"),
        .package(url: "https://github.com/orta/Komondor", from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "Netlib",
            dependencies: []),
        .testTarget(
            name: "NetlibTests",
            dependencies: ["Netlib"]),
    ]
)

// The settings for the git hooks for our repo
#if canImport(PackageConfig)
import PackageConfig

let config = PackageConfig([
    "komondor": [
        // When someone has run `git commit`, first run
        // run SwiftFormat and the auto-correcter for SwiftLint
        "pre-commit": [
            "swift run swiftformat .",
            "swift run swiftlint autocorrect --path Netlib/",
            "git add .",
        ],
    ]
    ])
#endif

