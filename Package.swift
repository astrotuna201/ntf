// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of
// Swift required to build this package.
import PackageDescription

let package = Package(
    name: "Netlib",
    products: [
        // Products define the executables and libraries produced by a package,
        // and make them visible to other packages.
        .library(name: "Netlib", targets: ["Netlib"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "Netlib",
            dependencies: []),
        .testTarget(
            name: "NetlibTests",
            dependencies: ["Netlib"]),
    ]
)
