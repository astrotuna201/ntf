// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of
// Swift required to build this package.
import PackageDescription

let package = Package(
    name: "Netlib",
    products: [
        // Products define the executables and libraries produced by a package,
        // and make them visible to other packages.
        .library(name: "Netlib", targets: ["Netlib"]),
        .library(name: "Jpeg", targets: ["Jpeg"]),
        .library(name: "Png", targets: ["Png"]),
        .library(name: "ZLib", targets: ["ZLib"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .systemLibrary(name: "Jpeg", path: "Libraries/Jpeg"),
        .systemLibrary(name: "Png", path: "Libraries/Png"),
        .systemLibrary(name: "ZLib", path: "Libraries/ZLib"),
        .target(
            name: "Netlib",
            dependencies: ["Jpeg", "Png", "ZLib"]),
        .testTarget(
            name: "NetlibTests",
            dependencies: [])
    ]
)
