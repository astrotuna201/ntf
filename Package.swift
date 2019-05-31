// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of
// Swift required to build this package.
import PackageDescription

#if os(Linux)
let excludeDirs: [String] = []

let NTFSwiftSettings: [SwiftSetting] = [
    SwiftSetting.define("CUDA"),
]

#else
let excludeDirs = ["device/cuda"]
#endif

let package = Package(
    name: "Netlib",
    products: [
        // Products define the executables and libraries produced by a package,
        // and make them visible to other packages.
        .library(name: "Netlib", targets: ["Netlib"]),
        .library(name: "Cuda", targets: ["Cuda"]),
        .library(name: "Jpeg", targets: ["Jpeg"]),
        .library(name: "Png", targets: ["Png"]),
        .library(name: "ZLib", targets: ["ZLib"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .systemLibrary(name: "Cuda", path: "Libraries/Cuda", pkgConfig: "cuda-10.0"),
        .systemLibrary(name: "Jpeg", path: "Libraries/Jpeg", pkgConfig: "libjpeg"),
        .systemLibrary(name: "Png", path: "Libraries/Png", pkgConfig: "libpng"),
        .systemLibrary(name: "ZLib", path: "Libraries/ZLib", pkgConfig: "zlib"),
        .target(name: "Netlib",  
                dependencies: ["Cuda", "Jpeg", "Png", "ZLib"], 
                exclude: excludeDirs,
                swiftSettings: NTFSwiftSettings),
        .testTarget(
                name: "NetlibTests", 
                dependencies: ["Netlib"],
                swiftSettings: NTFSwiftSettings)
    ]
)
