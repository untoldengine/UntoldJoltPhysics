// swift-tools-version: 6.0

import PackageDescription

// Jolt verifies at RegisterTypes() that every translation unit including its
// headers was built with the same feature defines (JPH_VERSION_ID). The
// JoltPhysics package defines NDEBUG in release builds (SwiftPM never does
// for C/C++ targets; without it Jolt turns on JPH_DEBUG and therefore
// JPH_ENABLE_ASSERTS, a version-ID bit); the bridge must match it.
let joltDefines: [CXXSetting] = [
    .define("NDEBUG", .when(configuration: .release)),
]

let package = Package(
    name: "UntoldJoltPhysics",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v2),
    ],
    products: [
        .library(name: "UntoldJoltPhysics", targets: ["UntoldJoltPhysics"]),
    ],
    dependencies: [
        // Jolt Physics itself (MIT), compiled from source by SwiftPM: the
        // upstream tree with a Package.swift added, on a tag of the fork.
        // To update Jolt, tag the fork's next spm/<version> branch and bump
        // this pin.
        .package(url: "https://github.com/untoldengine/JoltPhysics.git", exact: "5.6.0-spm.1"),
        // On this branch the plugin builds against the fork's deformation branch,
        // which CoolMirror (its consumer for the cape cloth) depends on.
        .package(url: "https://github.com/untoldengine/UntoldEngine.git", branch: "develop"),
    ],
    targets: [
        // C ABI shim over the Jolt C++ API: opaque handles and plain structs,
        // only what the engine's PhysicsBackend protocol needs.
        .target(
            name: "CJoltBridge",
            dependencies: [
                .product(name: "JoltPhysics", package: "JoltPhysics"),
            ],
            path: "Sources/CJoltBridge",
            cxxSettings: joltDefines
        ),
        // The plugin: PhysicsBackend conformance + PhysicsBackendPlugin manifest.
        .target(
            name: "UntoldJoltPhysics",
            dependencies: [
                "CJoltBridge",
                .product(name: "UntoldEngine", package: "UntoldEngine"),
            ],
            path: "Sources/UntoldJoltPhysics",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
        .testTarget(
            name: "UntoldJoltPhysicsTests",
            dependencies: [
                "UntoldJoltPhysics",
                .product(name: "UntoldEngine", package: "UntoldEngine"),
            ],
            path: "Tests/UntoldJoltPhysicsTests",
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
