// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SimilaritySearchKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(
            name: "SimilaritySearchKit",
            targets: ["SimilaritySearchKit"]
        ),
        .library(
            name: "SimilaritySearchKitDistilbert",
            targets: ["SimilaritySearchKitDistilbert"]
        ),
        .library(
            name: "SimilaritySearchKitMiniLMAll",
            targets: ["SimilaritySearchKitMiniLMAll"]
        ),
        .library(
            name: "SimilaritySearchKitMiniLMMultiQA",
            targets: ["SimilaritySearchKitMiniLMMultiQA"]
        ),
        .library(
            name: "SimilaritySearchKitBGESmall",
            targets: ["SimilaritySearchKitBGESmall"]
        ),
        .library(
            name: "SimilaritySearchKitE5Small",
            targets: ["SimilaritySearchKitE5Small"]
        ),
    ],
    targets: [
        .target(
            name: "SimilaritySearchKit",
            dependencies: [],
            path: "Sources/SimilaritySearchKit/Core",
            resources: [.process("Resources")]
        ),
        .target(
            name: "SimilaritySearchKitDistilbert",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/Distilbert"
        ),
        .target(
            name: "SimilaritySearchKitMiniLMAll",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/MiniLMAll"
        ),
        .target(
            name: "SimilaritySearchKitMiniLMMultiQA",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/MiniLMMultiQA"
        ),
        .target(
            name: "SimilaritySearchKitBGESmall",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/BGESmall"
        ),
        .target(
            name: "SimilaritySearchKitE5Small",
            dependencies: ["SimilaritySearchKit"],
            path: "Sources/SimilaritySearchKit/AddOns/Embeddings/E5Small"
        ),
        .testTarget(
            name: "SimilaritySearchKitTests",
            dependencies: [
                "SimilaritySearchKit",
                "SimilaritySearchKitDistilbert",
                "SimilaritySearchKitMiniLMAll",
                "SimilaritySearchKitMiniLMMultiQA",
            ],
            path: "Tests/SimilaritySearchKitTests",
            resources: [.process("Resources")]
        ),
    ]
)
