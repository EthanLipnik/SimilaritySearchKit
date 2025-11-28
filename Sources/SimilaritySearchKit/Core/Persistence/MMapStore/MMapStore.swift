//
//  MMapStore.swift
//  SimilaritySearchKit
//
//  Created by Claude on 11/28/25.
//

import Foundation

/// Memory-mapped vector store for fast loading and saving of embedding indexes.
///
/// File format:
/// ```
/// [Header: 48 bytes]
///   Magic: 4 bytes "SSKT"
///   Version: 4 bytes (currently 1)
///   Item count: 8 bytes (UInt64)
///   Embedding dimension: 4 bytes (UInt32)
///   IDs offset: 8 bytes (UInt64)
///   Metadata offset: 8 bytes (UInt64)
///   Text offset: 8 bytes (UInt64)
///   Reserved: 4 bytes
/// [Embeddings: item_count * dimension * 4 bytes]
///   Contiguous Float32 array
/// [IDs section]
///   For each item: length (UInt32) + UTF-8 bytes
/// [Metadata section]
///   Single JSON blob encoding [[String: String]]
/// [Text section]
///   For each item: length (UInt32) + UTF-8 bytes
/// ```
public class MMapStore: VectorStoreProtocol {

    // MARK: - Constants

    private static let magic: UInt32 = 0x54534B53 // "SSKT" in little-endian
    private static let version: UInt32 = 1
    private static let headerSize: Int = 48

    // MARK: - Errors

    public enum MMapStoreError: LocalizedError {
        case invalidMagic
        case unsupportedVersion(UInt32)
        case fileTooSmall
        case dimensionMismatch(expected: Int, got: Int)
        case corruptedData(String)
        case writeError(String)

        public var errorDescription: String? {
            switch self {
            case .invalidMagic:
                return "Invalid file format: not a valid MMapStore file"
            case .unsupportedVersion(let version):
                return "Unsupported file version: \(version)"
            case .fileTooSmall:
                return "File is too small to contain valid data"
            case .dimensionMismatch(let expected, let got):
                return "Embedding dimension mismatch: expected \(expected), got \(got)"
            case .corruptedData(let message):
                return "Corrupted data: \(message)"
            case .writeError(let message):
                return "Write error: \(message)"
            }
        }
    }

    // MARK: - Initialization

    public init() {}

    // MARK: - VectorStoreProtocol

    public func saveIndex(items: [IndexItem], to url: URL, as name: String) throws -> URL {
        let fileURL = url.appendingPathComponent("\(name).mmap")

        guard !items.isEmpty else {
            // Write empty file with just header
            var data = Data(count: Self.headerSize)
            writeHeader(to: &data, itemCount: 0, dimension: 0, idsOffset: UInt64(Self.headerSize), metadataOffset: UInt64(Self.headerSize), textOffset: UInt64(Self.headerSize))
            try data.write(to: fileURL, options: .atomic)
            return fileURL
        }

        let dimension = items[0].embedding.count

        // Validate all items have same dimension
        for (index, item) in items.enumerated() {
            guard item.embedding.count == dimension else {
                throw MMapStoreError.dimensionMismatch(expected: dimension, got: item.embedding.count)
            }
        }

        // Calculate section sizes
        let embeddingsSize = items.count * dimension * MemoryLayout<Float>.size
        let idsOffset = UInt64(Self.headerSize + embeddingsSize)

        // Build IDs section
        var idsData = Data()
        for item in items {
            let idData = item.id.data(using: .utf8) ?? Data()
            var length = UInt32(idData.count)
            idsData.append(Data(bytes: &length, count: 4))
            idsData.append(idData)
        }

        let metadataOffset = idsOffset + UInt64(idsData.count)

        // Build metadata section (single JSON array)
        let metadataArray = items.map(\.metadata)
        let metadataData = try JSONEncoder().encode(metadataArray)
        var metadataLength = UInt32(metadataData.count)
        var metadataSection = Data(bytes: &metadataLength, count: 4)
        metadataSection.append(metadataData)

        let textOffset = metadataOffset + UInt64(metadataSection.count)

        // Build text section
        var textData = Data()
        for item in items {
            let textBytes = item.text.data(using: .utf8) ?? Data()
            var length = UInt32(textBytes.count)
            textData.append(Data(bytes: &length, count: 4))
            textData.append(textBytes)
        }

        // Assemble final data
        let totalSize = Self.headerSize + embeddingsSize + idsData.count + metadataSection.count + textData.count
        var data = Data(capacity: totalSize)

        // Write header
        data.append(Data(count: Self.headerSize))
        writeHeader(
            to: &data,
            itemCount: UInt64(items.count),
            dimension: UInt32(dimension),
            idsOffset: idsOffset,
            metadataOffset: metadataOffset,
            textOffset: textOffset
        )

        // Write embeddings contiguously
        for item in items {
            item.embedding.withUnsafeBytes { ptr in
                data.append(contentsOf: ptr)
            }
        }

        // Write sections
        data.append(idsData)
        data.append(metadataSection)
        data.append(textData)

        try data.write(to: fileURL, options: .atomic)

        return fileURL
    }

    public func loadIndex(from url: URL) throws -> [IndexItem] {
        // Memory-map the file for efficient access
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { try? fileHandle.close() }

        guard let data = try fileHandle.readToEnd(), data.count >= Self.headerSize else {
            throw MMapStoreError.fileTooSmall
        }

        // Parse header
        let header = try parseHeader(from: data)

        guard header.itemCount > 0 else {
            return []
        }

        let dimension = Int(header.dimension)
        let itemCount = Int(header.itemCount)

        // Read embeddings
        let embeddingsStart = Self.headerSize
        let embeddingsEnd = embeddingsStart + itemCount * dimension * MemoryLayout<Float>.size

        guard data.count >= embeddingsEnd else {
            throw MMapStoreError.corruptedData("File too small for embeddings")
        }

        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(itemCount)

        let embeddingsData = data[embeddingsStart..<embeddingsEnd]
        embeddingsData.withUnsafeBytes { ptr in
            let floatPtr = ptr.bindMemory(to: Float.self)
            for i in 0..<itemCount {
                let start = i * dimension
                let embedding = Array(floatPtr[start..<(start + dimension)])
                embeddings.append(embedding)
            }
        }

        // Read IDs
        var ids: [String] = []
        ids.reserveCapacity(itemCount)
        var offset = Int(header.idsOffset)

        for _ in 0..<itemCount {
            guard offset + 4 <= data.count else {
                throw MMapStoreError.corruptedData("Unexpected end of IDs section")
            }
            let length = data[offset..<(offset + 4)].withUnsafeBytes { $0.load(as: UInt32.self) }
            offset += 4

            guard offset + Int(length) <= data.count else {
                throw MMapStoreError.corruptedData("ID string extends beyond file")
            }

            let idData = data[offset..<(offset + Int(length))]
            guard let id = String(data: idData, encoding: .utf8) else {
                throw MMapStoreError.corruptedData("Invalid UTF-8 in ID")
            }
            ids.append(id)
            offset += Int(length)
        }

        // Read metadata
        let metadataOffset = Int(header.metadataOffset)
        guard metadataOffset + 4 <= data.count else {
            throw MMapStoreError.corruptedData("Metadata section too small")
        }

        let metadataLength = data[metadataOffset..<(metadataOffset + 4)].withUnsafeBytes { $0.load(as: UInt32.self) }
        let metadataStart = metadataOffset + 4
        let metadataEnd = metadataStart + Int(metadataLength)

        guard metadataEnd <= data.count else {
            throw MMapStoreError.corruptedData("Metadata extends beyond file")
        }

        let metadataData = data[metadataStart..<metadataEnd]
        let metadataArray: [[String: String]]
        do {
            metadataArray = try JSONDecoder().decode([[String: String]].self, from: metadataData)
        } catch {
            throw MMapStoreError.corruptedData("Failed to decode metadata: \(error.localizedDescription)")
        }

        guard metadataArray.count == itemCount else {
            throw MMapStoreError.corruptedData("Metadata count mismatch: expected \(itemCount), got \(metadataArray.count)")
        }

        // Read texts
        var texts: [String] = []
        texts.reserveCapacity(itemCount)
        offset = Int(header.textOffset)

        for _ in 0..<itemCount {
            guard offset + 4 <= data.count else {
                throw MMapStoreError.corruptedData("Unexpected end of text section")
            }
            let length = data[offset..<(offset + 4)].withUnsafeBytes { $0.load(as: UInt32.self) }
            offset += 4

            guard offset + Int(length) <= data.count else {
                throw MMapStoreError.corruptedData("Text string extends beyond file")
            }

            let textData = data[offset..<(offset + Int(length))]
            guard let text = String(data: textData, encoding: .utf8) else {
                throw MMapStoreError.corruptedData("Invalid UTF-8 in text")
            }
            texts.append(text)
            offset += Int(length)
        }

        // Assemble IndexItems
        var items: [IndexItem] = []
        items.reserveCapacity(itemCount)

        for i in 0..<itemCount {
            let item = IndexItem(
                id: ids[i],
                text: texts[i],
                embedding: embeddings[i],
                metadata: metadataArray[i]
            )
            items.append(item)
        }

        return items
    }

    public func listIndexes(at url: URL) -> [URL] {
        let fileManager = FileManager.default
        do {
            let files = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil, options: [])
            return files.filter { $0.pathExtension == "mmap" }
        } catch {
            print("Error listing indexes: \(error)")
            return []
        }
    }

    // MARK: - Private Helpers

    private struct Header {
        let magic: UInt32
        let version: UInt32
        let itemCount: UInt64
        let dimension: UInt32
        let idsOffset: UInt64
        let metadataOffset: UInt64
        let textOffset: UInt64
    }

    private func writeHeader(
        to data: inout Data,
        itemCount: UInt64,
        dimension: UInt32,
        idsOffset: UInt64,
        metadataOffset: UInt64,
        textOffset: UInt64
    ) {
        data.replaceSubrange(0..<4, with: withUnsafeBytes(of: Self.magic.littleEndian) { Data($0) })
        data.replaceSubrange(4..<8, with: withUnsafeBytes(of: Self.version.littleEndian) { Data($0) })
        data.replaceSubrange(8..<16, with: withUnsafeBytes(of: itemCount.littleEndian) { Data($0) })
        data.replaceSubrange(16..<20, with: withUnsafeBytes(of: dimension.littleEndian) { Data($0) })
        data.replaceSubrange(20..<28, with: withUnsafeBytes(of: idsOffset.littleEndian) { Data($0) })
        data.replaceSubrange(28..<36, with: withUnsafeBytes(of: metadataOffset.littleEndian) { Data($0) })
        data.replaceSubrange(36..<44, with: withUnsafeBytes(of: textOffset.littleEndian) { Data($0) })
        // Reserved bytes 44-48 are left as zeros
    }

    private func parseHeader(from data: Data) throws -> Header {
        guard data.count >= Self.headerSize else {
            throw MMapStoreError.fileTooSmall
        }

        let magic = data[0..<4].withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        guard magic == Self.magic else {
            throw MMapStoreError.invalidMagic
        }

        let version = data[4..<8].withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        guard version == Self.version else {
            throw MMapStoreError.unsupportedVersion(version)
        }

        let itemCount = data[8..<16].withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        let dimension = data[16..<20].withUnsafeBytes { $0.load(as: UInt32.self).littleEndian }
        let idsOffset = data[20..<28].withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        let metadataOffset = data[28..<36].withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        let textOffset = data[36..<44].withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }

        return Header(
            magic: magic,
            version: version,
            itemCount: itemCount,
            dimension: dimension,
            idsOffset: idsOffset,
            metadataOffset: metadataOffset,
            textOffset: textOffset
        )
    }
}
