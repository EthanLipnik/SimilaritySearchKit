//
//  SimilarityIndex.swift
//
//
//  Created by Zach Nagengast on 4/10/23.
//

import Foundation

// MARK: - Type Aliases

public typealias IndexItem = SimilarityIndex.IndexItem
public typealias SearchResult = SimilarityIndex.SearchResult
public typealias EmbeddingModelType = SimilarityIndex.EmbeddingModelType
public typealias SimilarityMetricType = SimilarityIndex.SimilarityMetricType
public typealias TextSplitterType = SimilarityIndex.TextSplitterType
public typealias VectorStoreType = SimilarityIndex.VectorStoreType

@available(macOS 11.0, iOS 15.0, *)
public class SimilarityIndex: Identifiable, Hashable, @unchecked Sendable {
    // MARK: - Properties

    /// Unique identifier for this index instance
    public var id: UUID = .init()
    public static func == (lhs: SimilarityIndex, rhs: SimilarityIndex) -> Bool {
        return lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    /// The items stored in the index.
    public var indexItems: [IndexItem] = [] {
        didSet {
            // Rebuild secondary index when items change externally
            rebuildItemIndex()
        }
    }

    /// The dimension of the embeddings in the index.
    /// Used to validate emebdding updates
    public private(set) var dimension: Int = 0

    /// The name of the index.
    public var indexName: String

    public let indexModel: any EmbeddingsProtocol
    public var indexMetric: any DistanceMetricProtocol
    public let vectorStore: any VectorStoreProtocol

    /// HNSW index for approximate nearest neighbor search (O(log n) vs O(n))
    private var hnswIndex: HNSWIndex?

    /// Whether to use HNSW for search (default true for indexes > 100 items)
    public var useHNSW: Bool = true

    /// Secondary index for O(1) item lookup by ID
    private var itemsByID: [String: Int] = [:]

    /// Rebuilds the secondary item index
    private func rebuildItemIndex() {
        itemsByID.removeAll(keepingCapacity: true)
        for (index, item) in indexItems.enumerated() {
            itemsByID[item.id] = index
        }
    }

    /// Rebuilds the HNSW index from current items
    public func rebuildHNSWIndex() {
        guard useHNSW, !indexItems.isEmpty else {
            hnswIndex = nil
            return
        }

        let hnsw = HNSWIndex(M: 16, efConstruction: 200, efSearch: 50)
        for item in indexItems {
            hnsw.insert(id: item.id, embedding: item.embedding)
        }
        hnswIndex = hnsw
    }

    /// An object representing an item in the index.
    public struct IndexItem: Codable, Sendable {
        /// The unique identifier of the item.
        public let id: String

        /// The text associated with the item.
        public var text: String

        /// The embedding vector of the item.
        public var embedding: [Float]

        /// A dictionary containing metadata for the item.
        public var metadata: [String: String]
    }

    /// An Identifiable object containing information about a search result.
    public struct SearchResult: Identifiable {
        /// The unique identifier of the associated index item
        public let id: String

        /// The similarity score between the query and the result.
        public let score: Float

        /// The text associated with the result.
        public let text: String

        /// A dictionary containing metadata for the result.
        public let metadata: [String: String]
    }

    /// An enumeration of available embedding models.
    public enum EmbeddingModelType {
        /// DistilBERT, a small version of BERT model fine tuned for questing-answering.
        case distilbert

        /// MiniLM All, a smaller but faster model.
        case minilmAll

        /// Multi-QA MiniLM, a fast model fine-tuned for question-answering tasks.
        case minilmMultiQA

        /// A native model provided by Apple's NaturalLanguage library.
        case native
    }

    public enum SimilarityMetricType {
        case dotproduct
        case cosine
        case euclidian
    }

    public enum TextSplitterType {
        case token
        case character
        case recursive
    }

    public enum VectorStoreType {
        case json
        // TODO:
        // case mlmodel
        // case protobuf
        // case sqlite
    }

    // MARK: - Initializers

    public init(name: String? = nil, model: (any EmbeddingsProtocol)? = nil, metric: (any DistanceMetricProtocol)? = nil, vectorStore: (any VectorStoreProtocol)? = nil) async {
        // Setup index with defaults
        self.indexName = name ?? "SimilaritySearchKitIndex"
        self.indexModel = model ?? NativeEmbeddings()
        self.indexMetric = metric ?? CosineSimilarity()
        self.vectorStore = vectorStore ?? JsonStore()

        // Run the model once to discover dimention size
        await setupDimension()
    }

    private func setupDimension() async {
        if let testVector = await indexModel.encode(sentence: "Test sentence") {
            dimension = testVector.count
        } else {
            print("Failed to generate a test input vector.")
        }
    }

    // MARK: - Encoding

    public func getEmbedding(for text: String, embedding: [Float]? = nil) async -> [Float] {
        if let embedding = embedding, embedding.count == dimension {
            // Valid embedding, no encoding needed
            return embedding
        } else {
            // Encoding needed before adding to index
            guard let encoded = await indexModel.encode(sentence: text) else {
                print("Failed to encode text. \(text)")
                return Array(repeating: Float(0), count: dimension)
            }
            return encoded
        }
    }

    // MARK: - Search

    public func search(_ query: String, top resultCount: Int? = nil, metric: DistanceMetricProtocol? = nil) async -> [SearchResult] {
        let resultCount = resultCount ?? 5
        guard let queryEmbedding = await indexModel.encode(sentence: query) else {
            print("Failed to generate query embedding for '\(query)'.")
            return []
        }

        // Use HNSW for fast approximate search when available and index is large enough
        if useHNSW, let hnsw = hnswIndex, indexItems.count >= 100 {
            let hnswResults = hnsw.search(query: queryEmbedding, k: resultCount)

            return hnswResults.compactMap { result in
                if let item = getItem(id: result.id) {
                    // Convert distance to similarity score (1 - normalized_distance)
                    let score = max(0, 1 - result.distance / 10.0)
                    return SearchResult(id: item.id, score: score, text: item.text, metadata: item.metadata)
                }
                return nil
            }
        }

        // Fall back to brute force for small indexes or when HNSW is disabled
        var indexIds: [String] = []
        var indexEmbeddings: [[Float]] = []

        for item in indexItems {
            indexIds.append(item.id)
            indexEmbeddings.append(item.embedding)
        }

        // Calculate distances and find nearest neighbors
        if let customMetric = metric {
            // Allow custom metrics at time of query
            indexMetric = customMetric
        }
        let searchResults = indexMetric.findNearest(for: queryEmbedding, in: indexEmbeddings, resultsCount: resultCount)

        // Map results to index ids
        return searchResults.compactMap { result in
            let (score, index) = result
            let id = indexIds[index]

            if let item = getItem(id: id) {
                return SearchResult(id: item.id, score: score, text: item.text, metadata: item.metadata)
            } else {
                print("Failed to find item with id '\(id)' in indexItems.")
                return SearchResult(id: "000000", score: 0.0, text: "fail", metadata: [:])
            }
        }
    }

    public class func combinedResultsString(_ results: [SearchResult]) -> String {
        let combinedResults = results.map { result -> String in
            let metadataString = result.metadata.map { key, value in
                "\(key.uppercased()): \(value)"
            }.joined(separator: "\n")

            return "\(result.text)\n\(metadataString)"
        }.joined(separator: "\n\n")

        return combinedResults
    }

    public class func exportLLMPrompt(query: String, results: [SearchResult]) -> String {
        let sourcesText = combinedResultsString(results)
        let prompt =
            """
            Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            ALWAYS return a "SOURCES" part in your answer.

            QUESTION: \(query)
            =========
            \(sourcesText)
            =========
            FINAL ANSWER:
            """
        return prompt
    }
}

// MARK: - CRUD

@available(macOS 11.0, iOS 15.0, *)
public extension SimilarityIndex {
    // MARK: Create

    /// Add an item with optional pre-computed embedding
    func addItem(id: String, text: String, metadata: [String: String], embedding: [Float]? = nil) async {
        let embeddingResult = await getEmbedding(for: text, embedding: embedding)

        let item = IndexItem(id: id, text: text, embedding: embeddingResult, metadata: metadata)

        // Update secondary index
        itemsByID[id] = indexItems.count

        // Add to HNSW if enabled
        if useHNSW {
            if hnswIndex == nil {
                hnswIndex = HNSWIndex(M: 16, efConstruction: 200, efSearch: 50)
            }
            hnswIndex?.insert(id: id, embedding: embeddingResult)
        }

        indexItems.append(item)
    }

    /// Progress update for addItems operations
    struct AddItemsProgress: Sendable {
        public let id: String
        public let current: Int
        public let total: Int
    }

    func addItems(ids: [String], texts: [String], metadata: [[String: String]], embeddings: [[Float]?]? = nil) -> AsyncStream<AddItemsProgress> {
        precondition(ids.count == texts.count && texts.count == metadata.count, "Input arrays must have the same length.")

        if let embeddings = embeddings, embeddings.count != ids.count {
            print("Embeddings array length must be the same as ids array length. \(embeddings.count) vs \(ids.count)")
        }

        let (stream, continuation) = AsyncStream.makeStream(of: AddItemsProgress.self)

        let total = ids.count
        let capturedIds = ids
        let capturedTexts = texts
        let capturedMetadata = metadata
        let capturedEmbeddings = embeddings

        Task { [weak self] in
            for i in 0..<total {
                await self?.addItem(id: capturedIds[i], text: capturedTexts[i], metadata: capturedMetadata[i], embedding: capturedEmbeddings?[i])
                continuation.yield(AddItemsProgress(id: capturedIds[i], current: i + 1, total: total))
            }
            continuation.finish()
        }

        return stream
    }

    func addItems(_ items: [IndexItem]) -> AsyncStream<AddItemsProgress> {
        let (stream, continuation) = AsyncStream.makeStream(of: AddItemsProgress.self)

        let total = items.count
        let capturedItems = items

        Task { [weak self] in
            for (index, item) in capturedItems.enumerated() {
                await self?.addItem(id: item.id, text: item.text, metadata: item.metadata, embedding: item.embedding)
                continuation.yield(AddItemsProgress(id: item.id, current: index + 1, total: total))
            }
            continuation.finish()
        }

        return stream
    }

    // MARK: Read

    func getItem(id: String) -> IndexItem? {
        // O(1) lookup using secondary index
        guard let index = itemsByID[id], index < indexItems.count else {
            return nil
        }
        return indexItems[index]
    }

    func sample(_ count: Int) -> [IndexItem]? {
        return Array(indexItems.prefix(upTo: count))
    }

    // MARK: Update

    func updateItem(id: String, text: String? = nil, embedding: [Float]? = nil, metadata: [String: String]? = nil) {
        // Check if the provided embedding has the correct dimension
        if let embedding = embedding, embedding.count != dimension {
            print("Dimension mismatch, expected \(dimension), saw \(embedding.count)")
            return
        }

        // Find the item with the specified id using O(1) lookup
        guard let index = itemsByID[id], index < indexItems.count else {
            return
        }

        // Update the text if provided
        if let text = text {
            indexItems[index].text = text
        }

        // Update the embedding if provided
        if let embedding = embedding {
            indexItems[index].embedding = embedding
            // Update HNSW index - remove old and insert new
            if useHNSW, let hnsw = hnswIndex {
                hnsw.remove(id: id)
                hnsw.insert(id: id, embedding: embedding)
            }
        }

        // Update the metadata if provided
        if let metadata = metadata {
            indexItems[index].metadata = metadata
        }
    }

    // MARK: Delete

    func removeItem(id: String) {
        // Remove from HNSW first
        hnswIndex?.remove(id: id)

        // Remove from indexItems and rebuild secondary index
        indexItems.removeAll { $0.id == id }
        // Note: didSet will trigger rebuildItemIndex()
    }

    func removeAll() {
        hnswIndex = nil
        indexItems.removeAll()
        // Note: didSet will trigger rebuildItemIndex()
    }
}

// MARK: - Persistence

@available(macOS 13.0, iOS 16.0, *)
public extension SimilarityIndex {
    func saveIndex(toDirectory path: URL? = nil, name: String? = nil) throws -> URL {
        let indexName = name ?? self.indexName
        let basePath: URL

        if let specifiedPath = path {
            basePath = specifiedPath
        } else {
            // Default local path
            basePath = try getDefaultStoragePath()
        }

        let savedVectorStore = try vectorStore.saveIndex(items: indexItems, to: basePath, as: indexName)

        // Save HNSW index as companion file for fast loading
        if useHNSW, let hnsw = hnswIndex {
            let hnswURL = basePath.appendingPathComponent("\(indexName).hnsw")
            let hnswData = hnsw.serialize()
            try hnswData.write(to: hnswURL, options: .atomic)
            print("Saved HNSW index (\(hnswData.count) bytes) to \(hnswURL.absoluteString)")
        }

        print("Saved \(indexItems.count) index items to \(savedVectorStore.absoluteString)")

        return savedVectorStore
    }

    func loadIndex(fromDirectory path: URL? = nil, name: String? = nil) throws -> [IndexItem]? {
        if let indexPath = try getIndexPath(fromDirectory: path, name: name) {
            indexItems = try vectorStore.loadIndex(from: indexPath)

            // Try to load HNSW from companion file for instant access
            if useHNSW, !indexItems.isEmpty {
                let basePath = indexPath.deletingLastPathComponent()
                let indexName = name ?? self.indexName
                let hnswURL = basePath.appendingPathComponent("\(indexName).hnsw")

                if FileManager.default.fileExists(atPath: hnswURL.path),
                   let hnswData = try? Data(contentsOf: hnswURL) {
                    // Load existing HNSW graph
                    let hnsw = HNSWIndex(M: 16, efConstruction: 200, efSearch: 50)
                    do {
                        try hnsw.deserialize(from: hnswData, items: indexItems)
                        hnswIndex = hnsw
                        print("Loaded HNSW index from \(hnswURL.absoluteString)")
                    } catch {
                        print("Failed to deserialize HNSW, rebuilding: \(error)")
                        rebuildHNSWIndex()
                    }
                } else {
                    // No companion file, rebuild from items
                    rebuildHNSWIndex()
                }
            }
            return indexItems
        }

        return nil
    }

    /// This function returns the default location where the data from the loadIndex/saveIndex functions gets stored
    /// gets stored.
    /// - Parameters:
    ///   - fromDirectory: optional directory path where the file postfix is added to
    ///   - name: optional name
    ///
    /// - Returns: an optional URL
    func getIndexPath(fromDirectory path: URL? = nil, name: String? = nil) throws -> URL? {
        let indexName = name ?? self.indexName
        let basePath: URL

        if let specifiedPath = path {
            basePath = specifiedPath
        } else {
            // Default local path
            basePath = try getDefaultStoragePath()
        }
        return vectorStore.listIndexes(at: basePath).first(where: { $0.lastPathComponent.contains(indexName) })
    }

    private func getDefaultStoragePath() throws -> URL {
        let appName = Bundle.main.bundleIdentifier ?? "SimilaritySearchKit"
        let fileManager = FileManager.default
        let appSupportDirectory = try fileManager.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)

        let appSpecificDirectory = appSupportDirectory.appendingPathComponent(appName)

        if !fileManager.fileExists(atPath: appSpecificDirectory.path) {
            try fileManager.createDirectory(at: appSpecificDirectory, withIntermediateDirectories: true, attributes: nil)
        }

        return appSpecificDirectory
    }

    func estimatedSizeInBytes() -> Int {
        var totalSize = 0

        for item in indexItems {
            // Calculate the size of 'id' property
            let idSize = item.id.utf8.count

            // Calculate the size of 'text' property
            let textSize = item.text.utf8.count

            // Calculate the size of 'embedding' property
            let floatSize = MemoryLayout<Float>.size
            let embeddingSize = item.embedding.count * floatSize

            // Calculate the size of 'metadata' property
            let metadataSize = item.metadata.reduce(0) { size, keyValue -> Int in
                let keySize = keyValue.key.utf8.count
                let valueSize = keyValue.value.utf8.count
                return size + keySize + valueSize
            }

            totalSize += idSize + textSize + embeddingSize + metadataSize
        }

        return totalSize
    }
}
