//
//  HNSWIndex.swift
//  SimilaritySearchKit
//
//  Created by Claude on 11/28/25.
//

import Foundation
import Accelerate

/// Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search.
///
/// HNSW provides O(log n) search complexity compared to O(n) brute force,
/// achieving ~100x speedup for large indexes with minimal accuracy loss.
///
/// Reference: Malkov, Y.A., Yashunin, D.A. (2018). Efficient and robust approximate
/// nearest neighbor search using Hierarchical Navigable Small World graphs.
public final class HNSWIndex: @unchecked Sendable {

    // MARK: - Configuration

    /// Maximum number of connections per node at each layer
    public let M: Int

    /// Maximum connections at layer 0 (typically 2*M)
    public let M0: Int

    /// Beam width during construction (higher = better quality, slower build)
    public let efConstruction: Int

    /// Beam width during search (higher = better recall, slower search)
    public var efSearch: Int

    /// Multiplier for level generation: 1/ln(M)
    private let levelMultiplier: Double

    // MARK: - Graph Structure

    /// All nodes in the graph
    private var nodes: [HNSWNode] = []

    /// Entry point index (node at highest level)
    private var entryPointIndex: Int?

    /// Current maximum level in the graph
    private var maxLevel: Int = 0

    /// Map from item ID to node index for O(1) lookup
    private var idToIndex: [String: Int] = [:]

    /// Lock for thread-safe modifications
    private let lock = NSLock()

    // MARK: - Node Structure

    struct HNSWNode {
        let id: String
        var embedding: [Float]
        /// connections[level] contains indices of neighbors at that level
        var connections: [[Int]]
        let level: Int

        init(id: String, embedding: [Float], level: Int) {
            self.id = id
            self.embedding = embedding
            self.level = level
            self.connections = Array(repeating: [], count: level + 1)
        }
    }

    // MARK: - Initialization

    /// Creates a new HNSW index with specified parameters.
    ///
    /// - Parameters:
    ///   - M: Max connections per layer (default 16, higher = better quality but more memory)
    ///   - efConstruction: Construction beam width (default 200)
    ///   - efSearch: Search beam width (default 50, can be tuned at query time)
    public init(M: Int = 16, efConstruction: Int = 200, efSearch: Int = 50) {
        self.M = M
        self.M0 = 2 * M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.levelMultiplier = 1.0 / log(Double(M))
    }

    // MARK: - Properties

    /// Number of nodes in the index
    public var count: Int {
        nodes.count
    }

    /// Whether the index is empty
    public var isEmpty: Bool {
        nodes.isEmpty
    }

    // MARK: - Insert

    /// Inserts a new item into the index.
    ///
    /// - Parameters:
    ///   - id: Unique identifier for the item
    ///   - embedding: The embedding vector
    public func insert(id: String, embedding: [Float]) {
        lock.lock()
        defer { lock.unlock() }

        // Skip if already exists
        if idToIndex[id] != nil {
            return
        }

        let newLevel = randomLevel()
        let newNode = HNSWNode(id: id, embedding: embedding, level: newLevel)
        let newIndex = nodes.count

        // Handle first node
        guard let entryIndex = entryPointIndex else {
            nodes.append(newNode)
            idToIndex[id] = newIndex
            entryPointIndex = newIndex
            maxLevel = newLevel
            return
        }

        var currentIndex = entryIndex
        var currentLevel = maxLevel

        // Phase 1: Traverse from top level down to newLevel+1, finding closest node
        while currentLevel > newLevel {
            let changed = greedySearchClosest(
                query: embedding,
                startIndex: currentIndex,
                level: currentLevel
            )
            currentIndex = changed
            currentLevel -= 1
        }

        // Phase 2: From level min(maxLevel, newLevel) down to 0, find and connect neighbors
        // First, append the new node so it's accessible during neighbor updates
        nodes.append(newNode)
        idToIndex[id] = newIndex

        var level = min(maxLevel, newLevel)
        while level >= 0 {
            let candidates = searchLayer(
                query: embedding,
                entryPoints: [currentIndex],
                ef: efConstruction,
                level: level
            )

            let maxConnections = level == 0 ? M0 : M
            let neighbors = selectNeighbors(
                candidates: candidates,
                query: embedding,
                M: maxConnections
            )

            // Connect new node to neighbors
            nodes[newIndex].connections[level] = neighbors

            // Connect neighbors back to new node (with pruning if needed)
            for neighborIndex in neighbors {
                // Bounds check
                guard neighborIndex >= 0 && neighborIndex < nodes.count else { continue }
                guard level < nodes[neighborIndex].connections.count else { continue }

                var neighborConns = nodes[neighborIndex].connections[level]
                neighborConns.append(newIndex)

                if neighborConns.count > maxConnections {
                    // Prune connections using heuristic
                    neighborConns = selectNeighbors(
                        candidates: neighborConns,
                        query: nodes[neighborIndex].embedding,
                        M: maxConnections
                    )
                }
                nodes[neighborIndex].connections[level] = neighborConns
            }

            if level > 0 {
                currentIndex = neighbors.first ?? currentIndex
            }
            level -= 1
        }

        // Update entry point if new node has higher level
        if newLevel > maxLevel {
            entryPointIndex = newIndex
            maxLevel = newLevel
        }
    }

    // MARK: - Search

    /// Searches for the k nearest neighbors to the query.
    ///
    /// - Parameters:
    ///   - query: The query embedding vector
    ///   - k: Number of results to return
    /// - Returns: Array of (id, distance) tuples sorted by distance
    public func search(query: [Float], k: Int) -> [(id: String, distance: Float)] {
        lock.lock()
        defer { lock.unlock() }

        guard let entryIndex = entryPointIndex else {
            return []
        }

        var currentIndex = entryIndex
        var currentLevel = maxLevel

        // Traverse from top to level 1
        while currentLevel > 0 {
            currentIndex = greedySearchClosest(
                query: query,
                startIndex: currentIndex,
                level: currentLevel
            )
            currentLevel -= 1
        }

        // Search at level 0 with efSearch beam width
        let candidates = searchLayer(
            query: query,
            entryPoints: [currentIndex],
            ef: max(efSearch, k),
            level: 0
        )

        // Return top k results
        let results = candidates.prefix(k).map { index -> (id: String, distance: Float) in
            let node = nodes[index]
            let dist = euclideanDistance(query, node.embedding)
            return (id: node.id, distance: dist)
        }

        return Array(results)
    }

    // MARK: - Remove

    /// Removes an item from the index by ID.
    ///
    /// Note: HNSW removal is approximate - it marks the node as deleted
    /// but doesn't restructure the graph. For full cleanup, rebuild the index.
    ///
    /// - Parameter id: The ID of the item to remove
    /// - Returns: True if the item was found and removed
    @discardableResult
    public func remove(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        guard let index = idToIndex[id] else {
            return false
        }

        // Remove from ID map
        idToIndex.removeValue(forKey: id)

        // Remove connections to this node from all neighbors
        let node = nodes[index]
        for level in 0...node.level {
            for neighborIndex in node.connections[level] {
                nodes[neighborIndex].connections[level].removeAll { $0 == index }
            }
        }

        // Clear the node's connections (leave in array to maintain indices)
        nodes[index].connections = Array(repeating: [], count: node.level + 1)

        // Update entry point if we removed it
        if entryPointIndex == index {
            // Find new entry point at highest level
            entryPointIndex = nodes.enumerated()
                .filter { !$0.element.connections.isEmpty || $0.offset != index }
                .max(by: { $0.element.level < $1.element.level })?
                .offset
            maxLevel = entryPointIndex.map { nodes[$0].level } ?? 0
        }

        return true
    }

    // MARK: - Persistence

    /// Serializes the HNSW graph to Data for persistence.
    public func serialize() -> Data {
        lock.lock()
        defer { lock.unlock() }

        var data = Data()

        // Write header
        var nodeCount = UInt32(nodes.count)
        var maxLevelValue = UInt32(maxLevel)
        var entryPoint = UInt32(entryPointIndex ?? 0)
        var mValue = UInt32(M)
        var efSearchValue = UInt32(efSearch)

        data.append(Data(bytes: &nodeCount, count: 4))
        data.append(Data(bytes: &maxLevelValue, count: 4))
        data.append(Data(bytes: &entryPoint, count: 4))
        data.append(Data(bytes: &mValue, count: 4))
        data.append(Data(bytes: &efSearchValue, count: 4))

        // Write each node's connections
        for node in nodes {
            var levelCount = UInt8(node.connections.count)
            data.append(Data(bytes: &levelCount, count: 1))

            for level in node.connections {
                var connCount = UInt16(level.count)
                data.append(Data(bytes: &connCount, count: 2))

                for conn in level {
                    var connIndex = UInt32(conn)
                    data.append(Data(bytes: &connIndex, count: 4))
                }
            }
        }

        return data
    }

    /// Deserializes the HNSW graph from Data.
    ///
    /// - Parameters:
    ///   - data: Serialized graph data
    ///   - items: The IndexItems (must match order from serialization)
    public func deserialize(from data: Data, items: [IndexItem]) throws {
        lock.lock()
        defer { lock.unlock() }

        guard data.count >= 20 else {
            throw HNSWError.invalidData("Data too small for header")
        }

        var offset = 0

        // Use loadUnaligned to handle potentially misaligned data
        let nodeCount = data[offset..<(offset+4)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        offset += 4

        guard Int(nodeCount) == items.count else {
            throw HNSWError.invalidData("Node count mismatch: \(nodeCount) vs \(items.count)")
        }

        maxLevel = Int(data[offset..<(offset+4)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        offset += 4

        entryPointIndex = Int(data[offset..<(offset+4)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        offset += 4

        // Skip M and efSearch (already set in init)
        offset += 8

        // Rebuild nodes with connections
        nodes = []
        nodes.reserveCapacity(items.count)
        idToIndex = [:]

        for (index, item) in items.enumerated() {
            guard offset < data.count else {
                throw HNSWError.invalidData("Unexpected end of data at node \(index)")
            }

            let levelCount = Int(data[offset])
            offset += 1

            var connections: [[Int]] = []
            for _ in 0..<levelCount {
                guard offset + 2 <= data.count else {
                    throw HNSWError.invalidData("Unexpected end of data reading connections")
                }
                let connCount = Int(data[offset..<(offset+2)].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) })
                offset += 2

                var levelConns: [Int] = []
                for _ in 0..<connCount {
                    guard offset + 4 <= data.count else {
                        throw HNSWError.invalidData("Unexpected end of data reading connection index")
                    }
                    let connIndex = Int(data[offset..<(offset+4)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
                    offset += 4
                    levelConns.append(connIndex)
                }
                connections.append(levelConns)
            }

            var node = HNSWNode(
                id: item.id,
                embedding: item.embedding,
                level: max(0, levelCount - 1)
            )
            node.connections = connections
            nodes.append(node)
            idToIndex[item.id] = index
        }
    }

    // MARK: - Private Methods

    /// Generates a random level for a new node using exponential distribution.
    private func randomLevel() -> Int {
        let r = Double.random(in: 0..<1)
        return Int(floor(-log(r) * levelMultiplier))
    }

    /// Greedily finds the closest node starting from startIndex at given level.
    private func greedySearchClosest(query: [Float], startIndex: Int, level: Int) -> Int {
        guard startIndex >= 0 && startIndex < nodes.count else { return startIndex }

        var currentIndex = startIndex
        var currentDist = euclideanDistance(query, nodes[currentIndex].embedding)

        var changed = true
        while changed {
            changed = false

            let node = nodes[currentIndex]
            guard level < node.connections.count else { break }

            for neighborIndex in node.connections[level] {
                guard neighborIndex >= 0 && neighborIndex < nodes.count else { continue }
                let neighborDist = euclideanDistance(query, nodes[neighborIndex].embedding)
                if neighborDist < currentDist {
                    currentDist = neighborDist
                    currentIndex = neighborIndex
                    changed = true
                }
            }
        }

        return currentIndex
    }

    /// Searches a layer using beam search.
    ///
    /// Returns indices of the ef closest nodes.
    private func searchLayer(query: [Float], entryPoints: [Int], ef: Int, level: Int) -> [Int] {
        var visited = Set(entryPoints)

        // Candidates: min-heap ordered by distance (closest first)
        var candidates = Heap<(distance: Float, index: Int)>(sort: { $0.distance < $1.distance })

        // Results: max-heap ordered by distance (farthest first for easy pruning)
        var results = Heap<(distance: Float, index: Int)>(sort: { $0.distance > $1.distance })

        for ep in entryPoints where ep >= 0 && ep < nodes.count {
            let dist = euclideanDistance(query, nodes[ep].embedding)
            candidates.insert((dist, ep))
            results.insert((dist, ep))
        }

        while let (cDist, cIndex) = candidates.popMin() {
            // Stop if the closest candidate is farther than the farthest result
            if let farthest = results.max, cDist > farthest.distance {
                break
            }

            // Bounds check before accessing connections
            guard cIndex >= 0 && cIndex < nodes.count else { continue }
            let node = nodes[cIndex]
            guard level < node.connections.count else { continue }

            // Explore neighbors
            for neighborIndex in node.connections[level] {
                guard neighborIndex >= 0 && neighborIndex < nodes.count else { continue }
                guard !visited.contains(neighborIndex) else { continue }
                visited.insert(neighborIndex)

                let neighborDist = euclideanDistance(query, nodes[neighborIndex].embedding)
                let farthestDist = results.max?.distance ?? .infinity

                if neighborDist < farthestDist || results.count < ef {
                    candidates.insert((neighborDist, neighborIndex))
                    results.insert((neighborDist, neighborIndex))

                    if results.count > ef {
                        results.popMax()
                    }
                }
            }
        }

        // Extract indices sorted by distance
        var sortedResults: [(distance: Float, index: Int)] = []
        while let item = results.popMax() {
            sortedResults.append(item)
        }
        return sortedResults.reversed().map(\.index)
    }

    /// Selects the best M neighbors from candidates using simple heuristic.
    private func selectNeighbors(candidates: [Int], query: [Float], M: Int) -> [Int] {
        // Filter out any invalid indices
        let validCandidates = candidates.filter { $0 >= 0 && $0 < nodes.count }

        if validCandidates.count <= M {
            return validCandidates
        }

        // Sort by distance and take closest M
        let sorted = validCandidates.sorted { a, b in
            euclideanDistance(query, nodes[a].embedding) < euclideanDistance(query, nodes[b].embedding)
        }
        return Array(sorted.prefix(M))
    }

    /// Computes Euclidean distance between two vectors using Accelerate.
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }

        var result: Float = 0
        var diff = [Float](repeating: 0, count: a.count)

        // Compute a - b
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))

        // Compute sum of squares
        vDSP_dotpr(diff, 1, diff, 1, &result, vDSP_Length(a.count))

        return sqrt(result)
    }
}

// MARK: - Errors

extension HNSWIndex {
    public enum HNSWError: LocalizedError {
        case invalidData(String)

        public var errorDescription: String? {
            switch self {
            case .invalidData(let message):
                return "Invalid HNSW data: \(message)"
            }
        }
    }
}

// MARK: - Simple Heap Implementation

/// A simple binary heap for HNSW's priority queues.
private struct Heap<Element> {
    private var elements: [Element] = []
    private let sort: (Element, Element) -> Bool

    init(sort: @escaping (Element, Element) -> Bool) {
        self.sort = sort
    }

    var isEmpty: Bool { elements.isEmpty }
    var count: Int { elements.count }
    var max: Element? { elements.first }

    mutating func insert(_ element: Element) {
        elements.append(element)
        siftUp(from: elements.count - 1)
    }

    @discardableResult
    mutating func popMax() -> Element? {
        guard !elements.isEmpty else { return nil }
        if elements.count == 1 {
            return elements.removeLast()
        }
        let result = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return result
    }

    mutating func popMin() -> Element? {
        popMax() // For min-heap, sort is reversed
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2

        while child > 0 && sort(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index

        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var candidate = parent

            if left < elements.count && sort(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < elements.count && sort(elements[right], elements[candidate]) {
                candidate = right
            }

            if candidate == parent {
                return
            }

            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }
}
