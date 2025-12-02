//
//  BGESmallEmbeddings.swift
//  SimilaritySearchKit
//
//  BGE-Small-EN v1.5 embedding model wrapper.
//  Model source: https://huggingface.co/BAAI/bge-small-en-v1.5
//
//  To generate the .mlpackage, run:
//  python Scripts/convert_hf_to_coreml.py --model "BAAI/bge-small-en-v1.5" --output bge_small_en.mlpackage --validate
//

import CoreML
import Foundation
import SimilaritySearchKit

/// BGE-Small English v1.5 embeddings model.
///
/// A 384-dimensional embedding model optimized for semantic search and retrieval.
/// Trained by BAAI, this model provides excellent quality for English text.
@available(macOS 12.0, iOS 15.0, *)
public class BGESmallEmbeddings: EmbeddingsProtocol {
    public let model: bge_small_en
    public let tokenizer: BertTokenizer
    public let inputDimention: Int = 512
    public let outputDimention: Int = 384

    /// Initialize the BGE-Small embeddings model.
    /// - Parameter computeUnits: The compute units to use for inference (CPU, GPU, Neural Engine, or all).
    public init(computeUnits: MLComputeUnits = .all) {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        do {
            self.model = try bge_small_en(configuration: modelConfig)
        } catch {
            fatalError("Failed to load BGE-Small Core ML model. Error: \(error.localizedDescription)")
        }

        self.tokenizer = BertTokenizer()
    }

    // MARK: - EmbeddingsProtocol

    public func encode(sentence: String) async -> [Float]? {
        // Encode input text as BERT tokens
        let inputTokens = tokenizer.buildModelTokens(sentence: sentence)
        let (inputIds, attentionMask) = tokenizer.buildModelInputs(from: inputTokens)

        // Run inference
        return generateEmbeddings(inputIds: inputIds, attentionMask: attentionMask)
    }

    public func generateEmbeddings(inputIds: MLMultiArray, attentionMask: MLMultiArray) -> [Float]? {
        let inputFeatures = bge_small_enInput(input_ids: inputIds, attention_mask: attentionMask)

        guard let output = try? model.prediction(input: inputFeatures) else {
            return nil
        }

        let embeddings = output.embeddings
        return (0..<embeddings.count).map { Float(embeddings[$0].floatValue) }
    }
}
