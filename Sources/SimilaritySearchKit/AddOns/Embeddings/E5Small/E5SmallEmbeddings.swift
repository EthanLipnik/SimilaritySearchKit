//
//  E5SmallEmbeddings.swift
//  SimilaritySearchKit
//
//  E5-Small v2 embedding model wrapper.
//  Model source: https://huggingface.co/intfloat/e5-small-v2
//
//  To generate the .mlpackage, run:
//  python Scripts/convert_hf_to_coreml.py --model "intfloat/e5-small-v2" --output e5_small_v2.mlpackage --validate
//

import CoreML
import Foundation
import SimilaritySearchKit

/// E5-Small v2 embeddings model.
///
/// A 384-dimensional embedding model optimized for text retrieval.
/// Trained by Microsoft, this model provides excellent quality for English text.
/// Note: E5 models work best when queries are prefixed with "query: " and passages with "passage: "
@available(macOS 12.0, iOS 15.0, *)
public class E5SmallEmbeddings: EmbeddingsProtocol {
    public let model: e5_small_v2
    public let tokenizer: BertTokenizer
    public let inputDimention: Int = 512
    public let outputDimention: Int = 384

    /// Initialize the E5-Small embeddings model.
    /// - Parameter computeUnits: The compute units to use for inference (CPU, GPU, Neural Engine, or all).
    public init(computeUnits: MLComputeUnits = .all) {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        do {
            self.model = try e5_small_v2(configuration: modelConfig)
        } catch {
            fatalError("Failed to load E5-Small Core ML model. Error: \(error.localizedDescription)")
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
        let inputFeatures = e5_small_v2Input(input_ids: inputIds, attention_mask: attentionMask)

        guard let output = try? model.prediction(input: inputFeatures) else {
            return nil
        }

        let embeddings = output.embeddings
        return (0..<embeddings.count).map { Float(embeddings[$0].floatValue) }
    }
}
