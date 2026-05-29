use serde::Deserialize;

/// Top-level configuration for a [`QuorumRag`](crate::QuorumRag) pipeline.
///
/// This can be deserialized from a TOML file (see the `config.toml` shipped
/// with the repository) or constructed programmatically. Fields that are not
/// essential to the algorithm have sensible defaults so a minimal config only
/// needs to specify the retrievers and Ollama endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Minimum number of distinct retrievers that must agree on a cluster of
    /// evidence for it to survive quorum filtering.
    pub quorum_threshold: usize,
    /// Number of candidates each retriever returns per query.
    pub top_k: usize,
    /// Cosine-similarity threshold for merging candidates into the same cluster.
    pub cluster_threshold: f32,
    /// The retrievers that make up the quorum.
    pub retrievers: Vec<RetrieverConfig>,
    /// Ollama connection and model settings.
    pub ollama: OllamaConfig,

    /// Directory containing the `.txt` corpus files to index.
    #[serde(default = "default_corpus_dir")]
    pub corpus_dir: String,
    /// Directory where embedded-chunk caches are read from and written to.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
    /// Reciprocal-rank-fusion constant used when re-scoring candidates.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// How many chunks to embed concurrently in a single batch.
    #[serde(default = "default_embed_batch")]
    pub embed_batch: usize,
    /// Weight applied to a cluster's average score during ranking.
    #[serde(default = "default_rank_alpha")]
    pub rank_alpha: f32,
    /// Weight applied to a cluster's support count during ranking.
    #[serde(default = "default_rank_beta")]
    pub rank_beta: f32,
    /// Maximum number of clusters included in the generated context string.
    #[serde(default = "default_max_context_clusters")]
    pub max_context_clusters: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RetrieverConfig {
    pub retriever_type: RetrieverType,
    pub chunk_size: usize,
    #[serde(default)]
    pub overlap: usize,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RetrieverType {
    Dense,
    Bm25,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaConfig {
    /// Base URL of the Ollama server, e.g. `http://localhost:11434`.
    pub url: String,
    /// Model used for answer generation, e.g. `mistral`.
    pub model: String,
    /// Model used for text embeddings.
    #[serde(default = "default_embed_model")]
    pub embed_model: String,
}

fn default_corpus_dir() -> String {
    "data/corpus".to_string()
}
fn default_cache_dir() -> String {
    "data/cache".to_string()
}
fn default_rrf_k() -> f32 {
    60.0
}
fn default_embed_batch() -> usize {
    8
}
fn default_rank_alpha() -> f32 {
    0.7
}
fn default_rank_beta() -> f32 {
    0.3
}
fn default_max_context_clusters() -> usize {
    5
}
fn default_embed_model() -> String {
    "nomic-embed-text".to_string()
}
