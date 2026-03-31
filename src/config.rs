use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub quorum_threshold: usize,
    pub top_k: usize,
    pub cluster_threshold: f32,
    pub retrievers: Vec<RetrieverConfig>,
    pub ollama: OllamaConfig,
}

#[derive(Debug, Deserialize)]
pub struct RetrieverConfig {
    pub retriever_type: RetrieverType,
    pub chunk_size: usize,
    #[serde(default)]
    pub overlap: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RetrieverType {
    Dense,
    Bm25,
}

#[derive(Debug, Deserialize)]
pub struct OllamaConfig {
    pub url: String,
    pub model: String,
}








