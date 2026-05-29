use serde::{Deserialize, Serialize};

/// A chunk of text from the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// A query with its text and precomputed embedding.
#[derive(Debug, Clone)]
pub struct Query {
    pub text: String,
    pub embedding: Vec<f32>,
}
/// This is going to be a candidate returned by a single retriever.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub chunk: Chunk,
    pub score: f32,
    pub retriever_id: String,
}

/// A cluster of semantically similar candidates from multiple retrievers.
#[derive(Debug, Clone)]
pub struct EvidenceCluster {
    pub members: Vec<Candidate>,
    pub support: usize,
    pub avg_score: f32,
}
