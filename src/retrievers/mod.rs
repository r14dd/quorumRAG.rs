use crate::models::Candidate;
use anyhow::Result;

pub trait Retriever: Send + Sync {
    fn id(&self) -> &str;
    fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<Candidate>>;
}
