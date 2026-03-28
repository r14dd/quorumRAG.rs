pub mod dense;
pub mod bm25;

use crate::models::{Candidate, Query};
use anyhow::Result;

pub trait Retriever: Send + Sync {
    fn id(&self) -> &str;
    fn retrieve(&self, query: &Query, top_k: usize) -> Result<Vec<Candidate>>;
}
