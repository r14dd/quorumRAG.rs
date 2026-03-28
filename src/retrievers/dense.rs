use crate::embedding::EmbeddingClient;
use crate::models::{Candidate, Chunk};
use crate::retrievers::Retriever;
use anyhow::{Ok, Result};

pub struct DenseRetriever {
    id: String,
    chunks: Vec<Chunk>,
    embedder: EmbeddingClient,
}

impl DenseRetriever {
    pub fn new(
        id: &str,
        embedder: EmbeddingClient
    ) -> Self {
            Self {
            id: id.to_string(),
            chunks: Vec::new(),
            embedder
        }
    }

    pub async fn index(
        &mut self,
        texts: Vec<String>
    ) -> Result<()> {
        for (i, text) in texts.iter().enumerate() {
            let embedding = self.embedder.embed(text).await?;
            self.chunks.push(Chunk {
                id: format!("{}-{}", self.id, i),
                text: text.clone(),
                embedding
            });
        }
        Ok(())
    }
}


fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl Retriever for DenseRetriever {
    fn id(&self) -> &str {
        &self.id
    }

    fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<Candidate>> {
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .map(|chunk| (cosine_similarity(query_embedding, &chunk.embedding), chunk))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let candidates = scored
            .into_iter()
            .take(top_k)
            .map(|(score, chunk)| Candidate {
                chunk: chunk.clone(),
                score,
                retriever_id: self.id.to_string(),
            })
            .collect();

        Ok(candidates)
    }
}
