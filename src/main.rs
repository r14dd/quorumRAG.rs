mod clustering;
mod config;
mod context;
mod corpus;
mod embedding;
mod generation;
mod models;
mod quorum;
mod ranking;
mod retrievers;

use anyhow::Result;
use config::{Config, RetrieverType};
use embedding::EmbeddingClient;
use generation::Generator;
use models::Query;
use retrievers::bm25::Bm25Retriever;
use retrievers::dense::DenseRetriever;
use retrievers::Retriever;

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    let query_text = "Where is the Eiffel Tower?";

    let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");

    let query_embedding = embedder.embed(query_text).await?;
    let query = Query {
        text: query_text.to_string(),
        embedding: query_embedding,
    };

    let mut retrievers: Vec<Box<dyn Retriever>> = Vec::new();

    for r_config in &config.retrievers {
        let retriever_id = format!("{:?}-{}", r_config.retriever_type, r_config.chunk_size);

        match r_config.retriever_type {
            RetrieverType::Dense => {
                let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");
                let mut retriever = DenseRetriever::new(&retriever_id, embedder);
                let chunks = corpus::load_chunks_from_dir("data", r_config.chunk_size, &retriever_id)?;
                retriever.index(chunks.iter().map(|c| c.text.clone()).collect()).await?;
                retrievers.push(Box::new(retriever));
            }
            RetrieverType::Bm25 => {
                let mut retriever = Bm25Retriever::new(&retriever_id)?;
                let chunks = corpus::load_chunks_from_dir("data", r_config.chunk_size, &retriever_id)?;
                retriever.index(chunks)?;
                retrievers.push(Box::new(retriever));
            }
        }
    }

    let mut all_candidates = Vec::new();
    for retriever in &retrievers {
        let candidates = retriever.retrieve(&query, config.top_k)?;
        all_candidates.extend(candidates);
    }

    // embed BM25 candidates that have empty embeddings
    for candidate in &mut all_candidates {
        if candidate.chunk.embedding.is_empty() {
            candidate.chunk.embedding = embedder.embed(&candidate.chunk.text).await?;
        }
    }

    println!("Total candidates: {}", all_candidates.len());

    let clusters = clustering::cluster_candidates(all_candidates, config.cluster_threshold);
    println!("Clusters: {}", clusters.len());

    let filtered = quorum::filter_by_quorum(clusters, config.quorum_threshold);
    println!("After quorum filter: {}", filtered.len());

    let ranked = ranking::rank_clusters(filtered, 0.7, 0.3);

    let context = context::build_context(&ranked, 3);
    println!("\nContext:\n{}", context);

    let generator = Generator::new(&config.ollama.url, &config.ollama.model);
    let answer = generator.generate(&context, query_text).await?;
    println!("\nAnswer:\n{}", answer);

    Ok(())
}
