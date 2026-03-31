mod clustering;
mod config;
mod context;
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
use retrievers::dense::DenseRetriever;
use retrievers::bm25::Bm25Retriever;
use retrievers::Retriever;

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    // sample corpus
    let corpus = vec![
        "The Eiffel Tower is located in Paris, France.".to_string(),
        "Paris is the capital city of France.".to_string(),
        "The Louvre museum is also in Paris.".to_string(),
        "Berlin is the capital of Germany.".to_string(),
        "The Brandenburg Gate is a famous landmark in Berlin.".to_string(),
        "Germany is a country in central Europe.".to_string(),
    ];

    let query_text = "Where is the Eiffel Tower?";

    // build embedding client
    let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");

    // embed query
    let query_embedding = embedder.embed(query_text).await?;
    let query = Query {
        text: query_text.to_string(),
        embedding: query_embedding,
    };

    // build and index retrievers
    let mut retrievers: Vec<Box<dyn Retriever>> = Vec::new();

    for r_config in &config.retrievers {
        match r_config.retriever_type {
            RetrieverType::Dense => {
                let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");
                let mut retriever = DenseRetriever::new(
                    &format!("dense-{}", r_config.chunk_size),
                    embedder,
                );
                retriever.index(corpus.clone()).await?;
                retrievers.push(Box::new(retriever));
            }
            RetrieverType::Bm25 => {
                let mut retriever = Bm25Retriever::new("bm25")?;
                let chunks = corpus
                    .iter()
                    .enumerate()
                    .map(|(i, text)| models::Chunk {
                        id: format!("bm25-{}", i),
                        text: text.clone(),
                        embedding: vec![],
                    })
                    .collect();
                retriever.index(chunks)?;
                retrievers.push(Box::new(retriever));
            }
        }
    }

    // run all retrievers
    let mut all_candidates = Vec::new();
    for retriever in &retrievers {
        let candidates = retriever.retrieve(&query, config.top_k)?;
        all_candidates.extend(candidates);
    }

    println!("Total candidates: {}", all_candidates.len());

    // cluster
    let clusters = clustering::cluster_candidates(all_candidates, config.cluster_threshold);
    println!("Clusters: {}", clusters.len());

    // quorum filter
    let filtered = quorum::filter_by_quorum(clusters, config.quorum_threshold);
    println!("After quorum filter: {}", filtered.len());

    // rank
    let ranked = ranking::rank_clusters(filtered, 0.7, 0.3);

    // build context
    let context = context::build_context(&ranked, 3);
    println!("\nContext:\n{}", context);

    // generate
    let generator = Generator::new(&config.ollama.url, &config.ollama.model);
    let answer = generator.generate(&context, query_text).await?;
    println!("\nAnswer:\n{}", answer);

    Ok(())
}
