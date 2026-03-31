mod clustering;
mod config;
mod context;
mod corpus;
mod embedding;
mod evaluation;
mod generation;
mod models;
mod quorum;
mod ranking;
mod retrievers;

use anyhow::Result;
use config::{Config, RetrieverType};
use embedding::EmbeddingClient;
use evaluation::{EvalResult, load_eval_samples, print_results};
use generation::Generator;
use models::Query;
use retrievers::Retriever;
use retrievers::bm25::Bm25Retriever;
use retrievers::dense::DenseRetriever;

async fn run_pipeline(
    config: &Config,
    query_text: &str,
    use_quorum: bool,
) -> Result<(String, usize)> {
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
                let chunks =
                    corpus::load_chunks_from_dir("data", r_config.chunk_size, &retriever_id)?;
                retriever
                    .index(chunks.iter().map(|c| c.text.clone()).collect())
                    .await?;
                retrievers.push(Box::new(retriever));
            }
            RetrieverType::Bm25 => {
                let mut retriever = Bm25Retriever::new(&retriever_id)?;
                let chunks =
                    corpus::load_chunks_from_dir("data", r_config.chunk_size, &retriever_id)?;
                retriever.index(chunks)?;
                retrievers.push(Box::new(retriever));
            }
        }
    }

    let mut all_candidates = Vec::new();

    if use_quorum {
        for retriever in &retrievers {
            let candidates = retriever.retrieve(&query, config.top_k)?;
            all_candidates.extend(candidates);
        }
    } else {
        // baseline: only use first retriever
        let candidates = retrievers[0].retrieve(&query, config.top_k)?;
        all_candidates.extend(candidates);
    }

    for candidate in &mut all_candidates {
        if candidate.chunk.embedding.is_empty() {
            let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");
            candidate.chunk.embedding = embedder.embed(&candidate.chunk.text).await?;
        }
    }

    let clusters = clustering::cluster_candidates(all_candidates, config.cluster_threshold);

    let filtered = if use_quorum {
        quorum::filter_by_quorum(clusters, config.quorum_threshold)
    } else {
        clusters
    };

    let ranked = ranking::rank_clusters(filtered, 0.7, 0.3);
    let max_support = ranked.first().map(|c| c.support).unwrap_or(0);
    let context = context::build_context(&ranked, 3);

    Ok((context, max_support))
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    let samples = load_eval_samples("data/eval.json")?;
    let generator = Generator::new(&config.ollama.url, &config.ollama.model);

    let mut baseline_results = Vec::new();
    let mut quorum_results = Vec::new();

    for sample in &samples {
        // baseline
        let (baseline_context, _) = run_pipeline(&config, &sample.query, false).await?;
        let baseline_hit =
            evaluation::context_contains_relevant(&baseline_context, &sample.relevant);
        baseline_results.push(EvalResult {
            query: sample.query.clone(),
            hit: baseline_hit,
            support: 1,
        });

        // quorum
        let (quorum_context, support) = run_pipeline(&config, &sample.query, true).await?;
        let quorum_hit = evaluation::context_contains_relevant(&quorum_context, &sample.relevant);
        quorum_results.push(EvalResult {
            query: sample.query.clone(),
            hit: quorum_hit,
            support,
        });
    }

    print_results(&baseline_results, "Baseline (single retriever)");
    print_results(&quorum_results, "QuorumRAG");

    // demo generation on first query
    let (context, _) = run_pipeline(&config, &samples[0].query, true).await?;
    let answer = generator.generate(&context, &samples[0].query).await?;
    println!("\nSample answer: {}", answer);

    Ok(())
}
