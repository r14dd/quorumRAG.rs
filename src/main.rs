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
use futures::future::join_all;
use generation::Generator;
use models::Query;
use retrievers::Retriever;
use retrievers::bm25::Bm25Retriever;
use retrievers::dense::DenseRetriever;

const RRF_K: f32 = 60.0;
const EMBED_BATCH: usize = 8;

async fn embed_chunks_parallel(
    chunks: Vec<models::Chunk>,
    base_url: &str,
) -> Result<Vec<models::Chunk>> {
    let mut embedded = Vec::with_capacity(chunks.len());
    let total = chunks.len();

    for (batch_idx, batch) in chunks.chunks(EMBED_BATCH).enumerate() {
        let futures: Vec<_> = batch
            .iter()
            .map(|chunk| {
                let url = base_url.to_string();
                let text = chunk.text.clone();
                tokio::spawn(async move {
                    EmbeddingClient::new(&url, "nomic-embed-text")
                        .embed(&text)
                        .await
                })
            })
            .collect();

        let results = join_all(futures).await;
        for (chunk, result) in batch.iter().zip(results) {
            let mut c = chunk.clone();
            c.embedding = result.map_err(|e| anyhow::anyhow!(e))??;
            embedded.push(c);
        }

        let done = ((batch_idx + 1) * EMBED_BATCH).min(total);
        print!("\r  Embedded {}/{}", done, total);
    }
    println!();

    Ok(embedded)
}

async fn build_retrievers(config: &Config) -> Result<Vec<Box<dyn Retriever>>> {
    let mut retrievers: Vec<Box<dyn Retriever>> = Vec::new();

    for r_config in &config.retrievers {
        let retriever_id = format!(
            "{:?}-{}-ov{}",
            r_config.retriever_type, r_config.chunk_size, r_config.overlap
        );
        match r_config.retriever_type {
            RetrieverType::Dense => {
                let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");
                let mut retriever = DenseRetriever::new(&retriever_id, embedder);
                let chunks = if let Some(cached) = corpus::load_cache(&retriever_id) {
                    println!("Loaded cache for {}", retriever_id);
                    cached
                } else {
                    println!("Indexing {} ...", retriever_id);
                    let raw = corpus::load_chunks_from_dir(
                        "data/corpus",
                        r_config.chunk_size,
                        r_config.overlap,
                        &retriever_id,
                    )?;
                    let embedded =
                        embed_chunks_parallel(raw, &config.ollama.url).await?;
                    corpus::save_cache(&retriever_id, &embedded)?;
                    embedded
                };
                retriever.index_chunks(chunks).await?;
                retrievers.push(Box::new(retriever));
            }
            RetrieverType::Bm25 => {
                println!("Indexing {} ...", retriever_id);
                let mut retriever = Bm25Retriever::new(&retriever_id)?;
                let chunks = corpus::load_chunks_from_dir(
                    "data/corpus",
                    r_config.chunk_size,
                    r_config.overlap,
                    &retriever_id,
                )?;
                retriever.index(chunks)?;
                retrievers.push(Box::new(retriever));
            }
        }
    }

    if retrievers.is_empty() {
        anyhow::bail!("No retrievers configured in config.toml");
    }

    Ok(retrievers)
}

async fn run_query(
    config: &Config,
    retrievers: &[Box<dyn Retriever>],
    query_text: &str,
    use_quorum: bool,
) -> Result<(String, usize)> {
    let embedder = EmbeddingClient::new(&config.ollama.url, "nomic-embed-text");
    let query_embedding = embedder.embed(query_text).await?;
    let query = Query {
        text: query_text.to_string(),
        embedding: query_embedding,
    };

    let retriever_list: &[Box<dyn Retriever>] = if use_quorum {
        retrievers
    } else {
        &retrievers[..1]
    };

    // RRF: re-score each candidate by its rank within its retriever (1 / (k + rank))
    let mut all_candidates = Vec::new();
    for retriever in retriever_list {
        let candidates = retriever.retrieve(&query, config.top_k)?;
        for (rank, mut candidate) in candidates.into_iter().enumerate() {
            candidate.score = 1.0 / (RRF_K + (rank + 1) as f32);
            all_candidates.push(candidate);
        }
    }

    // Embed BM25 candidates that have no embedding
    let needs_embedding: Vec<usize> = all_candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.chunk.embedding.is_empty())
        .map(|(i, _)| i)
        .collect();

    if !needs_embedding.is_empty() {
        let futures: Vec<_> = needs_embedding
            .iter()
            .map(|&i| {
                let url = config.ollama.url.clone();
                let text = all_candidates[i].chunk.text.clone();
                tokio::spawn(async move {
                    EmbeddingClient::new(&url, "nomic-embed-text")
                        .embed(&text)
                        .await
                })
            })
            .collect();
        let results = join_all(futures).await;
        for (&i, result) in needs_embedding.iter().zip(results) {
            all_candidates[i].chunk.embedding =
                result.map_err(|e| anyhow::anyhow!(e))??;
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
    let context = context::build_context(&ranked, 5);

    Ok((context, max_support))
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    let retrievers = build_retrievers(&config).await?;

    let args: Vec<String> = std::env::args().collect();
    let query_arg = args
        .windows(2)
        .find(|w| w[0] == "--query")
        .map(|w| w[1].clone());

    if let Some(query_text) = query_arg {
        let generator = Generator::new(&config.ollama.url, &config.ollama.model);
        let (context, support) = run_query(&config, &retrievers, &query_text, true).await?;
        println!("Context (max support: {}):\n{}\n", support, context);
        let answer = generator.generate(&context, &query_text).await?;
        println!("Answer: {}", answer);
        return Ok(());
    }

    // Eval mode
    let samples = load_eval_samples("data/eval.json")?;
    let generator = Generator::new(&config.ollama.url, &config.ollama.model);

    let mut baseline_results = Vec::new();
    let mut quorum_results = Vec::new();

    for sample in &samples {
        let (baseline_context, _) =
            run_query(&config, &retrievers, &sample.query, false).await?;
        let baseline_hit =
            evaluation::context_contains_relevant(&baseline_context, &sample.relevant);
        baseline_results.push(EvalResult {
            query: sample.query.clone(),
            hit: baseline_hit,
            support: 1,
        });

        let (quorum_context, support) =
            run_query(&config, &retrievers, &sample.query, true).await?;
        let quorum_hit =
            evaluation::context_contains_relevant(&quorum_context, &sample.relevant);
        quorum_results.push(EvalResult {
            query: sample.query.clone(),
            hit: quorum_hit,
            support,
        });
    }

    print_results(&baseline_results, "Baseline (single retriever)");
    print_results(&quorum_results, "QuorumRAG");

    let (context, _) = run_query(&config, &retrievers, &samples[0].query, true).await?;
    let answer = generator.generate(&context, &samples[0].query).await?;
    println!("\nSample answer: {}", answer);

    Ok(())
}
