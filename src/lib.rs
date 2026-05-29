//! # quorumrag
//!
//! Quorum-based retrieval-augmented generation.
//!
//! Instead of trusting a single retriever, QuorumRAG runs several retrievers
//! (dense embeddings at different chunk sizes, plus BM25) over the same corpus,
//! fuses their results with reciprocal rank fusion, clusters semantically
//! similar evidence, and then keeps only the clusters that **multiple
//! retrievers agree on** (the "quorum"). The surviving evidence is ranked and
//! turned into a context string that can be handed to a generator.
//!
//! The crate talks to a local [Ollama](https://ollama.com) server for both
//! embeddings and generation.
//!
//! ## Quick start
//!
//! ```no_run
//! use quorumrag::{Config, QuorumRag};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let config_str = std::fs::read_to_string("config.toml")?;
//! let config: Config = toml::from_str(&config_str)?;
//!
//! let rag = QuorumRag::build(config).await?;
//! let answer = rag.answer("What is the capital of France?").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! For finer control, [`QuorumRag::retrieve`] returns the ranked
//! [`EvidenceCluster`]s and the assembled context without invoking the
//! generator.

pub mod clustering;
pub mod config;
pub mod context;
pub mod corpus;
pub mod embedding;
pub mod evaluation;
pub mod generation;
pub mod models;
pub mod quorum;
pub mod ranking;
pub mod retrievers;

use anyhow::Result;
use futures::future::join_all;

pub use config::{Config, OllamaConfig, RetrieverConfig, RetrieverType};
pub use embedding::EmbeddingClient;
pub use generation::Generator;
pub use models::{Candidate, Chunk, EvidenceCluster, Query};

use retrievers::Retriever;
use retrievers::bm25::Bm25Retriever;
use retrievers::dense::DenseRetriever;

/// The outcome of a retrieval: the assembled context plus the structured
/// evidence behind it.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The context string built from the top-ranked clusters, ready to feed to
    /// a generator.
    pub context: String,
    /// Support count of the strongest cluster (how many retrievers agreed on
    /// it). `0` when nothing was retrieved.
    pub max_support: usize,
    /// The ranked clusters that produced `context`.
    pub clusters: Vec<EvidenceCluster>,
}

/// A fully indexed QuorumRAG pipeline: retrievers, an embedder, and a generator
/// wired together from a [`Config`].
pub struct QuorumRag {
    config: Config,
    retrievers: Vec<Box<dyn Retriever>>,
    embedder: EmbeddingClient,
    generator: Generator,
}

impl QuorumRag {
    /// Build the pipeline: construct every configured retriever and index the
    /// corpus found in [`Config::corpus_dir`].
    ///
    /// Dense retrievers reuse a per-retriever embedding cache under
    /// [`Config::cache_dir`] when one exists, and write one otherwise.
    ///
    /// Requires a reachable Ollama server at [`OllamaConfig::url`].
    pub async fn build(config: Config) -> Result<Self> {
        let embedder = EmbeddingClient::new(&config.ollama.url, &config.ollama.embed_model);
        let generator = Generator::new(&config.ollama.url, &config.ollama.model);
        let retrievers = build_retrievers(&config).await?;
        Ok(Self {
            config,
            retrievers,
            embedder,
            generator,
        })
    }

    /// Access the configuration this pipeline was built from.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run the full quorum retrieval for `query_text` and return the ranked
    /// clusters and assembled context.
    ///
    /// When `use_quorum` is `false`, only the first configured retriever is
    /// used and quorum filtering is skipped — useful as a single-retriever
    /// baseline for evaluation.
    pub async fn retrieve(&self, query_text: &str, use_quorum: bool) -> Result<RetrievalResult> {
        let query_embedding = self.embedder.embed(query_text).await?;
        let query = Query {
            text: query_text.to_string(),
            embedding: query_embedding,
        };

        let retriever_list: &[Box<dyn Retriever>] = if use_quorum {
            &self.retrievers
        } else {
            &self.retrievers[..1]
        };

        // RRF: re-score each candidate by its rank within its retriever
        // (1 / (k + rank)).
        let mut all_candidates = Vec::new();
        for retriever in retriever_list {
            let candidates = retriever.retrieve(&query, self.config.top_k)?;
            for (rank, mut candidate) in candidates.into_iter().enumerate() {
                candidate.score = 1.0 / (self.config.rrf_k + (rank + 1) as f32);
                all_candidates.push(candidate);
            }
        }

        // Embed BM25 candidates that have no embedding yet so they can be
        // clustered alongside the dense candidates.
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
                    let url = self.config.ollama.url.clone();
                    let model = self.config.ollama.embed_model.clone();
                    let text = all_candidates[i].chunk.text.clone();
                    tokio::spawn(
                        async move { EmbeddingClient::new(&url, &model).embed(&text).await },
                    )
                })
                .collect();
            let results = join_all(futures).await;
            for (&i, result) in needs_embedding.iter().zip(results) {
                all_candidates[i].chunk.embedding = result.map_err(|e| anyhow::anyhow!(e))??;
            }
        }

        let clusters =
            clustering::cluster_candidates(all_candidates, self.config.cluster_threshold);

        let filtered = if use_quorum {
            quorum::filter_by_quorum(clusters, self.config.quorum_threshold)
        } else {
            clusters
        };

        let ranked =
            ranking::rank_clusters(filtered, self.config.rank_alpha, self.config.rank_beta);
        let max_support = ranked.first().map(|c| c.support).unwrap_or(0);
        let context = context::build_context(&ranked, self.config.max_context_clusters);

        Ok(RetrievalResult {
            context,
            max_support,
            clusters: ranked,
        })
    }

    /// Retrieve quorum-backed context for `query_text` and generate an answer
    /// from it.
    pub async fn answer(&self, query_text: &str) -> Result<String> {
        let result = self.retrieve(query_text, true).await?;
        self.generator.generate(&result.context, query_text).await
    }

    /// The generator this pipeline uses, for callers that want to generate
    /// against a context they assembled themselves.
    pub fn generator(&self) -> &Generator {
        &self.generator
    }
}

/// Embed `chunks` in batches of [`Config::embed_batch`], embedding each batch
/// concurrently against the Ollama server at `base_url`.
pub async fn embed_chunks_parallel(
    chunks: Vec<Chunk>,
    base_url: &str,
    model: &str,
    batch_size: usize,
) -> Result<Vec<Chunk>> {
    let mut embedded = Vec::with_capacity(chunks.len());
    let total = chunks.len();
    let batch_size = batch_size.max(1);

    for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
        let futures: Vec<_> = batch
            .iter()
            .map(|chunk| {
                let url = base_url.to_string();
                let model = model.to_string();
                let text = chunk.text.clone();
                tokio::spawn(async move { EmbeddingClient::new(&url, &model).embed(&text).await })
            })
            .collect();

        let results = join_all(futures).await;
        for (chunk, result) in batch.iter().zip(results) {
            let mut c = chunk.clone();
            c.embedding = result.map_err(|e| anyhow::anyhow!(e))??;
            embedded.push(c);
        }

        let done = ((batch_idx + 1) * batch_size).min(total);
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
                let embedder =
                    EmbeddingClient::new(&config.ollama.url, &config.ollama.embed_model);
                let mut retriever = DenseRetriever::new(&retriever_id, embedder);
                let chunks = if let Some(cached) =
                    corpus::load_cache(&config.cache_dir, &retriever_id)
                {
                    println!("Loaded cache for {}", retriever_id);
                    cached
                } else {
                    println!("Indexing {} ...", retriever_id);
                    let raw = corpus::load_chunks_from_dir(
                        &config.corpus_dir,
                        r_config.chunk_size,
                        r_config.overlap,
                        &retriever_id,
                    )?;
                    let embedded = embed_chunks_parallel(
                        raw,
                        &config.ollama.url,
                        &config.ollama.embed_model,
                        config.embed_batch,
                    )
                    .await?;
                    corpus::save_cache(&config.cache_dir, &retriever_id, &embedded)?;
                    embedded
                };
                retriever.index_chunks(chunks).await?;
                retrievers.push(Box::new(retriever));
            }
            RetrieverType::Bm25 => {
                println!("Indexing {} ...", retriever_id);
                let mut retriever = Bm25Retriever::new(&retriever_id)?;
                let chunks = corpus::load_chunks_from_dir(
                    &config.corpus_dir,
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
        anyhow::bail!("No retrievers configured");
    }

    Ok(retrievers)
}
