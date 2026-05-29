use anyhow::Result;
use quorumrag::Config;
use quorumrag::QuorumRag;
use quorumrag::evaluation::{self, EvalResult, load_eval_samples, print_results};

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    let rag = QuorumRag::build(config).await?;

    let args: Vec<String> = std::env::args().collect();
    let query_arg = args
        .windows(2)
        .find(|w| w[0] == "--query")
        .map(|w| w[1].clone());

    if let Some(query_text) = query_arg {
        let result = rag.retrieve(&query_text, true).await?;
        println!(
            "Context (max support: {}):\n{}\n",
            result.max_support, result.context
        );
        let answer = rag.generator().generate(&result.context, &query_text).await?;
        println!("Answer: {}", answer);
        return Ok(());
    }

    // Eval mode: compare a single-retriever baseline against full quorum.
    let samples = load_eval_samples("data/eval.json")?;

    let mut baseline_results = Vec::new();
    let mut quorum_results = Vec::new();

    for sample in &samples {
        let baseline = rag.retrieve(&sample.query, false).await?;
        let baseline_hit =
            evaluation::context_contains_relevant(&baseline.context, &sample.relevant);
        baseline_results.push(EvalResult {
            query: sample.query.clone(),
            hit: baseline_hit,
            support: 1,
        });

        let quorum = rag.retrieve(&sample.query, true).await?;
        let quorum_hit = evaluation::context_contains_relevant(&quorum.context, &sample.relevant);
        quorum_results.push(EvalResult {
            query: sample.query.clone(),
            hit: quorum_hit,
            support: quorum.max_support,
        });
    }

    print_results(&baseline_results, "Baseline (single retriever)");
    print_results(&quorum_results, "QuorumRAG");

    let answer = rag.answer(&samples[0].query).await?;
    println!("\nSample answer: {}", answer);

    Ok(())
}
