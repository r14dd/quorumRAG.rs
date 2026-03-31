use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
pub struct EvalSample {
    pub query: String,
    pub relevant: String,
}

pub fn load_eval_samples(path: &str) -> Result<Vec<EvalSample>> {
    let content = fs::read_to_string(path)?;
    let samples: Vec<EvalSample> = serde_json::from_str(&content)?;
    Ok(samples)
}

pub fn context_contains_relevant(context: &str, relevant: &str) -> bool {
    let context_lower = context.to_lowercase();
    let relevant_lower = relevant.to_lowercase();
    let relevant_words: Vec<&str> = relevant_lower.split_whitespace().collect();
    let match_count = relevant_words
        .iter()
        .filter(|w| context_lower.contains(*w))
        .count();
    let ratio = match_count as f32 / relevant_words.len() as f32;
    ratio >= 0.7
}

pub struct EvalResult {
    pub query: String,
    pub hit: bool,
    pub support: usize,
}

pub fn print_results(results: &[EvalResult], label: &str) {
    let hits = results.iter().filter(|r| r.hit).count();
    println!("\n=== {} ===", label);
    println!("Recall: {}/{}", hits, results.len());
    for r in results {
        println!(
            "[{}] (support: {}) {}",
            if r.hit { "HIT " } else { "MISS" },
            r.support,
            r.query
        );
    }
}
