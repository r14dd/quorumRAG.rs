use crate::models::Chunk;
use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn load_chunks_from_dir(dir: &str, chunk_size: usize, retriever_id: &str) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut chunk_index = 0;
    let overlap = chunk_size / 2;

    for entry in fs::read_dir(Path::new(dir))? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }

        let text = fs::read_to_string(&path)?;
        let words: Vec<&str> = text.split_whitespace().collect();

        let mut start = 0;
        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            let chunk_text = words[start..end].join(" ");

            chunks.push(Chunk {
                id: format!("{}-{}", retriever_id, chunk_index),
                text: chunk_text,
                embedding: vec![],
            });

            chunk_index += 1;

            if end == words.len() {
                break;
            }

            start += chunk_size - overlap;
        }
    }

    Ok(chunks)
}
