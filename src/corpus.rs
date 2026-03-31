use crate::models::Chunk;
use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn load_chunks_from_dir(dir: &str, chunk_size: usize, retriever_id: &str) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut chunk_index = 0;

    for entry in fs::read_dir(Path::new(dir))? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }

        let text = fs::read_to_string(&path)?;
        let words: Vec<&str> = text.split_whitespace().collect();

        for window in words.chunks(chunk_size) {
            let chunk_text = window.join(" ");
            chunks.push(Chunk {
                id: format!("{}-{}", retriever_id, chunk_index),
                text: chunk_text,
                embedding: vec![],
            });
            chunk_index += 1;
        }
    }

    Ok(chunks)
}
