use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct CachedChunks {
    pub chunks: Vec<crate::models::Chunk>,
}

pub fn load_chunks_from_dir(
    dir: &str,
    chunk_size: usize,
    overlap: usize,
    retriever_id: &str,
) -> anyhow::Result<Vec<crate::models::Chunk>> {
    let mut chunks = Vec::new();
    let dir_path = Path::new(dir);
    let stride = chunk_size.saturating_sub(overlap).max(1);

    let mut entries: Vec<_> = fs::read_dir(dir_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().and_then(|ext| ext.to_str()) == Some("txt")
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let filename = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let text = fs::read_to_string(&path)?;
        let words: Vec<&str> = text.split_whitespace().collect();
        let total = words.len();

        let mut i = 0;
        let mut chunk_idx = 0;
        while i < total {
            let end = (i + chunk_size).min(total);
            let chunk_text = words[i..end].join(" ");
            chunks.push(crate::models::Chunk {
                id: format!("{}-{}-{}", retriever_id, filename, chunk_idx),
                text: chunk_text,
                embedding: Vec::new(),
            });
            chunk_idx += 1;
            i += stride;
        }
    }

    Ok(chunks)
}

pub fn cache_path(cache_dir: &str, retriever_id: &str) -> String {
    format!("{}/{}.json", cache_dir, retriever_id.replace("/", "_"))
}

pub fn load_cache(cache_dir: &str, retriever_id: &str) -> Option<Vec<crate::models::Chunk>> {
    let path = cache_path(cache_dir, retriever_id);
    let content = std::fs::read_to_string(&path).ok()?;
    let cached: CachedChunks = serde_json::from_str(&content).ok()?;
    Some(cached.chunks)
}

pub fn save_cache(
    cache_dir: &str,
    retriever_id: &str,
    chunks: &[crate::models::Chunk],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(cache_dir)?;
    let cached = CachedChunks { chunks: chunks.to_vec() };
    let content = serde_json::to_string(&cached)?;
    std::fs::write(cache_path(cache_dir, retriever_id), content)?;
    Ok(())
}
