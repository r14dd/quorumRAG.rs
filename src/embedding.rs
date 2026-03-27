use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

pub struct EmbeddingClient {
    client: Client,
    url: String,
    model: String
}

impl EmbeddingClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            url: format!("{}/api/embeddings", base_url),
            model: model.to_string(),
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
        .client
        .post(&self.url)
        .json(&EmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        })
        .send()
        .await?
        .json::<EmbedResponse>()
        .await?;

    Ok(response.embedding)
    }
}

