use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

pub struct Generator {
    client: Client,
    url: String,
    model: String,
}

impl Generator {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            url: format!("{}/api/generate", base_url),
            model: model.to_string(),
        }
    }

    pub async fn generate(&self, context: &str, question: &str) -> Result<String> {
        let prompt = format!(
            "Use the following evidence to answer the question.\n\nEvidence:\n{}\n\nQuestion: {}\n\nAnswer:",
            context, question
        );

        let response = self
            .client
            .post(&self.url)
            .json(&GenerateRequest {
                model: self.model.clone(),
                prompt,
                stream: false,
            })
            .send()
            .await?
            .json::<GenerateResponse>()
            .await?;

        Ok(response.response)
    }
}
