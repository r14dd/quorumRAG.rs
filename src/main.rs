mod config;
mod models;
mod retrievers;
mod clustering;
mod quorum;
mod ranking;
mod context;
mod generation;

use anyhow::{Ok, Result};
use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;
    println!("Loaded config: {:#?}", config);
    Ok(())
}
