use crate::models::{Candidate, EvidenceCluster};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn centroid(candidates: &[Candidate]) -> Vec<f32> {
    let dim = candidates[0].chunk.embedding.len();
    let mut center = vec![0.0f32; dim];
    for c in candidates {
        for (i, v) in c.chunk.embedding.iter().enumerate() {
            center[i] += v;
        }
    }
    let n = candidates.len() as f32;
    center.iter_mut().for_each(|v| *v /= n);
    center
}

pub fn cluster_candidates(
    candidates: Vec<Candidate>,
    threshold: f32,
) -> Vec<EvidenceCluster> {
    let mut clusters: Vec<EvidenceCluster> = Vec::new();

    for candidate in candidates {
        let mut assigned = false;

        for cluster in &mut clusters {
            let center = centroid(&cluster.members);
            let sim = cosine_similarity(&candidate.chunk.embedding, &center);
            if sim >= threshold {
                cluster.members.push(candidate.clone());
                assigned = true;
                break;
            }
        }

        if !assigned {
            clusters.push(EvidenceCluster {
                members: vec![candidate],
                support: 0,
                avg_score: 0.0,
            });
        }
    }

    for cluster in &mut clusters {
        let unique_retrievers: std::collections::HashSet<&str> = cluster
            .members
            .iter()
            .map(|c| c.retriever_id.as_str())
            .collect();
        cluster.support = unique_retrievers.len();
        cluster.avg_score = cluster.members.iter().map(|c| c.score).sum::<f32>()
            / cluster.members.len() as f32;
    }

    clusters
}
