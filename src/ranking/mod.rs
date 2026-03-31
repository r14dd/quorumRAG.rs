use crate::models::EvidenceCluster;

pub fn rank_clusters(
    mut clusters: Vec<EvidenceCluster>,
    alpha: f32,
    beta: f32,
) -> Vec<EvidenceCluster> {
    clusters.sort_by(|a, b| {
        let score_a = alpha * a.avg_score + beta * a.support as f32;
        let score_b = alpha * b.avg_score + beta * b.support as f32;
        score_b.partial_cmp(&score_a).unwrap()
    });
    clusters
}
