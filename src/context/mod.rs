use crate::models::EvidenceCluster;

pub fn build_context(clusters: &[EvidenceCluster], max_clusters: usize) -> String {
    clusters
        .iter()
        .take(max_clusters)
        .enumerate()
        .map(|(i, cluster)| {
            let representative = &cluster.members[0].chunk.text;
            format!(
                "[{}] (support: {}, score: {:.3})\n{}",
                i + 1,
                cluster.support,
                cluster.avg_score,
                representative
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
