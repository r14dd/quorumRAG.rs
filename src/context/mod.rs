use crate::models::EvidenceCluster;
use std::collections::HashSet;

pub fn build_context(clusters: &[EvidenceCluster], max_clusters: usize) -> String {
    clusters
        .iter()
        .take(max_clusters)
        .enumerate()
        .map(|(i, cluster)| {
            // Deduplicate member texts, preserving order by descending score
            let mut seen = HashSet::new();
            let mut members_sorted = cluster.members.clone();
            members_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            let texts: Vec<&str> = members_sorted
                .iter()
                .filter_map(|m| {
                    if seen.insert(m.chunk.id.as_str()) {
                        Some(m.chunk.text.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            format!(
                "[{}] (support: {}, score: {:.3})\n{}",
                i + 1,
                cluster.support,
                cluster.avg_score,
                texts.join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}
