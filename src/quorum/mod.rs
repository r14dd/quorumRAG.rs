use crate::models::EvidenceCluster;

pub fn filter_by_quorum(
    clusters: Vec<EvidenceCluster>,
    threshold: usize,
) -> Vec<EvidenceCluster> {
    clusters.into_iter().filter(|c| c.support >= threshold).collect()
}
