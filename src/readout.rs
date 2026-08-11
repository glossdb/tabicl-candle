//! The read-out contract over density scores — the pieces the
//! evaluation harness computed downstream of `score_samples`, moved
//! into the crate as stage 3 closes: NLL, the q95 reference threshold,
//! the batch score, and AUROC. Semantics mirror the harness exactly
//! (numpy linear-interpolation quantile, sklearn's tie-averaged AUROC).

/// NLL per row: `-ln(score + tiny)` with numpy's `finfo(float).tiny`
/// (the smallest normal f64). Scores are ~1e-10-scale for wide tables;
/// the log space is where the information lives.
pub fn nll(scores: &[f64]) -> Vec<f64> {
    scores
        .iter()
        .map(|s| -(s + f64::MIN_POSITIVE).ln())
        .collect()
}

/// numpy's default linear-interpolation quantile.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    assert!(!values.is_empty() && (0.0..=1.0).contains(&q));
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (v.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}

/// Fraction of batch rows above the reference threshold.
pub fn batch_score(bat_nll: &[f64], threshold: f64) -> f64 {
    bat_nll.iter().filter(|v| **v > threshold).count() as f64 / bat_nll.len() as f64
}

/// AUROC with tie-averaged ranks (the Mann-Whitney statistic sklearn's
/// roc_auc_score computes). Labels are 0/1; higher score = positive.
pub fn auroc(labels: &[i64], scores: &[f64]) -> f64 {
    let n = labels.len();
    assert_eq!(n, scores.len());
    let n_pos = labels.iter().filter(|l| **l == 1).count();
    let n_neg = n - n_pos;
    assert!(n_pos > 0 && n_neg > 0, "AUROC needs both classes");

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

    // average ranks over tie groups (1-based)
    let mut ranks = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[order[j + 1]] == scores[order[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f64 / 2.0;
        for &idx in &order[i..=j] {
            ranks[idx] = avg;
        }
        i = j + 1;
    }

    let rank_sum: f64 = (0..n).filter(|&i| labels[i] == 1).map(|i| ranks[i]).sum();
    (rank_sum - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}
