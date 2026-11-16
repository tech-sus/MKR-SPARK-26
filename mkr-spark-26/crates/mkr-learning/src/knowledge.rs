use mkr_core::{SourceRef, time::now_unix};
use mkr_knowledge::{Knowledge, NewFact};
use mkr_verification::{Document, Status, Verdict};

#[derive(Clone, Debug)]
pub struct LearnPolicy {
    pub ttl_days: u64,
    /// minimum verdict confidence for automatic storage of verified claims
    pub min_confidence: f32,
    /// user explicitly approved storing a single-source (unconfirmed) claim
    pub approve_single_source: bool,
}

impl Default for LearnPolicy {
    fn default() -> Self {
        LearnPolicy { ttl_days: 180, min_confidence: 0.6, approve_single_source: false }
    }
}

#[derive(Debug, PartialEq)]
pub enum LearnOutcome {
    Stored { fact_id: i64, created: bool },
    /// not stored; the reason is shown to the user
    Skipped(String),
}

/// Store the best claim of a verdict as knowledge, with per-source excerpts. Conflicting or
/// insufficient evidence is never stored; single-source claims only with explicit user approval
/// (and at reduced confidence).
pub fn learn_from_verdict(kb: &Knowledge, topic: &str, verdict: &Verdict, docs: &[Document], fetched_at: i64, policy: &LearnPolicy) -> mkr_core::Result<LearnOutcome> {
    let Some(best) = &verdict.best else { return Ok(LearnOutcome::Skipped("no claim to store".into())) };
    match verdict.status {
        Status::Insufficient => return Ok(LearnOutcome::Skipped("insufficient evidence".into())),
        Status::Conflicting => return Ok(LearnOutcome::Skipped("sources conflict; not storing disputed information".into())),
        Status::SingleSource if !policy.approve_single_source => return Ok(LearnOutcome::Skipped("only one independent source; not stored without your approval".into())),
        Status::Verified if verdict.confidence < policy.min_confidence => return Ok(LearnOutcome::Skipped(format!("confidence {:.2} below threshold {:.2}", verdict.confidence, policy.min_confidence))),
        _ => {}
    }
    let sources: Vec<(SourceRef, String)> = best
        .per_doc
        .iter()
        .map(|(d, sentence)| {
            let doc = &docs[*d];
            (SourceRef { url: doc.url.clone(), title: doc.title.clone(), domain: doc.domain.clone(), published: doc.published, fetched_at }, sentence.clone())
        })
        .collect();
    let confidence = if verdict.status == Status::SingleSource { verdict.confidence.min(0.45) } else { verdict.confidence };
    let out = kb.add_fact(&NewFact { claim: best.sentence.clone(), topic: topic.to_string(), confidence, ttl_days: Some(policy.ttl_days), user_verified: false, sources })?;
    let _ = now_unix();
    Ok(LearnOutcome::Stored { fact_id: out.id, created: out.created })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mkr_verification::{verify, VerifyConfig};

    fn docs() -> Vec<Document> {
        vec![
            Document { url: "https://en.wikipedia.org/wiki/Paris".into(), title: "Paris".into(), domain: "wikipedia.org".into(), published: Some(1_780_000_000), text: "Paris is the capital and most populous city of France.".into() },
            Document { url: "https://www.britannica.com/place/Paris".into(), title: "Paris | Britannica".into(), domain: "britannica.com".into(), published: Some(1_781_000_000), text: "Paris, the capital of France, is the country's largest city.".into() },
        ]
    }

    #[test]
    fn verified_claims_are_stored_with_per_source_excerpts() {
        let kb = Knowledge::open_in_memory().unwrap();
        let d = docs();
        let v = verify("what is the capital of France", &d, &VerifyConfig::new(1_790_000_000));
        assert_eq!(v.status, Status::Verified);
        let o = learn_from_verdict(&kb, "france", &v, &d, 1_790_000_000, &LearnPolicy::default()).unwrap();
        let LearnOutcome::Stored { fact_id, created } = o else { panic!("{o:?}") };
        assert!(created);
        let f = kb.get(fact_id).unwrap().unwrap();
        assert_eq!(f.sources.len(), 2);
        assert!(f.sources.iter().all(|s| s.excerpt.to_lowercase().contains("capital")));
        // learning again merges instead of duplicating
        let again = learn_from_verdict(&kb, "france", &v, &d, 1_790_000_100, &LearnPolicy::default()).unwrap();
        assert!(matches!(again, LearnOutcome::Stored { created: false, .. }));
        assert_eq!(kb.stats().unwrap().0, 1);
    }

    #[test]
    fn unverified_evidence_is_not_learned_without_approval() {
        let kb = Knowledge::open_in_memory().unwrap();
        let one = vec![docs().remove(0)];
        let v = verify("what is the capital of France", &one, &VerifyConfig::new(1_790_000_000));
        assert_eq!(v.status, Status::SingleSource);
        assert!(matches!(learn_from_verdict(&kb, "t", &v, &one, 0, &LearnPolicy::default()).unwrap(), LearnOutcome::Skipped(_)));
        assert_eq!(kb.stats().unwrap().0, 0);
        let approved = LearnPolicy { approve_single_source: true, ..Default::default() };
        let LearnOutcome::Stored { fact_id, .. } = learn_from_verdict(&kb, "t", &v, &one, 0, &approved).unwrap() else { panic!() };
        assert!(kb.get(fact_id).unwrap().unwrap().confidence <= 0.45, "single-source facts keep low confidence");
        // conflicting evidence is never stored, even with approval
        let conflict = vec![
            Document { url: "https://x.org/a".into(), title: "a".into(), domain: "x.org".into(), published: None, text: "The tower has a height of 300 metres in total.".into() },
            Document { url: "https://y.org/b".into(), title: "b".into(), domain: "y.org".into(), published: None, text: "The tower has a height of 330 metres in total.".into() },
        ];
        let vc = verify("what is the height of the tower", &conflict, &VerifyConfig::new(1_790_000_000));
        assert_eq!(vc.status, Status::Conflicting);
        assert!(matches!(learn_from_verdict(&kb, "t", &vc, &conflict, 0, &approved).unwrap(), LearnOutcome::Skipped(_)));
    }
}
