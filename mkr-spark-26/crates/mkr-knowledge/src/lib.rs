//! Verified knowledge records. This database is separate from the neural weights: facts here are
//! retrieved evidence with sources and an expiry, never "learned into" the network.
use mkr_core::{db, text, time::now_unix, MkrError, Result, SourceRef};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashSet;
use std::path::Path;

pub struct Knowledge {
    conn: Connection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactStatus {
    Active = 0,
    Disputed = 1,
    Expired = 2,
    Superseded = 3,
}
impl FactStatus {
    fn from_i(i: i64) -> FactStatus {
        match i {
            1 => FactStatus::Disputed,
            2 => FactStatus::Expired,
            3 => FactStatus::Superseded,
            _ => FactStatus::Active,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            FactStatus::Active => "active",
            FactStatus::Disputed => "disputed",
            FactStatus::Expired => "expired",
            FactStatus::Superseded => "superseded",
        }
    }
}

#[derive(Clone, Debug)]
pub struct StoredSource {
    pub url: String,
    pub title: String,
    pub domain: String,
    pub published: Option<i64>,
    pub fetched_at: i64,
    pub excerpt: String,
}

#[derive(Clone, Debug)]
pub struct Fact {
    pub id: i64,
    pub claim: String,
    pub topic: String,
    pub confidence: f32,
    pub created: i64,
    pub updated: i64,
    pub expires_at: Option<i64>,
    pub status: FactStatus,
    pub user_verified: bool,
    pub sources: Vec<StoredSource>,
}

#[derive(Clone, Debug)]
pub struct NewFact {
    pub claim: String,
    pub topic: String,
    pub confidence: f32,
    pub ttl_days: Option<u64>,
    pub user_verified: bool,
    pub sources: Vec<(SourceRef, String)>,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub fact: Fact,
    /// fraction of the query's content words found in the claim (0..1)
    pub coverage: f32,
}

#[derive(Debug, PartialEq)]
pub struct AddOutcome {
    pub id: i64,
    pub created: bool,
    pub new_sources: usize,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, domain TEXT NOT NULL, published INTEGER, fetched_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, claim TEXT NOT NULL, hash INTEGER NOT NULL UNIQUE, topic TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL, created INTEGER NOT NULL, updated INTEGER NOT NULL, expires_at INTEGER, status INTEGER NOT NULL DEFAULT 0, user_verified INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS fact_sources(fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE, source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE, excerpt TEXT NOT NULL DEFAULT '', PRIMARY KEY(fact_id, source_id));
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(norm, tokenize='unicode61');
PRAGMA user_version=1;
";

fn claim_hash(claim: &str) -> i64 {
    text::stable_hash(&text::tokens(claim, false).join(" ")) as i64
}

impl Knowledge {
    pub fn open(path: &Path) -> Result<Knowledge> {
        let conn = db::open_db(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Knowledge { conn })
    }
    pub fn open_in_memory() -> Result<Knowledge> {
        let conn = db::open_memory_db()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Knowledge { conn })
    }

    /// Insert a fact, or merge into an existing identical claim (adds sources, raises confidence
    /// by noisy-OR, refreshes expiry). A user-verified fact keeps confidence 1.0.
    pub fn add_fact(&self, nf: &NewFact) -> Result<AddOutcome> {
        let claim = nf.claim.trim();
        if claim.chars().count() < 10 || claim.chars().count() > 1500 {
            return Err(MkrError::Invalid("claim must be 10..1500 characters".into()));
        }
        if !(0.0..=1.0).contains(&nf.confidence) {
            return Err(MkrError::Invalid("confidence must be within 0..1".into()));
        }
        let now = now_unix();
        let expires = nf.ttl_days.map(|d| now + d as i64 * 86400);
        let h = claim_hash(claim);
        let existing: Option<(i64, f64, i64)> = self.conn.query_row("SELECT id,confidence,user_verified FROM facts WHERE hash=?", [h], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).optional()?;
        let (id, created) = match existing {
            Some((id, old_conf, uv)) => {
                let merged = if uv != 0 || nf.user_verified { 1.0 } else { (1.0 - (1.0 - old_conf) * (1.0 - nf.confidence as f64)).min(0.99) };
                self.conn.execute("UPDATE facts SET confidence=?, updated=?, expires_at=?, status=0, user_verified=MAX(user_verified,?) WHERE id=?", params![merged, now, expires, nf.user_verified as i64, id])?;
                (id, false)
            }
            None => {
                self.conn.execute(
                    "INSERT INTO facts(claim,hash,topic,confidence,created,updated,expires_at,status,user_verified) VALUES(?,?,?,?,?,?,?,0,?)",
                    params![claim, h, nf.topic, if nf.user_verified { 1.0 } else { nf.confidence as f64 }, now, now, expires, nf.user_verified as i64],
                )?;
                let id = self.conn.last_insert_rowid();
                self.conn.execute("INSERT INTO facts_fts(rowid,norm) VALUES(?,?)", params![id, text::normalize(&format!("{claim} {}", nf.topic))])?;
                (id, true)
            }
        };
        let mut new_sources = 0;
        for (s, excerpt) in &nf.sources {
            self.conn.execute(
                "INSERT INTO sources(url,title,domain,published,fetched_at) VALUES(?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET title=excluded.title, published=COALESCE(excluded.published, published), fetched_at=excluded.fetched_at",
                params![s.url, s.title, s.domain, s.published, s.fetched_at],
            )?;
            let sid: i64 = self.conn.query_row("SELECT id FROM sources WHERE url=?", [&s.url], |r| r.get(0))?;
            let n = self.conn.execute("INSERT OR IGNORE INTO fact_sources(fact_id,source_id,excerpt) VALUES(?,?,?)", params![id, sid, excerpt])?;
            new_sources += n;
        }
        Ok(AddOutcome { id, created, new_sources })
    }

    fn load_fact(&self, id: i64) -> Result<Option<Fact>> {
        let row = self
            .conn
            .query_row("SELECT id,claim,topic,confidence,created,updated,expires_at,status,user_verified FROM facts WHERE id=?", [id], |r| {
                Ok(Fact { id: r.get(0)?, claim: r.get(1)?, topic: r.get(2)?, confidence: r.get::<_, f64>(3)? as f32, created: r.get(4)?, updated: r.get(5)?, expires_at: r.get(6)?, status: FactStatus::from_i(r.get(7)?), user_verified: r.get::<_, i64>(8)? != 0, sources: vec![] })
            })
            .optional()?;
        let Some(mut f) = row else { return Ok(None) };
        let mut st = self.conn.prepare("SELECT s.url,s.title,s.domain,s.published,s.fetched_at,fs.excerpt FROM fact_sources fs JOIN sources s ON s.id=fs.source_id WHERE fs.fact_id=? ORDER BY s.id")?;
        f.sources = st
            .query_map([id], |r| Ok(StoredSource { url: r.get(0)?, title: r.get(1)?, domain: r.get(2)?, published: r.get(3)?, fetched_at: r.get(4)?, excerpt: r.get(5)? }))?
            .collect::<std::result::Result<_, _>>()?;
        Ok(Some(f))
    }
    pub fn get(&self, id: i64) -> Result<Option<Fact>> {
        self.load_fact(id)
    }

    /// Search stored knowledge. Expired and superseded facts are excluded unless `include_inactive`.
    pub fn query(&self, q: &str, limit: usize, include_inactive: bool) -> Result<Vec<Hit>> {
        let Some(fq) = mkr_memory::fts_query(q) else { return Ok(vec![]) };
        let qtoks: HashSet<String> = text::tokens(q, true).into_iter().collect();
        if qtoks.is_empty() {
            return Ok(vec![]);
        }
        let mut st = self.conn.prepare("SELECT rowid FROM facts_fts WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT 40")?;
        let ids: Vec<i64> = st.query_map([fq], |r| r.get(0))?.collect::<std::result::Result<_, _>>()?;
        let now = now_unix();
        let mut hits = Vec::new();
        for id in ids {
            let Some(f) = self.load_fact(id)? else { continue };
            let expired = f.expires_at.map_or(false, |e| e < now);
            if !include_inactive && (expired || matches!(f.status, FactStatus::Expired | FactStatus::Superseded)) {
                continue;
            }
            let ftoks: HashSet<String> = text::tokens(&format!("{} {}", f.claim, f.topic), true).into_iter().collect();
            let found = qtoks.iter().filter(|t| ftoks.contains(*t) || ftoks.iter().any(|x| x.starts_with(t.as_str()) && t.len() >= 4)).count();
            hits.push(Hit { coverage: found as f32 / qtoks.len() as f32, fact: f });
        }
        hits.sort_by(|a, b| {
            let sa = a.coverage * 0.7 + a.fact.confidence * 0.3;
            let sb = b.coverage * 0.7 + b.fact.confidence * 0.3;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn list(&self, limit: usize, offset: usize) -> Result<Vec<Fact>> {
        let mut st = self.conn.prepare("SELECT id FROM facts ORDER BY id DESC LIMIT ? OFFSET ?")?;
        let ids: Vec<i64> = st.query_map([limit as i64, offset as i64], |r| r.get(0))?.collect::<std::result::Result<_, _>>()?;
        let mut v = Vec::new();
        for id in ids {
            if let Some(f) = self.load_fact(id)? {
                v.push(f);
            }
        }
        Ok(v)
    }

    /// User correction: rewrites the claim, marks it user-verified (confidence 1.0).
    pub fn correct(&self, id: i64, new_claim: &str) -> Result<bool> {
        let new_claim = new_claim.trim();
        if new_claim.chars().count() < 10 {
            return Err(MkrError::Invalid("claim too short".into()));
        }
        let Some(f) = self.load_fact(id)? else { return Ok(false) };
        self.conn
            .execute("UPDATE facts SET claim=?, hash=?, confidence=1.0, user_verified=1, status=0, updated=? WHERE id=?", params![new_claim, claim_hash(new_claim), now_unix(), id])
            .map_err(|e| MkrError::Db(format!("{e} (an identical fact may already exist)")))?;
        self.conn.execute("UPDATE facts_fts SET norm=? WHERE rowid=?", params![text::normalize(&format!("{new_claim} {}", f.topic)), id])?;
        Ok(true)
    }

    pub fn delete(&self, id: i64) -> Result<bool> {
        self.conn.execute("DELETE FROM facts_fts WHERE rowid=?", [id])?;
        let n = self.conn.execute("DELETE FROM facts WHERE id=?", [id])?;
        self.conn.execute("DELETE FROM sources WHERE id NOT IN (SELECT source_id FROM fact_sources)", [])?;
        Ok(n > 0)
    }

    pub fn set_status(&self, id: i64, s: FactStatus) -> Result<bool> {
        Ok(self.conn.execute("UPDATE facts SET status=?, updated=? WHERE id=?", params![s as i64, now_unix(), id])? > 0)
    }

    /// Mark `old` as superseded by a newer fact.
    pub fn supersede(&self, old: i64, _new: i64) -> Result<bool> {
        self.set_status(old, FactStatus::Superseded)
    }

    /// Flag facts past their expiry. Returns how many were newly expired. User-verified facts never auto-expire.
    pub fn expire_due(&self, now: i64) -> Result<usize> {
        Ok(self.conn.execute("UPDATE facts SET status=2 WHERE status=0 AND user_verified=0 AND expires_at IS NOT NULL AND expires_at < ?", [now])?)
    }

    /// Permanently remove expired facts older than `days` past their expiry.
    pub fn prune_expired(&self, now: i64, days: u64) -> Result<usize> {
        let ids: Vec<i64> = {
            let mut st = self.conn.prepare("SELECT id FROM facts WHERE status=2 AND expires_at < ?")?;
            let v = st.query_map([now - days as i64 * 86400], |r| r.get(0))?.collect::<std::result::Result<_, _>>()?;
            v
        };
        for id in &ids {
            self.delete(*id)?;
        }
        Ok(ids.len())
    }

    pub fn stats(&self) -> Result<(i64, i64, i64)> {
        let facts = self.conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0))?;
        let active = self.conn.query_row("SELECT COUNT(*) FROM facts WHERE status=0", [], |r| r.get(0))?;
        let sources = self.conn.query_row("SELECT COUNT(*) FROM sources", [], |r| r.get(0))?;
        Ok((facts, active, sources))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(url: &str, domain: &str) -> SourceRef {
        SourceRef { url: url.into(), title: format!("Title of {domain}"), domain: domain.into(), published: Some(1_700_000_000), fetched_at: now_unix() }
    }
    fn nf(claim: &str, conf: f32, sources: Vec<(SourceRef, String)>) -> NewFact {
        NewFact { claim: claim.into(), topic: "test".into(), confidence: conf, ttl_days: Some(30), user_verified: false, sources }
    }

    #[test]
    fn add_query_merge_sources_and_confidence() {
        let k = Knowledge::open_in_memory().unwrap();
        let a = k.add_fact(&nf("Rust was first released as a stable 1.0 version in May 2015.", 0.6, vec![(src("https://a.org/x", "a.org"), "ex1".into())])).unwrap();
        assert!(a.created && a.new_sources == 1);
        let b = k.add_fact(&nf("rust was first released as a stable 1.0 version in may 2015", 0.6, vec![(src("https://b.org/y", "b.org"), "ex2".into()), (src("https://a.org/x", "a.org"), "ex1".into())])).unwrap();
        assert_eq!(b.id, a.id);
        assert!(!b.created);
        assert_eq!(b.new_sources, 1, "same url must not duplicate");
        let f = k.get(a.id).unwrap().unwrap();
        assert_eq!(f.sources.len(), 2);
        assert!((f.confidence - 0.84).abs() < 0.01, "noisy-or: {}", f.confidence);
        let hits = k.query("when was Rust 1.0 released", 5, false).unwrap();
        assert_eq!(hits[0].fact.id, a.id);
        assert!(hits[0].coverage >= 0.6);
        assert!(k.query("quantum chromodynamics", 5, false).unwrap().is_empty());
    }

    #[test]
    fn arabic_knowledge() {
        let k = Knowledge::open_in_memory().unwrap();
        let o = k.add_fact(&nf("القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها.", 0.9, vec![(src("https://ar.example.org/cairo", "example.org"), "".into())])).unwrap();
        let hits = k.query("ما هي عاصمة مصر", 3, false).unwrap();
        assert_eq!(hits[0].fact.id, o.id);
        assert!(hits[0].coverage > 0.9);
    }

    #[test]
    fn expiry_user_verified_and_prune() {
        let k = Knowledge::open_in_memory().unwrap();
        let a = k.add_fact(&nf("The current record for the test benchmark stands at 42 points.", 0.7, vec![])).unwrap();
        let mut verified = nf("The user confirmed this exact statement about the project.", 0.7, vec![]);
        verified.user_verified = true;
        let b = k.add_fact(&verified).unwrap();
        let future = now_unix() + 40 * 86400;
        assert_eq!(k.expire_due(future).unwrap(), 1, "user-verified facts must not auto-expire");
        assert!(k.query("record benchmark points", 5, false).unwrap().is_empty());
        assert_eq!(k.query("record benchmark points", 5, true).unwrap().len(), 1);
        assert_eq!(k.get(a.id).unwrap().unwrap().status, FactStatus::Expired);
        assert_eq!(k.get(b.id).unwrap().unwrap().status, FactStatus::Active);
        assert_eq!(k.prune_expired(future + 100 * 86400, 30).unwrap(), 1);
        assert!(k.get(a.id).unwrap().is_none());
    }

    #[test]
    fn user_can_correct_and_delete() {
        let k = Knowledge::open_in_memory().unwrap();
        let o = k.add_fact(&nf("The library was written by Alice in the year 2010.", 0.5, vec![(src("https://x.org/1", "x.org"), "".into())])).unwrap();
        assert!(k.correct(o.id, "The library was written by Bob in the year 2011.").unwrap());
        let f = k.get(o.id).unwrap().unwrap();
        assert!(f.user_verified && f.confidence == 1.0);
        assert!(k.query("alice", 3, false).unwrap().is_empty());
        assert_eq!(k.query("bob library", 3, false).unwrap().len(), 1);
        assert!(k.delete(o.id).unwrap());
        assert!(k.get(o.id).unwrap().is_none());
        assert_eq!(k.stats().unwrap(), (0, 0, 0), "orphan sources must be cleaned up");
        assert!(k.add_fact(&nf("short", 0.5, vec![])).is_err());
        assert!(k.add_fact(&nf("A valid length claim here.", 1.5, vec![])).is_err());
    }

    #[test]
    fn supersede_hides_old_fact() {
        let k = Knowledge::open_in_memory().unwrap();
        let old = k.add_fact(&nf("The stable version of the tool is 1.2 as of this year.", 0.8, vec![])).unwrap();
        let new = k.add_fact(&nf("The stable version of the tool is 1.3 as of this year.", 0.8, vec![])).unwrap();
        k.supersede(old.id, new.id).unwrap();
        let hits = k.query("stable version of the tool", 5, false).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].fact.id, new.id);
    }
}
