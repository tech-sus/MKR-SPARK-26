//! User-facing memory. Everything here is local and works fully offline.
//! Only information the user explicitly approves is stored as a memory item; conversation
//! messages are kept as a rolling log for context and can be cleared at any time.
use mkr_core::{db, text, time::now_unix, MkrError, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

pub struct Memory {
    conn: Connection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub id: i64,
    pub session: String,
    pub role: String,
    pub content: String,
    pub ts: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryItem {
    pub id: i64,
    pub content: String,
    pub tags: String,
    pub created: i64,
    pub updated: i64,
}

#[derive(Clone, Debug)]
pub struct SearchLog {
    pub ts: i64,
    pub query: String,
    pub provider: String,
    pub results: i64,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, ts INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id);
CREATE TABLE IF NOT EXISTS preferences(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', hash INTEGER NOT NULL UNIQUE, created INTEGER NOT NULL, updated INTEGER NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(norm, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS search_log(id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, query TEXT NOT NULL, provider TEXT NOT NULL, results INTEGER NOT NULL);
PRAGMA user_version=1;
";

/// Turn free text into a safe FTS5 query: quoted prefix terms joined with OR.
pub fn fts_query(q: &str) -> Option<String> {
    let toks = text::tokens(q, true);
    let toks: Vec<String> = toks.into_iter().filter(|t| !t.is_empty()).collect();
    if toks.is_empty() {
        return None;
    }
    Some(toks.iter().map(|t| format!("\"{}\"*", t.replace('"', ""))).collect::<Vec<_>>().join(" OR "))
}

fn norm_hash(content: &str) -> i64 {
    text::stable_hash(&text::tokens(content, false).join(" ")) as i64
}

impl Memory {
    pub fn open(path: &Path) -> Result<Memory> {
        let conn = db::open_db(path)?;
        conn.execute_batch(SCHEMA)?;
        Ok(Memory { conn })
    }
    pub fn open_in_memory() -> Result<Memory> {
        let conn = db::open_memory_db()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Memory { conn })
    }

    // ---- conversation memory ----
    pub fn add_message(&self, session: &str, role: &str, content: &str) -> Result<i64> {
        if content.trim().is_empty() {
            return Err(MkrError::Invalid("empty message".into()));
        }
        self.conn.execute("INSERT INTO messages(session,role,content,ts) VALUES(?,?,?,?)", params![session, role, content, now_unix()])?;
        Ok(self.conn.last_insert_rowid())
    }
    pub fn recent_messages(&self, session: &str, n: usize) -> Result<Vec<Message>> {
        let mut st = self.conn.prepare("SELECT id,session,role,content,ts FROM messages WHERE session=? ORDER BY id DESC LIMIT ?")?;
        let mut v: Vec<Message> = st
            .query_map(params![session, n as i64], |r| Ok(Message { id: r.get(0)?, session: r.get(1)?, role: r.get(2)?, content: r.get(3)?, ts: r.get(4)? }))?
            .collect::<std::result::Result<_, _>>()?;
        v.reverse();
        Ok(v)
    }
    pub fn clear_conversations(&self) -> Result<usize> {
        Ok(self.conn.execute("DELETE FROM messages", [])?)
    }
    pub fn message_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?)
    }

    // ---- preferences ----
    pub fn set_pref(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute("INSERT INTO preferences(key,value,updated) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated", params![key, value, now_unix()])?;
        Ok(())
    }
    pub fn get_pref(&self, key: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT value FROM preferences WHERE key=?", [key], |r| r.get(0)).optional()?)
    }
    pub fn list_prefs(&self) -> Result<Vec<(String, String)>> {
        let mut st = self.conn.prepare("SELECT key,value FROM preferences ORDER BY key")?;
        let v = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<std::result::Result<_, _>>()?;
        Ok(v)
    }
    pub fn delete_pref(&self, key: &str) -> Result<bool> {
        Ok(self.conn.execute("DELETE FROM preferences WHERE key=?", [key])? > 0)
    }

    // ---- user-approved memory items ----
    /// Store something the user approved remembering. Duplicates (same normalised words) are merged.
    /// Returns (id, newly_created).
    pub fn remember(&self, content: &str, tags: &str) -> Result<(i64, bool)> {
        let content = content.trim();
        if content.is_empty() {
            return Err(MkrError::Invalid("nothing to remember".into()));
        }
        if content.chars().count() > 2000 {
            return Err(MkrError::Invalid("memory item too long (max 2000 characters)".into()));
        }
        let h = norm_hash(content);
        if let Some(id) = self.conn.query_row("SELECT id FROM items WHERE hash=?", [h], |r| r.get::<_, i64>(0)).optional()? {
            self.conn.execute("UPDATE items SET updated=? WHERE id=?", params![now_unix(), id])?;
            return Ok((id, false));
        }
        let now = now_unix();
        self.conn.execute("INSERT INTO items(content,tags,hash,created,updated) VALUES(?,?,?,?,?)", params![content, tags, h, now, now])?;
        let id = self.conn.last_insert_rowid();
        self.conn.execute("INSERT INTO items_fts(rowid,norm) VALUES(?,?)", params![id, text::normalize(&format!("{content} {tags}"))])?;
        Ok((id, true))
    }

    /// Relevance-ranked recall (BM25 over normalised text, Arabic and English).
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>> {
        let Some(q) = fts_query(query) else { return Ok(vec![]) };
        let mut st = self.conn.prepare("SELECT i.id,i.content,i.tags,i.created,i.updated FROM items_fts f JOIN items i ON i.id=f.rowid WHERE items_fts MATCH ? ORDER BY bm25(items_fts) LIMIT ?")?;
        let v = st
            .query_map(params![q, limit as i64], |r| Ok(MemoryItem { id: r.get(0)?, content: r.get(1)?, tags: r.get(2)?, created: r.get(3)?, updated: r.get(4)? }))?
            .collect::<std::result::Result<_, _>>()?;
        Ok(v)
    }
    pub fn list_items(&self) -> Result<Vec<MemoryItem>> {
        let mut st = self.conn.prepare("SELECT id,content,tags,created,updated FROM items ORDER BY id")?;
        let v = st.query_map([], |r| Ok(MemoryItem { id: r.get(0)?, content: r.get(1)?, tags: r.get(2)?, created: r.get(3)?, updated: r.get(4)? }))?.collect::<std::result::Result<_, _>>()?;
        Ok(v)
    }
    pub fn update_item(&self, id: i64, content: &str) -> Result<bool> {
        let content = content.trim();
        if content.is_empty() {
            return Err(MkrError::Invalid("empty content".into()));
        }
        let tags: Option<String> = self.conn.query_row("SELECT tags FROM items WHERE id=?", [id], |r| r.get(0)).optional()?;
        let Some(tags) = tags else { return Ok(false) };
        self.conn.execute("UPDATE items SET content=?, hash=?, updated=? WHERE id=?", params![content, norm_hash(content), now_unix(), id]).map_err(|e| MkrError::Db(format!("{e} (an identical memory may already exist)")))?;
        self.conn.execute("UPDATE items_fts SET norm=? WHERE rowid=?", params![text::normalize(&format!("{content} {tags}")), id])?;
        Ok(true)
    }
    pub fn delete_item(&self, id: i64) -> Result<bool> {
        self.conn.execute("DELETE FROM items_fts WHERE rowid=?", [id])?;
        Ok(self.conn.execute("DELETE FROM items WHERE id=?", [id])? > 0)
    }
    pub fn clear_items(&self) -> Result<usize> {
        self.conn.execute("DELETE FROM items_fts", [])?;
        Ok(self.conn.execute("DELETE FROM items", [])?)
    }

    // ---- search history ----
    pub fn log_search(&self, query: &str, provider: &str, results: usize) -> Result<()> {
        self.conn.execute("INSERT INTO search_log(ts,query,provider,results) VALUES(?,?,?,?)", params![now_unix(), query, provider, results as i64])?;
        Ok(())
    }
    pub fn search_history(&self, limit: usize) -> Result<Vec<SearchLog>> {
        let mut st = self.conn.prepare("SELECT ts,query,provider,results FROM search_log ORDER BY id DESC LIMIT ?")?;
        let v = st.query_map([limit as i64], |r| Ok(SearchLog { ts: r.get(0)?, query: r.get(1)?, provider: r.get(2)?, results: r.get(3)? }))?.collect::<std::result::Result<_, _>>()?;
        Ok(v)
    }
    pub fn clear_search_history(&self) -> Result<usize> {
        Ok(self.conn.execute("DELETE FROM search_log", [])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_log_in_order() {
        let m = Memory::open_in_memory().unwrap();
        for i in 0..5 {
            m.add_message("s1", if i % 2 == 0 { "user" } else { "assistant" }, &format!("msg {i}")).unwrap();
        }
        m.add_message("s2", "user", "other session").unwrap();
        let r = m.recent_messages("s1", 3).unwrap();
        assert_eq!(r.iter().map(|x| x.content.as_str()).collect::<Vec<_>>(), ["msg 2", "msg 3", "msg 4"]);
        assert!(m.add_message("s1", "user", "   ").is_err());
        assert_eq!(m.clear_conversations().unwrap(), 6);
    }

    #[test]
    fn preferences_upsert() {
        let m = Memory::open_in_memory().unwrap();
        m.set_pref("lang", "ar").unwrap();
        m.set_pref("lang", "en").unwrap();
        assert_eq!(m.get_pref("lang").unwrap().as_deref(), Some("en"));
        assert_eq!(m.list_prefs().unwrap().len(), 1);
        assert!(m.delete_pref("lang").unwrap());
        assert_eq!(m.get_pref("lang").unwrap(), None);
    }

    #[test]
    fn remember_recall_dedupe_update_delete_english_and_arabic() {
        let m = Memory::open_in_memory().unwrap();
        let (a, new_a) = m.remember("My favorite programming language is Rust", "pref").unwrap();
        let (b, _) = m.remember("اسمي زياد وأعمل مطور برمجيات", "profile").unwrap();
        m.remember("The deploy server runs Ubuntu 24.04", "").unwrap();
        assert!(new_a);
        // duplicate (different case/punctuation) merges
        let (a2, new2) = m.remember("my favorite programming language is rust!", "").unwrap();
        assert_eq!((a2, new2), (a, false));
        assert_eq!(m.list_items().unwrap().len(), 3);

        let r = m.recall("which programming language do I like", 5).unwrap();
        assert_eq!(r[0].id, a);
        let r = m.recall("ما اسمي", 5).unwrap();
        assert_eq!(r[0].id, b, "Arabic recall failed: {r:?}");
        let r = m.recall("برمجيات", 5).unwrap();
        assert_eq!(r[0].id, b);
        assert!(m.recall("zzzz unknown", 5).unwrap().is_empty());
        assert!(m.recall("???", 5).unwrap().is_empty());

        assert!(m.update_item(a, "My favorite language is Python").unwrap());
        assert!(m.recall("rust", 5).unwrap().is_empty());
        assert_eq!(m.recall("python", 5).unwrap()[0].id, a);
        assert!(m.delete_item(a).unwrap());
        assert!(m.recall("python", 5).unwrap().is_empty());
        assert!(!m.delete_item(a).unwrap());
        assert!(m.remember("x".repeat(2001).as_str(), "").is_err());
    }

    #[test]
    fn hostile_queries_do_not_break_fts() {
        let m = Memory::open_in_memory().unwrap();
        m.remember("normal fact about cats", "").unwrap();
        for q in ["\"", "cats OR", "NEAR(a b)", "a\" OR \"b", "*", "cats; DROP TABLE items;--"] {
            let _ = m.recall(q, 3).unwrap();
        }
        assert_eq!(m.list_items().unwrap().len(), 1);
    }

    #[test]
    fn persistence_across_reopen_and_search_log() {
        let dir = std::env::temp_dir().join(format!("mkr_mem_{}", std::process::id()));
        let p = dir.join("m.db");
        {
            let m = Memory::open(&p).unwrap();
            m.remember("persist me please", "").unwrap();
            m.log_search("rust language", "wikipedia", 5).unwrap();
        }
        let m = Memory::open(&p).unwrap();
        assert_eq!(m.recall("persist", 3).unwrap().len(), 1);
        assert_eq!(m.search_history(10).unwrap()[0].results, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
