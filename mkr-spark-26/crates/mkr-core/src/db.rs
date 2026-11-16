//! SQLite opening with integrity verification and automatic recovery from corruption.
use crate::error::{MkrError, Result};
use rusqlite::Connection;
use std::path::Path;

pub fn open_db(path: &Path) -> Result<Connection> {
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    match try_open(path) {
        Ok(c) => Ok(c),
        Err(e) if path.exists() => {
            // corrupt or unreadable: keep the broken file for forensics, start fresh
            let backup = path.with_extension(format!("corrupt-{}", crate::time::now_unix()));
            std::fs::rename(path, &backup).map_err(|re| MkrError::Db(format!("{e}; and could not move it aside: {re}")))?;
            eprintln!("warning: database {} failed integrity check ({e}); moved to {} and created a new one", path.display(), backup.display());
            try_open(path)
        }
        Err(e) => Err(e),
    }
}

fn try_open(path: &Path) -> Result<Connection> {
    let c = Connection::open(path)?;
    c.busy_timeout(std::time::Duration::from_secs(5))?;
    let ok: String = c.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    if ok != "ok" {
        return Err(MkrError::Db(format!("integrity check failed: {ok}")));
    }
    let _: String = c.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))?;
    c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=NORMAL;")?;
    Ok(c)
}

pub fn open_memory_db() -> Result<Connection> {
    let c = Connection::open_in_memory()?;
    c.execute_batch("PRAGMA foreign_keys=ON;")?;
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_file_is_moved_aside_and_recreated() {
        let dir = std::env::temp_dir().join(format!("mkr_db_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.db");
        std::fs::write(&p, b"this is definitely not a sqlite database, just garbage bytes ....................").unwrap();
        let c = open_db(&p).unwrap();
        c.execute("CREATE TABLE t(a)", []).unwrap();
        drop(c);
        let n = std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("corrupt")).count();
        assert_eq!(n, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
