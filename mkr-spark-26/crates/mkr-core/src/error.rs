use std::fmt;

#[derive(Debug)]
pub enum MkrError {
    Io(String),
    Db(String),
    Config(String),
    Network(String),
    Invalid(String),
    Model(String),
}

pub type Result<T> = std::result::Result<T, MkrError>;

impl fmt::Display for MkrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (k, m) = match self {
            MkrError::Io(m) => ("I/O error", m),
            MkrError::Db(m) => ("database error", m),
            MkrError::Config(m) => ("configuration error", m),
            MkrError::Network(m) => ("network error", m),
            MkrError::Invalid(m) => ("invalid input", m),
            MkrError::Model(m) => ("model error", m),
        };
        write!(f, "{k}: {m}")
    }
}
impl std::error::Error for MkrError {}
impl From<std::io::Error> for MkrError {
    fn from(e: std::io::Error) -> Self {
        MkrError::Io(e.to_string())
    }
}
impl From<rusqlite::Error> for MkrError {
    fn from(e: rusqlite::Error) -> Self {
        MkrError::Db(e.to_string())
    }
}
