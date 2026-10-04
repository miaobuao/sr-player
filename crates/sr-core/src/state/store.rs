//! The SQLite job store.

use crate::error::{Error, Result};
use crate::events::{JobState, Level, LogRecord, Stage, StageStatus};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewJob {
    pub id: String,
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub profile: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobRow {
    pub id: String,
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub state: JobState,
    pub message: Option<String>,
    pub profile: Option<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub elapsed_ms: Option<i64>,
    pub plan_json: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageRow {
    pub stage: Stage,
    pub status: StageStatus,
    pub note: Option<String>,
    pub result_json: Option<String>,
    pub updated_ms: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkRow {
    pub stage: Stage,
    pub chunk_index: u32,
    pub start_frame: Option<i64>,
    pub end_frame: Option<i64>,
    pub status: String,
    pub artifact: Option<PathBuf>,
    pub updated_ms: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResumePoint {
    /// Stages that finished successfully and whose result is stored.
    pub completed: Vec<Stage>,
    /// Stages that were interrupted and must be redone.
    pub incomplete: Vec<Stage>,
    pub committed_chunks: Vec<ChunkRow>,
}

impl ResumePoint {
    pub fn is_done(&self, stage: Stage) -> bool {
        self.completed.contains(&stage)
    }
}

pub struct Store {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl Store {
    /// Opens (creating if needed) the store at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
        }
        let conn = Connection::open(path)?;
        Self::prepare(conn, path.to_path_buf())
    }

    /// In-memory store for tests.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::prepare(conn, PathBuf::from(":memory:"))
    }

    /// The default location: `<state dir>/jobs.sqlite3`.
    pub fn default_path() -> PathBuf {
        if let Some(explicit) = std::env::var_os("SR_STATE_DB") {
            return PathBuf::from(explicit);
        }
        let base = std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("XDG_STATE_HOME"))
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("sr-player").join("jobs.sqlite3")
    }

    fn prepare(conn: Connection, path: PathBuf) -> Result<Self> {
        let store = Store {
            conn: Mutex::new(conn),
            path,
        };
        store.migrate()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock();
        // WAL keeps a crash from corrupting the file; NORMAL is the right
        // durability/speed trade for a job database.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS jobs (
                 id          TEXT PRIMARY KEY,
                 input       TEXT NOT NULL,
                 output      TEXT,
                 state       TEXT NOT NULL,
                 message     TEXT,
                 profile     TEXT,
                 plan_json   TEXT,
                 created_ms  INTEGER NOT NULL,
                 updated_ms  INTEGER NOT NULL,
                 elapsed_ms  INTEGER
             );
             CREATE TABLE IF NOT EXISTS job_stages (
                 job_id      TEXT NOT NULL,
                 stage       TEXT NOT NULL,
                 status      TEXT NOT NULL,
                 note        TEXT,
                 result_json TEXT,
                 updated_ms  INTEGER NOT NULL,
                 PRIMARY KEY (job_id, stage)
             );
             CREATE TABLE IF NOT EXISTS job_logs (
                 seq         INTEGER PRIMARY KEY AUTOINCREMENT,
                 job_id      TEXT NOT NULL,
                 at_ms       INTEGER NOT NULL,
                 level       TEXT NOT NULL,
                 stage       TEXT,
                 message     TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_job_logs ON job_logs(job_id, seq);
             CREATE TABLE IF NOT EXISTS chunks (
                 job_id      TEXT NOT NULL,
                 stage       TEXT NOT NULL,
                 chunk_index INTEGER NOT NULL,
                 start_frame INTEGER,
                 end_frame   INTEGER,
                 status      TEXT NOT NULL,
                 artifact    TEXT,
                 updated_ms  INTEGER NOT NULL,
                 PRIMARY KEY (job_id, stage, chunk_index)
             );",
        )?;
        Ok(())
    }

    // ---- jobs -------------------------------------------------------------

    pub fn create_job(&self, job: &NewJob) -> Result<()> {
        let now = now_ms();
        self.conn.lock().execute(
            "INSERT INTO jobs (id, input, output, state, profile, created_ms, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
             ON CONFLICT(id) DO UPDATE SET
                input=excluded.input, output=excluded.output, profile=excluded.profile,
                updated_ms=excluded.updated_ms",
            params![
                job.id,
                job.input.display().to_string(),
                job.output.as_ref().map(|p| p.display().to_string()),
                JobState::Queued.as_str(),
                job.profile,
                now
            ],
        )?;
        Ok(())
    }

    pub fn set_job_state(
        &self,
        job_id: &str,
        state: JobState,
        message: Option<&str>,
    ) -> Result<()> {
        self.conn.lock().execute(
            "UPDATE jobs SET state=?2, message=?3, updated_ms=?4 WHERE id=?1",
            params![job_id, state.as_str(), message, now_ms()],
        )?;
        Ok(())
    }

    pub fn finish_job(
        &self,
        job_id: &str,
        state: JobState,
        message: Option<&str>,
        elapsed_ms: i64,
    ) -> Result<()> {
        self.conn.lock().execute(
            "UPDATE jobs SET state=?2, message=?3, elapsed_ms=?4, updated_ms=?5 WHERE id=?1",
            params![job_id, state.as_str(), message, elapsed_ms, now_ms()],
        )?;
        Ok(())
    }

    pub fn set_plan(&self, job_id: &str, plan_json: &str) -> Result<()> {
        self.conn.lock().execute(
            "UPDATE jobs SET plan_json=?2, updated_ms=?3 WHERE id=?1",
            params![job_id, plan_json, now_ms()],
        )?;
        Ok(())
    }

    fn job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<JobRow> {
        let state: String = row.get("state")?;
        let output: Option<String> = row.get("output")?;
        Ok(JobRow {
            id: row.get("id")?,
            input: PathBuf::from(row.get::<_, String>("input")?),
            output: output.map(PathBuf::from),
            state: JobState::parse(&state),
            message: row.get("message")?,
            profile: row.get("profile")?,
            created_ms: row.get("created_ms")?,
            updated_ms: row.get("updated_ms")?,
            elapsed_ms: row.get("elapsed_ms")?,
            plan_json: row.get("plan_json")?,
        })
    }

    pub fn job(&self, job_id: &str) -> Result<Option<JobRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT * FROM jobs WHERE id=?1")?;
        let row = stmt
            .query_row(params![job_id], Self::job_from_row)
            .optional()?;
        Ok(row)
    }

    pub fn list_jobs(&self, limit: usize) -> Result<Vec<JobRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT * FROM jobs ORDER BY created_ms DESC LIMIT ?1")?;
        let rows = stmt
            .query_map(params![limit as i64], Self::job_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Jobs left in `running` from a previous session: they were interrupted.
    pub fn interrupted_jobs(&self) -> Result<Vec<JobRow>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT * FROM jobs WHERE state=?1 ORDER BY updated_ms DESC")?;
        let rows = stmt
            .query_map(params![JobState::Running.as_str()], Self::job_from_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ---- stages -----------------------------------------------------------

    pub fn set_stage_status(
        &self,
        job_id: &str,
        stage: Stage,
        status: StageStatus,
        note: Option<&str>,
    ) -> Result<()> {
        let status_text = format!("{status:?}").to_lowercase();
        self.conn.lock().execute(
            "INSERT INTO job_stages (job_id, stage, status, note, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(job_id, stage) DO UPDATE SET
                status=excluded.status,
                note=COALESCE(excluded.note, job_stages.note),
                updated_ms=excluded.updated_ms",
            params![job_id, stage.id(), status_text, note, now_ms()],
        )?;
        Ok(())
    }

    /// Stores a stage result *and* marks the stage done in one transaction, so a
    /// resumed run can never see a 'done' stage without its payload.
    pub fn commit_stage_result(&self, job_id: &str, stage: Stage, json: &str) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO job_stages (job_id, stage, status, result_json, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(job_id, stage) DO UPDATE SET
                status=excluded.status, result_json=excluded.result_json,
                updated_ms=excluded.updated_ms",
            params![
                job_id,
                stage.id(),
                "done",
                json,
                now_ms()
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn load_stage_result(&self, job_id: &str, stage: Stage) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT result_json FROM job_stages WHERE job_id=?1 AND stage=?2")?;
        let value: Option<Option<String>> = stmt
            .query_row(params![job_id, stage.id()], |row| row.get(0))
            .optional()?;
        Ok(value.flatten())
    }

    pub fn stage_rows(&self, job_id: &str) -> Result<Vec<StageRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT stage, status, note, result_json, updated_ms FROM job_stages WHERE job_id=?1",
        )?;
        let rows = stmt
            .query_map(params![job_id], |row| {
                let stage: String = row.get(0)?;
                let status: String = row.get(1)?;
                Ok(StageRow {
                    stage: parse_stage(&stage),
                    status: parse_status(&status),
                    note: row.get(2)?,
                    result_json: row.get(3)?,
                    updated_ms: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// What a resumed run may skip.
    pub fn resume_point(&self, job_id: &str) -> Result<ResumePoint> {
        let mut point = ResumePoint::default();
        for row in self.stage_rows(job_id)? {
            match row.status {
                StageStatus::Done | StageStatus::Skipped => point.completed.push(row.stage),
                StageStatus::Running | StageStatus::Degraded => point.incomplete.push(row.stage),
                StageStatus::Pending | StageStatus::Failed => {}
            }
        }
        point.committed_chunks = self
            .chunks(job_id)?
            .into_iter()
            .filter(|c| c.status == "committed")
            .collect();
        Ok(point)
    }

    // ---- logs -------------------------------------------------------------

    pub fn append_log(&self, job_id: &str, record: &LogRecord) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO job_logs (job_id, at_ms, level, stage, message) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                job_id,
                record.at_ms as i64,
                record.level.as_str(),
                record.stage.map(|s| s.id()),
                record.message
            ],
        )?;
        Ok(())
    }

    pub fn logs(&self, job_id: &str, limit: usize, min_level: Level) -> Result<Vec<LogRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT seq, at_ms, level, stage, message FROM job_logs
             WHERE job_id=?1 ORDER BY seq DESC LIMIT ?2",
        )?;
        let mut rows = stmt
            .query_map(params![job_id, limit as i64], |row| {
                let level: String = row.get(2)?;
                let stage: Option<String> = row.get(3)?;
                Ok(LogRecord {
                    seq: row.get::<_, i64>(0)? as u64,
                    at_ms: row.get::<_, i64>(1)? as u64,
                    level: Level::parse(&level),
                    stage: stage.as_deref().map(parse_stage),
                    message: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows.retain(|r| r.level >= min_level);
        rows.reverse();
        Ok(rows)
    }

    // ---- chunks -----------------------------------------------------------

    /// Records a completed unit of work. The artifact must already be durable.
    pub fn commit_chunk(&self, job_id: &str, chunk: &ChunkRow) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO chunks (job_id, stage, chunk_index, start_frame, end_frame, status, artifact, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(job_id, stage, chunk_index) DO UPDATE SET
                start_frame=excluded.start_frame, end_frame=excluded.end_frame,
                status=excluded.status, artifact=excluded.artifact,
                updated_ms=excluded.updated_ms",
            params![
                job_id,
                chunk.stage.id(),
                chunk.chunk_index as i64,
                chunk.start_frame,
                chunk.end_frame,
                chunk.status,
                chunk.artifact.as_ref().map(|p| p.display().to_string()),
                now_ms()
            ],
        )?;
        Ok(())
    }

    pub fn chunks(&self, job_id: &str) -> Result<Vec<ChunkRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT stage, chunk_index, start_frame, end_frame, status, artifact, updated_ms
             FROM chunks WHERE job_id=?1 ORDER BY stage, chunk_index",
        )?;
        let rows = stmt
            .query_map(params![job_id], |row| {
                let stage: String = row.get(0)?;
                let artifact: Option<String> = row.get(5)?;
                Ok(ChunkRow {
                    stage: parse_stage(&stage),
                    chunk_index: row.get::<_, i64>(1)? as u32,
                    start_frame: row.get(2)?,
                    end_frame: row.get(3)?,
                    status: row.get(4)?,
                    artifact: artifact.map(PathBuf::from),
                    updated_ms: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Trims the log for a job, keeping the most recent `keep` rows.
    pub fn trim_logs(&self, job_id: &str, keep: usize) -> Result<usize> {
        let deleted = self.conn.lock().execute(
            "DELETE FROM job_logs WHERE job_id=?1 AND seq NOT IN
             (SELECT seq FROM job_logs WHERE job_id=?1 ORDER BY seq DESC LIMIT ?2)",
            params![job_id, keep as i64],
        )?;
        Ok(deleted)
    }
}

fn parse_stage(id: &str) -> Stage {
    Stage::ALL
        .iter()
        .copied()
        .find(|s| s.id() == id)
        .unwrap_or(Stage::Probe)
}

fn parse_status(text: &str) -> StageStatus {
    match text {
        "running" => StageStatus::Running,
        "done" => StageStatus::Done,
        "skipped" => StageStatus::Skipped,
        "degraded" => StageStatus::Degraded,
        "failed" => StageStatus::Failed,
        _ => StageStatus::Pending,
    }
}

/// Publishes a finished artifact: fsync, then rename over the destination.
///
/// On Windows `rename` refuses to replace an existing file, so the destination
/// is removed first — after the source is durable, so a crash in between leaves
/// the temporary file, never a half-written output.
pub fn commit_file_atomic(temp: &Path, destination: &Path) -> Result<()> {
    if !temp.exists() {
        return Err(Error::State(format!(
            "cannot commit {}: the temporary file does not exist",
            temp.display()
        )));
    }
    {
        // Opened read+write: `sync_all` needs write access, and this is the
        // durability barrier the whole commit rule depends on.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(temp)
            .map_err(|e| Error::io(temp, e))?;
        file.sync_all().map_err(|e| Error::io(temp, e))?;
    }
    if let Some(parent) = destination.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
    }
    if destination.exists() {
        std::fs::remove_file(destination).map_err(|e| Error::io(destination, e))?;
    }
    std::fs::rename(temp, destination).map_err(|e| Error::io(destination, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Store {
        Store::open_in_memory().expect("in-memory store")
    }

    fn job(id: &str) -> NewJob {
        NewJob {
            id: id.into(),
            input: PathBuf::from("in.mkv"),
            output: Some(PathBuf::from("out.mkv")),
            profile: "rtx-5070ti-safe".into(),
        }
    }

    #[test]
    fn creates_and_reads_back_a_job() {
        let store = store();
        store.create_job(&job("j1")).unwrap();
        let row = store.job("j1").unwrap().unwrap();
        assert_eq!(row.input, PathBuf::from("in.mkv"));
        assert_eq!(row.state, JobState::Queued);
        store
            .finish_job("j1", JobState::Completed, Some("done"), 1234)
            .unwrap();
        let row = store.job("j1").unwrap().unwrap();
        assert_eq!(row.state, JobState::Completed);
        assert_eq!(row.elapsed_ms, Some(1234));
    }

    #[test]
    fn stage_result_and_status_are_written_together() {
        let store = store();
        store.create_job(&job("j2")).unwrap();
        store
            .commit_stage_result("j2", Stage::Probe, r#"{"duration":72.072}"#)
            .unwrap();
        assert_eq!(
            store.load_stage_result("j2", Stage::Probe).unwrap().as_deref(),
            Some(r#"{"duration":72.072}"#)
        );
        let rows = store.stage_rows("j2").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, StageStatus::Done);
    }

    #[test]
    fn resume_point_separates_finished_from_interrupted_work() {
        let store = store();
        store.create_job(&job("j3")).unwrap();
        store
            .commit_stage_result("j3", Stage::Probe, "{}")
            .unwrap();
        store
            .set_stage_status("j3", Stage::Scenes, StageStatus::Running, None)
            .unwrap();
        store
            .set_stage_status("j3", Stage::Encode, StageStatus::Pending, None)
            .unwrap();
        let point = store.resume_point("j3").unwrap();
        assert!(point.is_done(Stage::Probe));
        assert_eq!(point.incomplete, vec![Stage::Scenes]);
        assert!(!point.is_done(Stage::Encode));
    }

    #[test]
    fn logs_round_trip_with_level_filtering() {
        let store = store();
        store.create_job(&job("j4")).unwrap();
        for (level, text) in [
            (Level::Debug, "detail"),
            (Level::Info, "started"),
            (Level::Warn, "slow"),
            (Level::Error, "boom"),
        ] {
            store
                .append_log("j4", &LogRecord::now(0, level, Some(Stage::Encode), text))
                .unwrap();
        }
        let all = store.logs("j4", 100, Level::Trace).unwrap();
        assert_eq!(all.len(), 4);
        assert_eq!(all.first().unwrap().message, "detail");
        let warnings = store.logs("j4", 100, Level::Warn).unwrap();
        assert_eq!(warnings.len(), 2);
        assert_eq!(warnings[0].level, Level::Warn);
        assert_eq!(warnings[0].stage, Some(Stage::Encode));
    }

    #[test]
    fn log_trimming_keeps_the_newest_lines() {
        let store = store();
        store.create_job(&job("j5")).unwrap();
        for index in 0..50 {
            store
                .append_log("j5", &LogRecord::now(0, Level::Info, None, format!("line {index}")))
                .unwrap();
        }
        store.trim_logs("j5", 10).unwrap();
        let rows = store.logs("j5", 100, Level::Trace).unwrap();
        assert_eq!(rows.len(), 10);
        assert_eq!(rows.last().unwrap().message, "line 49");
    }

    #[test]
    fn chunks_are_upserted_per_stage_and_index() {
        let store = store();
        store.create_job(&job("j6")).unwrap();
        let chunk = ChunkRow {
            stage: Stage::Encode,
            chunk_index: 0,
            start_frame: Some(0),
            end_frame: Some(1000),
            status: "committed".into(),
            artifact: Some(PathBuf::from("chunk0.mkv")),
            updated_ms: 0,
        };
        store.commit_chunk("j6", &chunk).unwrap();
        store
            .commit_chunk(
                "j6",
                &ChunkRow {
                    end_frame: Some(2000),
                    ..chunk.clone()
                },
            )
            .unwrap();
        let chunks = store.chunks("j6").unwrap();
        assert_eq!(chunks.len(), 1, "same stage+index must upsert");
        assert_eq!(chunks[0].end_frame, Some(2000));
        assert_eq!(store.resume_point("j6").unwrap().committed_chunks.len(), 1);
    }

    #[test]
    fn interrupted_jobs_are_discoverable_for_resume() {
        let store = store();
        store.create_job(&job("j7")).unwrap();
        store
            .set_job_state("j7", JobState::Running, None)
            .unwrap();
        let interrupted = store.interrupted_jobs().unwrap();
        assert_eq!(interrupted.len(), 1);
        assert_eq!(interrupted[0].id, "j7");
    }

    #[test]
    fn atomic_commit_replaces_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let temp = dir.path().join("chunk.part");
        let final_path = dir.path().join("chunk.mkv");
        std::fs::write(&temp, b"new bytes").unwrap();
        std::fs::write(&final_path, b"old bytes").unwrap();
        commit_file_atomic(&temp, &final_path).unwrap();
        assert_eq!(std::fs::read(&final_path).unwrap(), b"new bytes");
        assert!(!temp.exists());
    }

    #[test]
    fn atomic_commit_refuses_a_missing_source() {
        let dir = tempfile::tempdir().unwrap();
        let err = commit_file_atomic(&dir.path().join("nope.part"), &dir.path().join("x.mkv"))
            .unwrap_err();
        assert!(matches!(err, Error::State(_)));
    }
}
