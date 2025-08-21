CREATE TABLE IF NOT EXISTS qtables (
  patient_id TEXT PRIMARY KEY,
  states     INTEGER NOT NULL,
  actions    INTEGER NOT NULL,
  data       TEXT    NOT NULL,
  updated_ts INTEGER NOT NULL 
);

CREATE TABLE IF NOT EXISTS transitions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id TEXT    NOT NULL,
  ts         INTEGER NOT NULL,
  prev_state INTEGER NOT NULL,
  action     INTEGER NOT NULL,
  reward     REAL    NOT NULL,
  curr_state INTEGER NOT NULL,
  prob       REAL             
);

CREATE INDEX IF NOT EXISTS idx_transitions_patient_ts
  ON transitions(patient_id, ts DESC);

CREATE TABLE IF NOT EXISTS configs (
  patient_id TEXT PRIMARY KEY,
  data       TEXT    NOT NULL,   
  updated_ts INTEGER NOT NULL
);
