import os, sqlite3, time

from abc import abstractmethod, ABC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from pathlib import Path

class Transition(BaseModel):
    prev_state: int
    curr_state: int
    reward: int
    action: int

class TransitionStore(ABC):
    @abstractmethod
    def get_last(self, id: int) -> Optional[Transition]:
        ...

    @abstractmethod
    def recent(self, id: int, limit: int):
        ...

    @abstractmethod
    def update(self, id: int, transition: Transition):
        ...

class SQLiteTransitions(TransitionStore):
    def __init__(self,
                 db_path: Optional[str|Path] = None,
                 timeout: Optional[float] = None):
        self.db_path = str(db_path or os.getenv("DB_PATH", "./data/health_coach.db"))
        self.timeout = float(timeout or os.getenv("DB_TIMEOUT", "5"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=self.timeout, check_same_thread=False)
        con.execute(f"PRAGMA journal_mode={os.getenv('DB_JOURNAL_MODE','WAL')};")
        con.execute(f"PRAGMA synchronous={os.getenv('DB_SYNCHRONOUS','NORMAL')};")
        return con

    def _init_db(self):
        with self._connect() as con:
            con.executescript("""
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
            """)

    def get_last(self, id: int) -> Optional[Transition]:
        pid = str(id)
        with self._connect() as con:
            cur = con.execute(
                "SELECT prev_state, curr_state, reward, action "
                "FROM transitions WHERE patient_id=? ORDER BY id DESC LIMIT 1",
                (pid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            prev_state, curr_state, reward, action = row
            return Transition(prev_state=int(prev_state),
                              curr_state=int(curr_state),
                              reward=int(reward),
                              action=int(action))

    def update(self, id: int, transition: Transition, prob: float | None = None) -> None:
        pid = str(id)
        with self._connect() as con:
            con.execute(
                "INSERT INTO transitions (patient_id, ts, prev_state, action, reward, curr_state, prob) "
                "VALUES (?,?,?,?,?,?,?)",
                (pid, int(time.time()),
                 int(transition.prev_state), int(transition.action),
                 float(transition.reward), int(transition.curr_state),
                 None if prob is None else float(prob)),
            )

    def recent(self, patient_id: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Return up to `limit` most recent transitions as a list of dicts (oldest→newest)."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT ts, prev_state, action, reward, curr_state, prob
                FROM transitions
                WHERE patient_id = ?
                ORDER BY ts ASC
                """,
                (str(patient_id),),
            )
            rows = cur.fetchall()

        out = [
            {"ts": r[0], "prev": r[1], "action": r[2], "reward": r[3], "cur": r[4], "prob": r[5]}
            for r in rows
        ]
        return out[-limit:]


