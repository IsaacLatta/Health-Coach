from abc import abstractmethod, ABC
from typing import Dict, Any, List, Optional
import os, json, sqlite3, time
from pathlib import Path

QTable = List[List[float]]

class QTableStore(ABC):
    @abstractmethod
    def get(self, id: int) -> QTable: 
        ...
    
    @abstractmethod
    def save(self, id: int, table: QTable) -> None:
        ...

class InMemQTables(QTableStore):
    def __init__(self, states, actions): 
        self._qtables = {}
        self._states = states
        self._actions = actions

    def get(self, id: int) -> QTable:
        qtable = self._qtables.get(id)
        if qtable is None:
            qtable = [[0.0] * self._actions for _ in range(self._states)]
            self._qtables[id] = qtable
        return qtable  

    def save(self, id: int, table: QTable): 
        self._qtables[id] = table

class SQLiteQTables(QTableStore):
    def __init__(self,
                 states: int,
                 actions: int,
                 db_path: Optional[str|Path] = None,
                 timeout: Optional[float] = None):
        self._states = int(states)
        self._actions = int(actions)
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
            CREATE TABLE IF NOT EXISTS qtables (
              patient_id TEXT PRIMARY KEY,
              states     INTEGER NOT NULL,
              actions    INTEGER NOT NULL,
              data       TEXT    NOT NULL,
              updated_ts INTEGER NOT NULL
            );
            """)

    def _zero_table(self, states: int, actions: int) -> QTable:
        return [[0.0] * actions for _ in range(states)]

    def get(self, id: int) -> QTable:
        pid = str(id)
        with self._connect() as con:
            cur = con.execute("SELECT data, states, actions FROM qtables WHERE patient_id=?", (pid,))
            row = cur.fetchone()
            if row:
                data_json, s, a = row
                try:
                    table = json.loads(data_json)
                except Exception:
                    table = self._zero_table(int(s), int(a))
                return table
            table = self._zero_table(self._states, self._actions)
            con.execute(
                "INSERT INTO qtables (patient_id, states, actions, data, updated_ts) VALUES (?,?,?,?,?)",
                (pid, self._states, self._actions, json.dumps(table), int(time.time())),
            )
            return table

    def save(self, id: int, table: QTable) -> None:
        pid = str(id)
        s = len(table)
        a = len(table[0]) if s else 0
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO qtables (patient_id, states, actions, data, updated_ts)
                VALUES (?,?,?,?,?)
                ON CONFLICT(patient_id) DO UPDATE SET
                  states=excluded.states,
                  actions=excluded.actions,
                  data=excluded.data,
                  updated_ts=excluded.updated_ts
                """,
                (pid, s, a, json.dumps(table), int(time.time())),
            )
