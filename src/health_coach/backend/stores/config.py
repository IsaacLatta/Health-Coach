from abc import abstractmethod, ABC
from typing import Dict, Any
import os, json, sqlite3, time
from pathlib import Path
from typing import Optional, Dict, Any

class ConfigStore(ABC):
    @abstractmethod
    def get(self, id) -> Dict[str, Any]:
        ...

    @abstractmethod
    def save(self, id, config: Dict[str, Any]):
        ...

class InMemConfigs(ConfigStore):
    def __init__(self, defaults=None): 
        self._configs = {}
        self._default_config = defaults or {"moderate":0.33,"high":0.66,"top_k":5}
    
    def get(self, id: int): 
        return self._configs.get(id, dict(self._default_config))
    
    def save(self, id, cfg): 
        self._configs[id] = dict(cfg)

class SQLiteConfigs(ConfigStore):
    def __init__(self,
                 defaults: Optional[Dict[str, Any]] = None,
                 db_path: Optional[str|Path] = None,
                 timeout: Optional[float] = None):
        self._defaults = defaults or {"moderate": 0.33, "high": 0.66, "top_k": 5}
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
            CREATE TABLE IF NOT EXISTS configs (
              patient_id TEXT PRIMARY KEY,
              data       TEXT    NOT NULL,
              updated_ts INTEGER NOT NULL
            );
            """)

    def get(self, id: int) -> Dict[str, Any]:
        pid = str(id)
        with self._connect() as con:
            cur = con.execute("SELECT data FROM configs WHERE patient_id=?", (pid,))
            row = cur.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
            return dict(self._defaults)

    def save(self, id: int, config: Dict[str, Any]):
        pid = str(id)
        data = json.dumps(dict(config))
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO configs (patient_id, data, updated_ts)
                VALUES (?,?,?)
                ON CONFLICT(patient_id) DO UPDATE SET
                  data=excluded.data,
                  updated_ts=excluded.updated_ts
                """,
                (pid, data, int(time.time())),
            )



