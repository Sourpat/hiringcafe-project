import os
import pickle
from collections import OrderedDict
import numpy as np


def read_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def read_bool_env(name, default):
    return os.getenv(name, str(int(default))) == "1"


def query_cache_max():
    return read_int_env("QUERY_CACHE_MAX", 500)


def query_cache_write():
    return read_bool_env("QUERY_CACHE_WRITE", True)


def query_cache_prewarm_write():
    return read_bool_env("QUERY_CACHE_PREWARM_WRITE", False)


def query_cache_path():
    return os.getenv("QUERY_CACHE_PATH", os.path.join("data", "query_vec_cache.pkl"))


def normalize_query_key(q):
    return " ".join(q.strip().lower().split())


def should_log_cache_debug():
    return os.getenv("DEBUG_CACHE") == "1" and os.getenv("PYTEST_CURRENT_TEST") is None


class QueryEmbeddingCache:
    def __init__(
        self,
        max_size=None,
        path=None,
        enable_disk=None,
        embedding_dim=1536,
        debug=False,
    ):
        """
        Product-grade query embedding cache.

        Backward compatible with existing env-driven defaults.
        """
        self.max_size = int(max_size) if max_size is not None else int(query_cache_max())
        self.path = path or query_cache_path()
        self.enable_disk = bool(enable_disk) if enable_disk is not None else False
        self.embedding_dim = embedding_dim
        self.debug = debug
        self._store = OrderedDict()
        self._disk_keys = set()
        self._loaded = False
        self._evictions = 0
        self._hits = 0
        self._misses = 0

    def load(self):
        if self._loaded:
            return
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    items = list(loaded.items())
                else:
                    items = list(loaded)
                trimmed_on_load = 0
                if len(items) > self.max_size:
                    trimmed_on_load = len(items) - self.max_size
                    items = items[-self.max_size:]
                self._store = OrderedDict()
                for k, v in items:
                    key = normalize_query_key(str(k))
                    vec = np.asarray(v, dtype=np.float32)
                    if vec.shape == (self.embedding_dim,):
                        self._store[key] = vec
                self._disk_keys = set(self._store.keys())
                if trimmed_on_load:
                    self.save()
                print(
                    f"Query cache: loaded_from_disk=true, entries={len(self._store)}, "
                    f"trimmed_on_load={trimmed_on_load}, path={self.path}"
                )
            except Exception:
                self._store = OrderedDict()
                self._disk_keys = set()
                print(
                    f"Query cache: loaded_from_disk=false, entries=0, trimmed_on_load=0, "
                    f"path={self.path}"
                )
        else:
            print(
                f"Query cache: loaded_from_disk=false, entries=0, trimmed_on_load=0, "
                f"path={self.path}"
            )
        self._loaded = True
        assert len(self._store) <= self.max_size

    def get(self, key, *, with_source: bool = True):
        self.load()
        norm_key = normalize_query_key(str(key))
        vec = self._store.get(norm_key)
        if vec is None:
            self._misses += 1
            return (None, None) if with_source else None
        if not isinstance(vec, np.ndarray) or vec.dtype != np.float32 or vec.shape != (self.embedding_dim,):
            self._store.pop(norm_key, None)
            self._misses += 1
            return (None, None) if with_source else None
        self._store.move_to_end(norm_key)
        self._hits += 1
        if not with_source:
            return vec
        if norm_key in self._disk_keys:
            self._disk_keys.discard(norm_key)
            return vec, "disk"
        return vec, "mem"

    def put(self, key, vec):
        self.load()
        norm_key = normalize_query_key(str(key))
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (self.embedding_dim,):
            raise ValueError(f"Embedding must have shape ({self.embedding_dim},), got {vec.shape}")
        self._store[norm_key] = vec
        self._store.move_to_end(norm_key)
        self._evict_if_needed()

    # Backward compatible alias
    def set(self, key, vec):
        return self.put(key, vec)

    def _evict_if_needed(self):
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
            self._evictions += 1
        assert len(self._store) <= self.max_size

    def save(self, *, force: bool = False) -> bool:
        if not (self.enable_disk or force):
            return False
        try:
            from pathlib import Path

            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._evict_if_needed()
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(self._store, f)
            os.replace(tmp_path, self.path)
            assert len(self._store) <= self.max_size
            return True
        except Exception:
            return False

    def size(self):
        return len(self._store)

    def keys_mru(self):
        keys = list(self._store.keys())
        if len(keys) > self.max_size:
            keys = keys[-self.max_size:]
        return list(reversed(keys))

    def evictions(self):
        return self._evictions

    def stats(self):
        return {
            "hits": getattr(self, "_hits", 0),
            "misses": getattr(self, "_misses", 0),
            "evictions": getattr(self, "_evictions", 0),
            "size": len(getattr(self, "_store", {})),
            "max_size": getattr(self, "max_size", 0),
        }

    def get_stats(self):
        """Audit-friendly stats contract."""
        hits = int(getattr(self, "_hits", 0))
        misses = int(getattr(self, "_misses", 0))
        return {
            "total_requests": hits + misses,
            "hits": hits,
            "misses": misses,
            "evictions": int(getattr(self, "_evictions", 0)),
            "current_size": len(getattr(self, "_store", {})),
        }
