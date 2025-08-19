from typing import Any
import hashlib
import json

def _normalize_for_hash(obj: Any):
    """
    Produce a JSON-serializable, order-stable structure:
    - dicts: sort keys (stringified) and normalize values
    - lists/tuples: normalize each item
    - sets: convert to sorted list of normalized items
    - floats: round to a stable textual form
    - everything else: leave as-is (or str() as a last resort)
    """
    if isinstance(obj, dict):
        # sort by key string so non-str keys won't break ordering
        return {str(k): _normalize_for_hash(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, set):
        # sort normalized items for deterministic order
        return sorted((_normalize_for_hash(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, float):
        # stable textual representation (adjust precision to your needs)
        return float(f"{obj:.12g}")
    # primitives (str/int/bool/None) are fine; for anything else, fall back to str
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def stable_hash(obj: Any, *, digest_size: int = 16) -> str:
    """
    Make a fast, stable hex hash of any JSON-like object.
    Uses BLAKE2b (non-cryptographic here) with configurable digest size.
    """
    canon = json.dumps(
        _normalize_for_hash(obj),
        ensure_ascii=False,
        separators=(",", ":"),   # no spaces → stable bytes
    ).encode("utf-8")
    return hashlib.blake2b(canon, digest_size=digest_size).hexdigest()

