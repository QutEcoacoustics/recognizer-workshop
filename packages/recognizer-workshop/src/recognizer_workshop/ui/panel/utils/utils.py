from pathlib import Path
from recognizer_workshop.app.agile_modeling_helpers import list_audio_files


def ppath(p: str) -> Path:
    return Path(p).expanduser().absolute()

def resolve_query_uri(query: str, labeled_dir: Path) -> str:
    #TODO: this is already done somewhere else. replace references to this one, and then delete this one
    q = str(query).strip()
    if q.isdigit():
        files = list_audio_files(labeled_dir, quiet=True)
        idx = int(q)
        if 0 <= idx < len(files):
            return str(files[idx])
        raise ValueError(f"Index {idx} out of range (0..{len(files)-1})")
    p = labeled_dir / q
    if p.exists():
        return str(p)
    return q  # URL / absolute / XC id passthrough

#TODO: this is not UI specific, move to genneral utils
def parse_target_score(s: str):
    s = s.strip()
    if s.lower() == "none":
        return None
    return float(s)








