from pathlib import Path
from typing import Union

from recognizer_workshop.app.agile_modeling_helpers import list_audio_files

def resolve_query_uri(query: Union[str, int], labeled_examples_folder: Path) -> str:
    s = str(query).strip()
    
    # xc ids need to be lowercased
    if s.upper().startswith('XC'):
        return s.lower()
    
    # We can accept a file number (index into the list of files in labeled_examples_folder)
    # used in some UIs
    if s.isdigit():
        files = list_audio_files(labeled_examples_folder, quiet=True)
        i = int(s)
        if i < 0 or i >= len(files):
            raise ValueError(f"Index {i} out of range (0..{len(files)-1})")
        return str(files[i])
    
    # relative filename inside labeled_examples_folder
    p = (labeled_examples_folder / s)
    if p.exists():
        return str(p)
    
    # otherwise assume absolute path or URL
    return s

