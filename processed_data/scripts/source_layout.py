"""Resolve explicitly supplied input paths; no server-specific source mapping."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def resolve_source(original):
    file=os.environ.get('GIG_SOURCE_MAP')
    mapping=json.loads(Path(file).read_text()) if file else {}
    return Path(mapping.get(original,ROOT/original)).expanduser().resolve()
