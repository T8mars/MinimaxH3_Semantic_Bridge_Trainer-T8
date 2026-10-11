"""Read source identity without requiring Git in a portable installation."""
from pathlib import Path
import json
import subprocess

REPOSITORY = "T8mars/MinimaxH3_Semantic_Bridge_Trainer-T8"


def valid_commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def revision(path):
    root = Path(path)
    # The updater always replaces app_version.json. A legacy receipt may belong
    # to the previous release, or contain the old human-readable packaging note.
    # The release-bound commit is authoritative for this application's packages.
    app = root / "app_version.json"
    if app.is_file():
        metadata = json.loads(app.read_text(encoding="utf-8-sig"))
        if metadata.get("repository") == REPOSITORY and "source_commit" in metadata:
            commit = metadata["source_commit"]
            if not valid_commit(commit):
                raise ValueError("Invalid app_version.json source_commit; expected one Git SHA")
            return commit
    receipt = root / "SOURCE_REVISION.txt"
    if receipt.is_file():
        value = receipt.read_text(encoding="utf-8-sig").strip()
        if valid_commit(value) or value == "unversioned":
            return value
        raise ValueError("Invalid SOURCE_REVISION.txt; expected one Git SHA or unversioned")
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 and valid_commit(p.stdout.strip()) else "unversioned"
    except FileNotFoundError:
        return "unversioned"
