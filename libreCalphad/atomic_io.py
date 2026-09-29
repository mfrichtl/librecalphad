"""Atomic writes for JSON data and pycalphad XML databases."""

import json
import os
from pathlib import Path
import tempfile


def _atomic_write(path, writer):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=path.suffix,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer(stream, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path, data, **kwargs):
    """Serialize JSON and replace path only after serialization succeeds."""
    _atomic_write(path, lambda stream, temporary: json.dump(data, stream, **kwargs))


def write_xml_atomic(path, database):
    """Write a pycalphad database to XML and atomically replace path."""

    def write(stream, temporary):
        stream.close()
        database.to_file(temporary, if_exists="overwrite")

    _atomic_write(path, write)
