import json

import pytest

from libreCalphad.atomic_io import write_json_atomic, write_xml_atomic


def test_json_atomic_write(tmp_path):
    target = tmp_path / "data.json"
    write_json_atomic(target, {"a": 1}, indent=4)
    assert json.loads(target.read_text()) == {"a": 1}

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert json.loads(target.read_text()) == {"a": 1}
    assert sorted(tmp_path.iterdir()) == [target]


def test_xml_atomic_write(tmp_path):
    target = tmp_path / "database.xml"
    target.write_text("original")

    class Database:
        def to_file(self, path, if_exists):
            assert path.suffix == ".xml"
            path.write_text("replacement")

    write_xml_atomic(target, Database())
    assert target.read_text() == "replacement"

    class BrokenDatabase:
        def to_file(self, path, if_exists):
            path.write_text("partial")
            raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        write_xml_atomic(target, BrokenDatabase())
    assert target.read_text() == "replacement"
    assert sorted(tmp_path.iterdir()) == [target]
