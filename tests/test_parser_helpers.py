from pathlib import Path

import pytest

from nodifyctx.parser import CodeParser, ParsedEntity, ParsedRelationship


def test_parser_validates_paths_and_unsupported_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CodeParser(tmp_path / "missing")

    file_path = tmp_path / "not_dir.py"
    file_path.write_text("print('x')", encoding="utf8")
    with pytest.raises(NotADirectoryError):
        CodeParser(file_path)


def test_iter_source_files_excludes_ignored_directories(tmp_path: Path):
    included = tmp_path / "src" / "app.py"
    ignored = tmp_path / "node_modules" / "lib.py"
    included.parent.mkdir(parents=True)
    ignored.parent.mkdir(parents=True)
    included.write_text("print('ok')", encoding="utf8")
    ignored.write_text("print('skip')", encoding="utf8")

    parser = CodeParser(tmp_path)
    files = parser._iter_source_files()

    assert files == [included]
    with pytest.raises(ValueError):
        parser._language_spec_for_file(tmp_path / "README.md")


def test_resolve_call_target_prefers_parent_then_file_then_global(tmp_path: Path):
    parser = CodeParser(tmp_path)
    source = ParsedEntity("1", "function", "call", "pkg.call", "pkg.py", 1, 2, "", parent_id="parent-1")
    parent_target = ParsedEntity("2", "function", "helper", "pkg.Parent.helper", "pkg.py", 3, 4, "")
    file_target = ParsedEntity("3", "function", "helper", "pkg.helper", "pkg.py", 5, 6, "")
    global_target = ParsedEntity("4", "function", "helper", "other.helper", "other.py", 1, 2, "")

    assert parser._resolve_call_target(
        source=source,
        call_name="helper",
        functions_by_name={"helper": [global_target]},
        functions_by_file_and_name={("pkg.py", "helper"): [file_target]},
        functions_by_parent_and_name={("parent-1", "helper"): [parent_target]},
    ) is parent_target

    source.parent_id = None
    assert parser._resolve_call_target(
        source=source,
        call_name="helper",
        functions_by_name={"helper": [global_target]},
        functions_by_file_and_name={("pkg.py", "helper"): [file_target]},
        functions_by_parent_and_name={},
    ) is file_target

    assert parser._resolve_call_target(
        source=ParsedEntity("1", "function", "call", "other.call", "other.py", 1, 2, ""),
        call_name="helper",
        functions_by_name={"helper": [global_target]},
        functions_by_file_and_name={},
        functions_by_parent_and_name={},
    ) is global_target


def test_dedup_helpers_preserve_order_and_uniqueness():
    assert CodeParser._dedupe_preserving_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    relationships = [
        ParsedRelationship("1", "a", "2", "b", "CALLS"),
        ParsedRelationship("1", "a", "2", "b", "CALLS"),
        ParsedRelationship("1", "a", "3", "c", "CONTAINS"),
    ]
    unique = CodeParser._dedupe_relationships(relationships)

    assert [(relationship.target_id, relationship.type) for relationship in unique] == [("2", "CALLS"), ("3", "CONTAINS")]