from pathlib import Path

from nodifyctx.parser import CodeParser


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")


def _entity_by_path(parsed: dict[str, list[dict[str, object]]], entity_type: str, file_path: str) -> dict[str, object]:
    return next(entity for entity in parsed["entities"] if entity["type"] == entity_type and entity["file_path"] == file_path)


def _entity_by_qualified_name(parsed: dict[str, list[dict[str, object]]], qualified_name: str) -> dict[str, object]:
    return next(entity for entity in parsed["entities"] if entity["qualified_name"] == qualified_name)


def test_parser_includes_folder_file_and_symbol_hierarchy(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "pkg" / "service.py",
        "class Greeter:\n"
        "    def greet(self):\n"
        "        helper()\n\n"
        "def helper():\n"
        "    return 'hi'\n",
    )

    parsed = CodeParser(tmp_path).parse()

    root_folder = _entity_by_path(parsed, "folder", ".")
    package_folder = _entity_by_path(parsed, "folder", "pkg")
    source_file = _entity_by_path(parsed, "file", "pkg/service.py")
    greeter_class = _entity_by_qualified_name(parsed, "pkg.service.Greeter")
    greet_method = _entity_by_qualified_name(parsed, "pkg.service.Greeter.greet")
    helper_function = _entity_by_qualified_name(parsed, "pkg.service.helper")

    assert "pkg" in root_folder["raw_code"]
    assert "service.py" in package_folder["raw_code"]
    assert "class Greeter" in source_file["raw_code"]

    contains_edges = {
        (edge["source_id"], edge["target_id"], edge["type"])
        for edge in parsed["relationships"]
    }

    assert (root_folder["id"], package_folder["id"], "CONTAINS") in contains_edges
    assert (package_folder["id"], source_file["id"], "CONTAINS") in contains_edges
    assert (source_file["id"], greeter_class["id"], "CONTAINS") in contains_edges
    assert (source_file["id"], helper_function["id"], "CONTAINS") in contains_edges
    assert (greeter_class["id"], greet_method["id"], "CONTAINS") in contains_edges
    assert (greet_method["id"], helper_function["id"], "CALLS") in contains_edges