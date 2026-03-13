from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser, Query, QueryCursor


PYTHON_LANGUAGE = Language(tspython.language())
JAVASCRIPT_LANGUAGE = Language(tsjavascript.language())
TYPESCRIPT_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    name: str
    file_suffixes: tuple[str, ...]
    language: Language
    definition_query: Query
    call_query: Query
    class_ancestor_types: tuple[str, ...]


def _python_language_spec() -> LanguageSpec:
    return LanguageSpec(
        name="python",
        file_suffixes=(".py",),
        language=PYTHON_LANGUAGE,
        definition_query=Query(
            PYTHON_LANGUAGE,
            """
            (function_definition
                name: (identifier) @function.name) @function.node

            (class_definition
                name: (identifier) @class.name) @class.node
            """,
        ),
        call_query=Query(
            PYTHON_LANGUAGE,
            """
            (call
                function: [
                    (identifier) @call.name
                    (attribute
                        attribute: (identifier) @call.name)
                ]) @call.node
            """,
        ),
        class_ancestor_types=("class_definition",),
    )


def _javascript_definition_query(language: Language) -> Query:
    return Query(
        language,
        """
        (function_declaration
            name: (identifier) @function.name) @function.node

        (method_definition
            name: [
                (property_identifier)
                (private_property_identifier)
            ] @function.name) @function.node

        (field_definition
            property: [
                (property_identifier)
                (private_property_identifier)
            ] @function.name
            value: [
                (arrow_function) @function.owner
                (function_expression) @function.owner
            ]) @function.node

        [
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @function.name
                    value: [
                        (arrow_function) @function.owner
                        (function_expression) @function.owner
                    ]) @function.node)
            (variable_declaration
                (variable_declarator
                    name: (identifier) @function.name
                    value: [
                        (arrow_function) @function.owner
                        (function_expression) @function.owner
                    ]) @function.node)
        ]

        (class_declaration
            name: (identifier) @class.name) @class.node
        """,
    )


def _typescript_definition_query(language: Language) -> Query:
    return Query(
        language,
        """
        (function_declaration
            name: (identifier) @function.name) @function.node

        (method_definition
            name: [
                (property_identifier)
                (private_property_identifier)
            ] @function.name) @function.node

        (public_field_definition
            name: [
                (property_identifier)
                (private_property_identifier)
            ] @function.name
            value: [
                (arrow_function) @function.owner
                (function_expression) @function.owner
            ]) @function.node

        [
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @function.name
                    value: [
                        (arrow_function) @function.owner
                        (function_expression) @function.owner
                    ]) @function.node)
            (variable_declaration
                (variable_declarator
                    name: (identifier) @function.name
                    value: [
                        (arrow_function) @function.owner
                        (function_expression) @function.owner
                    ]) @function.node)
        ]

        (class_declaration
            name: (type_identifier) @class.name) @class.node
        """,
    )


def _javascript_like_call_query(language: Language) -> Query:
    return Query(
        language,
        """
        (call_expression
            function: [
                (identifier) @call.name
                (member_expression
                    property: [
                        (property_identifier)
                        (private_property_identifier)
                    ] @call.name)
            ]) @call.node
        """,
    )


LANGUAGE_SPECS = (
    _python_language_spec(),
    LanguageSpec(
        name="javascript",
        file_suffixes=(".js", ".jsx"),
        language=JAVASCRIPT_LANGUAGE,
        definition_query=_javascript_definition_query(JAVASCRIPT_LANGUAGE),
        call_query=_javascript_like_call_query(JAVASCRIPT_LANGUAGE),
        class_ancestor_types=("class_declaration",),
    ),
    LanguageSpec(
        name="typescript",
        file_suffixes=(".ts",),
        language=TYPESCRIPT_LANGUAGE,
        definition_query=_typescript_definition_query(TYPESCRIPT_LANGUAGE),
        call_query=_javascript_like_call_query(TYPESCRIPT_LANGUAGE),
        class_ancestor_types=("class_declaration",),
    ),
    LanguageSpec(
        name="tsx",
        file_suffixes=(".tsx",),
        language=TSX_LANGUAGE,
        definition_query=_typescript_definition_query(TSX_LANGUAGE),
        call_query=_javascript_like_call_query(TSX_LANGUAGE),
        class_ancestor_types=("class_declaration",),
    ),
)


@dataclass(slots=True)
class ParsedEntity:
    id: str
    type: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    raw_code: str
    parent_id: str | None = None
    parent_name: str | None = None
    calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedRelationship:
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodeParser:
    """Parse Python, JavaScript, TypeScript, and React code into code entities."""

    DEFAULT_EXCLUDES = {
        ".git",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }

    def __init__(
        self,
        directory_path: str | Path,
        *,
        exclude_directories: set[str] | None = None,
    ) -> None:
        self.root_path = Path(directory_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.root_path}")

        self.exclude_directories = exclude_directories or self.DEFAULT_EXCLUDES
        self.language_specs_by_suffix = {
            suffix: spec
            for spec in LANGUAGE_SPECS
            for suffix in spec.file_suffixes
        }
        self.parsers = {
            spec.name: Parser(spec.language)
            for spec in LANGUAGE_SPECS
        }

    def parse(self) -> dict[str, list[dict[str, Any]]]:
        entities: list[ParsedEntity] = []
        relationships: list[ParsedRelationship] = []

        source_files = self._iter_source_files()
        structure_entities, structure_relationships, file_entities_by_path = self._build_structure_entities(source_files)
        entities.extend(structure_entities)
        relationships.extend(structure_relationships)

        for source_file in source_files:
            file_entity = file_entities_by_path[str(source_file.relative_to(self.root_path).as_posix())]
            file_entities, file_relationships = self._parse_file(source_file, file_entity=file_entity)
            entities.extend(file_entities)
            relationships.extend(file_relationships)

        relationships.extend(self._resolve_call_relationships(entities))

        return {
            "entities": [entity.to_dict() for entity in entities],
            "relationships": [relationship.to_dict() for relationship in self._dedupe_relationships(relationships)],
        }

    def _iter_source_files(self) -> list[Path]:
        supported_suffixes = set(self.language_specs_by_suffix)
        return sorted(
            [
                path
                for path in self.root_path.rglob("*")
                if path.is_file()
                and path.suffix in supported_suffixes
                and not any(part in self.exclude_directories for part in path.parts)
            ],
            key=lambda path: path.as_posix(),
        )

    def _build_structure_entities(
        self,
        source_files: list[Path],
    ) -> tuple[list[ParsedEntity], list[ParsedRelationship], dict[str, ParsedEntity]]:
        entities: list[ParsedEntity] = []
        relationships: list[ParsedRelationship] = []
        folder_entities: dict[str, ParsedEntity] = {}
        file_entities: dict[str, ParsedEntity] = {}
        folder_children: dict[str, list[str]] = {}

        root_entity = ParsedEntity(
            id=self._entity_id(
                entity_type="folder",
                file_path=".",
                qualified_name=self.root_path.name,
                start_line=0,
                end_line=0,
            ),
            type="folder",
            name=self.root_path.name,
            qualified_name=self.root_path.name,
            file_path=".",
            start_line=0,
            end_line=0,
            raw_code=f"Folder . (repository root: {self.root_path.name})",
        )
        folder_entities["."] = root_entity
        entities.append(root_entity)

        for source_file in source_files:
            relative_file = source_file.relative_to(self.root_path).as_posix()
            parent_relative = Path(relative_file).parent.as_posix()
            if parent_relative == ".":
                parent_relative = "."

            current_path = Path(relative_file).parent
            ancestors: list[str] = []
            while current_path != Path("."):
                ancestors.append(current_path.as_posix())
                current_path = current_path.parent
            for folder_path in reversed(ancestors):
                if folder_path in folder_entities:
                    continue

                parent_path = Path(folder_path).parent.as_posix()
                if parent_path == ".":
                    parent_path = "."
                parent_entity = folder_entities[parent_path]
                folder_name = Path(folder_path).name
                folder_entity = ParsedEntity(
                    id=self._entity_id(
                        entity_type="folder",
                        file_path=folder_path,
                        qualified_name=folder_path,
                        start_line=0,
                        end_line=0,
                    ),
                    type="folder",
                    name=folder_name,
                    qualified_name=folder_path,
                    file_path=folder_path,
                    start_line=0,
                    end_line=0,
                    raw_code=f"Folder {folder_path}",
                    parent_id=parent_entity.id,
                    parent_name=parent_entity.name,
                )
                folder_entities[folder_path] = folder_entity
                entities.append(folder_entity)
                relationships.append(
                    ParsedRelationship(
                        source_id=parent_entity.id,
                        source_name=parent_entity.name,
                        target_id=folder_entity.id,
                        target_name=folder_entity.name,
                        type="CONTAINS",
                    )
                )
                folder_children.setdefault(parent_path, []).append(folder_name)

            file_content = source_file.read_text(encoding="utf8")
            file_entity = ParsedEntity(
                id=self._entity_id(
                    entity_type="file",
                    file_path=relative_file,
                    qualified_name=relative_file,
                    start_line=1 if file_content else 0,
                    end_line=file_content.count("\n") + 1 if file_content else 0,
                ),
                type="file",
                name=source_file.name,
                qualified_name=relative_file,
                file_path=relative_file,
                start_line=1 if file_content else 0,
                end_line=file_content.count("\n") + 1 if file_content else 0,
                raw_code=file_content,
                parent_id=folder_entities[parent_relative].id,
                parent_name=folder_entities[parent_relative].name,
            )
            file_entities[relative_file] = file_entity
            entities.append(file_entity)
            relationships.append(
                ParsedRelationship(
                    source_id=folder_entities[parent_relative].id,
                    source_name=folder_entities[parent_relative].name,
                    target_id=file_entity.id,
                    target_name=file_entity.name,
                    type="CONTAINS",
                )
            )
            folder_children.setdefault(parent_relative, []).append(source_file.name)

        for folder_path, folder_entity in folder_entities.items():
            children = ", ".join(sorted(folder_children.get(folder_path, []))) or "empty"
            folder_entity.raw_code = f"Folder {folder_entity.file_path} contains: {children}"

        return entities, relationships, file_entities

    def _parse_file(
        self,
        file_path: Path,
        *,
        file_entity: ParsedEntity,
    ) -> tuple[list[ParsedEntity], list[ParsedRelationship]]:
        language_spec = self._language_spec_for_file(file_path)
        source_bytes = file_path.read_bytes()
        tree = self.parsers[language_spec.name].parse(source_bytes)
        if tree is None:
            raise RuntimeError(f"Tree-sitter failed to parse: {file_path}")

        relative_path = file_path.relative_to(self.root_path)
        file_key = str(relative_path)
        module_name = relative_path.with_suffix("").as_posix().replace("/", ".")

        drafts: dict[tuple[int, int, str], dict[str, Any]] = {}
        parent_names: dict[tuple[int, int, str], str] = {}
        owner_entity_by_key: dict[tuple[int, int, str], tuple[int, int, str]] = {}
        definition_matches = QueryCursor(language_spec.definition_query).matches(tree.root_node)

        for _, captures in definition_matches:
            if "function.node" in captures:
                entity_node = captures["function.node"][0]
                name_node = captures["function.name"][0]
                entity_type = "function"
                owner_node = captures.get("function.owner", captures["function.node"])[0]
            elif "class.node" in captures:
                entity_node = captures["class.node"][0]
                name_node = captures["class.name"][0]
                entity_type = "class"
                owner_node = None
            else:
                continue

            node_key = self._node_key(entity_node)
            parent_class = self._find_ancestor(entity_node, language_spec.class_ancestor_types)
            parent_key = self._node_key(parent_class) if parent_class is not None else None
            parent_name = self._node_text(parent_class.child_by_field_name("name"), source_bytes) if parent_class else None

            name = self._node_text(name_node, source_bytes)
            qualified_name = ".".join(part for part in (module_name, parent_name, name) if part)

            drafts[node_key] = {
                "type": entity_type,
                "name": name,
                "qualified_name": qualified_name,
                "file_path": file_key,
                "start_line": entity_node.start_point[0] + 1,
                "end_line": entity_node.end_point[0] + 1,
                "raw_code": source_bytes[entity_node.start_byte : entity_node.end_byte].decode("utf8"),
                "parent_key": parent_key,
            }
            if parent_key and parent_name:
                parent_names[parent_key] = parent_name
            if entity_type == "function" and owner_node is not None:
                owner_entity_by_key[self._node_key(owner_node)] = node_key

        entity_ids = {
            node_key: self._entity_id(
                entity_type=draft["type"],
                file_path=draft["file_path"],
                qualified_name=draft["qualified_name"],
                start_line=draft["start_line"],
                end_line=draft["end_line"],
            )
            for node_key, draft in drafts.items()
        }

        entities_by_key: dict[tuple[int, int, str], ParsedEntity] = {}
        relationships: list[ParsedRelationship] = []

        for node_key, draft in drafts.items():
            parent_key = draft["parent_key"]
            parent_id = entity_ids.get(parent_key)
            entity = ParsedEntity(
                id=entity_ids[node_key],
                type=draft["type"],
                name=draft["name"],
                qualified_name=draft["qualified_name"],
                file_path=draft["file_path"],
                start_line=draft["start_line"],
                end_line=draft["end_line"],
                raw_code=draft["raw_code"],
                parent_id=parent_id,
                parent_name=parent_names.get(parent_key),
            )
            entities_by_key[node_key] = entity

            if entity.type == "function" and parent_id is not None and entity.parent_name is not None:
                relationships.append(
                    ParsedRelationship(
                        source_id=parent_id,
                        source_name=entity.parent_name,
                        target_id=entity.id,
                        target_name=entity.name,
                        type="CONTAINS",
                    )
                )
            elif parent_id is None:
                relationships.append(
                    ParsedRelationship(
                        source_id=file_entity.id,
                        source_name=file_entity.name,
                        target_id=entity.id,
                        target_name=entity.name,
                        type="CONTAINS",
                    )
                )

        call_matches = QueryCursor(language_spec.call_query).matches(tree.root_node)
        for _, captures in call_matches:
            call_name_node = captures["call.name"][0]
            call_node = captures["call.node"][0]
            owner_entity_key = self._find_owner_entity_key(call_node, owner_entity_by_key)
            if owner_entity_key is None:
                continue

            owner_entity = entities_by_key.get(owner_entity_key)
            if owner_entity is None:
                continue

            owner_entity.calls.append(self._node_text(call_name_node, source_bytes))

        for entity in entities_by_key.values():
            entity.calls = self._dedupe_preserving_order(entity.calls)

        return list(entities_by_key.values()), relationships

    def _resolve_call_relationships(self, entities: list[ParsedEntity]) -> list[ParsedRelationship]:
        functions = [entity for entity in entities if entity.type == "function"]
        functions_by_name: dict[str, list[ParsedEntity]] = {}
        functions_by_file_and_name: dict[tuple[str, str], list[ParsedEntity]] = {}
        functions_by_parent_and_name: dict[tuple[str, str], list[ParsedEntity]] = {}

        for function in functions:
            functions_by_name.setdefault(function.name, []).append(function)
            functions_by_file_and_name.setdefault((function.file_path, function.name), []).append(function)
            if function.parent_id:
                functions_by_parent_and_name.setdefault((function.parent_id, function.name), []).append(function)

        relationships: list[ParsedRelationship] = []
        for function in functions:
            for call_name in function.calls:
                target = self._resolve_call_target(
                    source=function,
                    call_name=call_name,
                    functions_by_name=functions_by_name,
                    functions_by_file_and_name=functions_by_file_and_name,
                    functions_by_parent_and_name=functions_by_parent_and_name,
                )
                if target is None:
                    continue

                relationships.append(
                    ParsedRelationship(
                        source_id=function.id,
                        source_name=function.name,
                        target_id=target.id,
                        target_name=target.name,
                        type="CALLS",
                    )
                )

        return relationships

    def _resolve_call_target(
        self,
        *,
        source: ParsedEntity,
        call_name: str,
        functions_by_name: dict[str, list[ParsedEntity]],
        functions_by_file_and_name: dict[tuple[str, str], list[ParsedEntity]],
        functions_by_parent_and_name: dict[tuple[str, str], list[ParsedEntity]],
    ) -> ParsedEntity | None:
        if source.parent_id:
            parent_candidates = functions_by_parent_and_name.get((source.parent_id, call_name), [])
            if len(parent_candidates) == 1:
                return parent_candidates[0]

        file_candidates = functions_by_file_and_name.get((source.file_path, call_name), [])
        if len(file_candidates) == 1:
            return file_candidates[0]

        global_candidates = functions_by_name.get(call_name, [])
        if len(global_candidates) == 1:
            return global_candidates[0]

        return None

    def _language_spec_for_file(self, file_path: Path) -> LanguageSpec:
        try:
            return self.language_specs_by_suffix[file_path.suffix]
        except KeyError as exc:
            raise ValueError(f"Unsupported source file type: {file_path}") from exc

    @staticmethod
    def _entity_id(
        *,
        entity_type: str,
        file_path: str,
        qualified_name: str,
        start_line: int,
        end_line: int,
    ) -> str:
        fingerprint = f"{entity_type}|{file_path}|{qualified_name}|{start_line}|{end_line}"
        return str(uuid5(NAMESPACE_URL, fingerprint))

    @staticmethod
    def _node_key(node: Node | None) -> tuple[int, int, str] | None:
        if node is None:
            return None
        return (node.start_byte, node.end_byte, node.type)

    @staticmethod
    def _find_ancestor(node: Node, ancestor_types: str | tuple[str, ...]) -> Node | None:
        if isinstance(ancestor_types, str):
            ancestor_types = (ancestor_types,)
        current = node.parent
        while current is not None:
            if current.type in ancestor_types:
                return current
            current = current.parent
        return None

    @staticmethod
    def _find_owner_entity_key(
        node: Node,
        owner_entity_by_key: dict[tuple[int, int, str], tuple[int, int, str]],
    ) -> tuple[int, int, str] | None:
        current: Node | None = node
        while current is not None:
            owner_entity_key = owner_entity_by_key.get(CodeParser._node_key(current))
            if owner_entity_key is not None:
                return owner_entity_key
            current = current.parent
        return None

    @staticmethod
    def _node_text(node: Node | None, source_bytes: bytes) -> str:
        if node is None:
            raise ValueError("Expected a Tree-sitter node but received None.")
        return source_bytes[node.start_byte : node.end_byte].decode("utf8")

    @staticmethod
    def _dedupe_preserving_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _dedupe_relationships(
        relationships: list[ParsedRelationship],
    ) -> list[ParsedRelationship]:
        seen: set[tuple[str, str, str]] = set()
        unique_relationships: list[ParsedRelationship] = []
        for relationship in relationships:
            key = (relationship.source_id, relationship.target_id, relationship.type)
            if key in seen:
                continue
            seen.add(key)
            unique_relationships.append(relationship)
        return unique_relationships