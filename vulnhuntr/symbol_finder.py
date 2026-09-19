import ast
import jedi
import pathlib
import re
import logging
from typing import List, Dict, Any, Optional
from jedi.api.classes import Name

log = logging.getLogger("vulnhuntr.symbol_finder")


class SymbolExtractor:
    def __init__(self, repo_path: str | pathlib.Path) -> None:
        self.repo_path = pathlib.Path(repo_path)
        try:
            self.project = jedi.Project(self.repo_path)
        except Exception as e:
            log.warning("Could not initialize Jedi project, falling back to AST: %s", e)
            self.project = None
        self.ignore = ['/test', '_test/', '/docs', '/example', '/site-packages', '.venv', 'virtualenv', '/dist']

    def extract(self, symbol_name: str, code_line: str, filtered_files: List) -> Optional[Dict[str, Any]]:
        """
        Extracts the definition of a symbol from the repository.
        Uses a resilient multi-tier approach:
          1. Jedi Script search on files matching code_line
          2. Jedi Project search across repo
          3. Jedi all-names inspection
          4. Native Python AST definition search fallback
          5. Regex definition search fallback
        """
        symbol_name = symbol_name.strip()
        symbol_parts = symbol_name.split('.')

        # Find matching files for the code line
        matching_files = [file for file in filtered_files if self._search_string_in_file(file, code_line)]
        
        # 1. Jedi Script Search
        if self.project and matching_files:
            try:
                scripts = [jedi.Script(path=file, project=self.project) for file in matching_files]
                match = self.file_search(symbol_name, scripts)
                if match:
                    return match
            except Exception as e:
                log.debug("Jedi file_search failed: %s", e)

        # 2. Jedi Project Search
        if self.project:
            try:
                match = self.project_search(symbol_name)
                if match:
                    return match
            except Exception as e:
                log.debug("Jedi project_search failed: %s", e)

        # 3. Jedi All Names Search
        if self.project and matching_files:
            try:
                scripts = [jedi.Script(path=file, project=self.project) for file in matching_files]
                match = self.all_names_search(symbol_name, symbol_parts, scripts, code_line)
                if match:
                    return match
            except Exception as e:
                log.debug("Jedi all_names_search failed: %s", e)

        # 4. AST-based Search Fallback across all project files
        ast_match = self.ast_search(symbol_name, filtered_files)
        if ast_match:
            return ast_match

        # 5. Fast Regex Definition Fallback
        regex_match = self.regex_search(symbol_name, filtered_files)
        if regex_match:
            return regex_match

        log.debug("No match found for symbol: %s", symbol_name)
        return None

    def ast_search(self, symbol_name: str, files: List[pathlib.Path]) -> Optional[Dict[str, Any]]:
        """Searches for symbol definition using Python's standard ast module."""
        parts = symbol_name.split('.')
        target_name = parts[-1]
        parent_class = parts[-2] if len(parts) > 1 else None

        for file_path in files:
            p = pathlib.Path(file_path)
            if self._should_exclude(str(p)):
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content, filename=str(p))
                lines = content.splitlines(keepends=True)

                for node in ast.walk(tree):
                    # Class definition match
                    if isinstance(node, ast.ClassDef) and node.name == target_name:
                        start_line = node.lineno - 1
                        end_line = getattr(node, "end_lineno", len(lines))
                        source = "".join(lines[start_line:end_line])
                        return {
                            "name": target_name,
                            "context_name_requested": symbol_name,
                            "file_path": str(p),
                            "source": source
                        }

                    # Function definition match
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == target_name:
                            # If a parent class was specified, check if this function is inside that class
                            if parent_class:
                                # Look for parent class in AST
                                for parent_node in ast.walk(tree):
                                    if isinstance(parent_node, ast.ClassDef) and parent_node.name == parent_class:
                                        if any(getattr(child, "name", None) == target_name for child in parent_node.body):
                                            start_line = node.lineno - 1
                                            end_line = getattr(node, "end_lineno", len(lines))
                                            source = "".join(lines[start_line:end_line])
                                            return {
                                                "name": target_name,
                                                "context_name_requested": symbol_name,
                                                "file_path": str(p),
                                                "source": source
                                            }
                            else:
                                start_line = node.lineno - 1
                                end_line = getattr(node, "end_lineno", len(lines))
                                source = "".join(lines[start_line:end_line])
                                return {
                                    "name": target_name,
                                    "context_name_requested": symbol_name,
                                    "file_path": str(p),
                                    "source": source
                                }
            except Exception:
                continue

        return None

    def regex_search(self, symbol_name: str, files: List[pathlib.Path]) -> Optional[Dict[str, Any]]:
        """Fallback regex search for class or function definitions."""
        target = symbol_name.split('.')[-1]
        pattern = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(target)}\b", re.MULTILINE)

        for file_path in files:
            p = pathlib.Path(file_path)
            if self._should_exclude(str(p)):
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                match = pattern.search(content)
                if match:
                    # Capture up to 60 lines from match position
                    start_pos = match.start()
                    lines = content[start_pos:].splitlines()[:60]
                    return {
                        "name": target,
                        "context_name_requested": symbol_name,
                        "file_path": str(p),
                        "source": "\n".join(lines)
                    }
            except Exception:
                continue

        return None

    def file_search(self, symbol_name: str, scripts: List) -> Optional[Dict[str, Any]]:
        for script in scripts:
            try:
                res = script.search(symbol_name)
            except Exception:
                continue

            for name in res:
                if self._should_exclude(str(name.module_path)):
                    continue

                if name.type == 'statement' and symbol_name in name.description:
                    return self._create_match_obj(name, symbol_name)

                elif name.type in ['function', 'class']:
                    if symbol_name == name.name or symbol_name.endswith(f".{name.name}") or symbol_name in name.description:
                        try:
                            inferred = name.infer()
                            for inf in inferred:
                                return self._create_match_obj(inf, symbol_name)
                        except Exception:
                            return self._create_match_obj(name, symbol_name)

                elif name.type == 'instance':
                    try:
                        inferred = name.infer()
                        for inf in inferred:
                            if inf.type in ['class', 'function']:
                                return self._create_match_obj(inf, symbol_name)
                            elif inf.type == 'instance':
                                go = name.goto()
                                if go:
                                    return self._create_match_obj(go[0], symbol_name)
                    except Exception:
                        pass

                elif name.type == 'module' and name.name == symbol_name:
                    match = self._create_match_obj(name, symbol_name)
                    if 'import ' in match.get('source', ''):
                        try:
                            loc = name.goto()
                            if loc:
                                return self._create_match_obj(loc[0], symbol_name)
                        except Exception:
                            pass
                    return match

        return None

    def project_search(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        if not self.project:
            return None
        try:
            res = list(self.project.search(symbol_name))
        except Exception:
            return None

        for name in res:
            if self._should_exclude(str(name.module_path)):
                continue

            if name.type == 'statement' and symbol_name in name.description:
                return self._create_match_obj(name, symbol_name)

            elif name.type in ['function', 'class']:
                if symbol_name == name.name or symbol_name.endswith(f".{name.name}") or symbol_name in name.description:
                    try:
                        inferred = name.infer()
                        for inf in inferred:
                            return self._create_match_obj(inf, symbol_name)
                    except Exception:
                        return self._create_match_obj(name, symbol_name)

            elif name.type == 'instance':
                try:
                    inferred = name.infer()
                    for inf in inferred:
                        if inf.type in ['instance', 'class', 'function']:
                            return self._create_match_obj(inf, symbol_name)
                except Exception:
                    pass

            elif name.type == 'module' and name.name == symbol_name:
                match = self._create_match_obj(name, symbol_name)
                if 'import ' in match.get('source', ''):
                    try:
                        loc = name.goto()
                        if loc:
                            return self._create_match_obj(loc[0], symbol_name)
                    except Exception:
                        pass
                return match

        return None

    def all_names_search(self, symbol_name: str, symbol_parts: List, scripts: List, code_line: str) -> Optional[Dict[str, Any]]:
        for script in scripts:
            try:
                names = script.get_names(all_scopes=True, definitions=True, references=True)
            except Exception:
                continue
            for name in names:
                if name.type in ['function', 'class', 'instance']:
                    if name.full_name and name.full_name.endswith(symbol_name):
                        try:
                            for inf in name.infer():
                                return self._create_match_obj(inf, symbol_name)
                        except Exception:
                            return self._create_match_obj(name, symbol_name)
                    elif name.name == symbol_parts[-1]:
                        try:
                            for inf in name.infer():
                                return self._create_match_obj(inf, symbol_name)
                        except Exception:
                            return self._create_match_obj(name, symbol_name)

        return None

    def _should_exclude(self, module_path: str) -> bool:
        norm = module_path.lower().replace('\\', '/')
        return any(x in norm for x in self.ignore)

    def _search_string_in_file(self, file_path: pathlib.Path, string: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_text = file.read()
                clean_target = re.sub(r'[\s"\'\r\n\t]', '', string)
                if not clean_target:
                    return False
                clean_file = re.sub(r'[\s"\'\r\n\t]', '', file_text)
                return clean_target in clean_file
        except Exception:
            return False

    def _get_definition_source(self, file_path: pathlib.Path, start, end) -> str:
        try:
            with file_path.open(encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if not start and not end:
                    return ''.join(lines)
                definition = lines[start[0]-1:end[0]]
                end_len_diff = len(definition[-1]) - end[1]
                s = ''.join(definition)[start[1]:-end_len_diff] if end_len_diff > 0 else ''.join(definition)[start[1]:]
                return s if s else 'None'
        except Exception:
            return 'None'

    def _create_match_obj(self, name: Name, symbol_name: str) -> Dict[str, Any]:
        module_path = str(getattr(name, 'module_path', 'None'))
        if '/third_party/' in module_path or module_path == 'None':
            full_name = getattr(name, 'full_name', symbol_name)
            source = f'External module or third-party package: {full_name}. Analyze based on standard library/package specifications.'
        else:
            try:
                start = name.get_definition_start_position()
                end = name.get_definition_end_position()
                source = self._get_definition_source(name.module_path, start, end)
            except Exception:
                source = str(getattr(name, 'description', 'None'))

        return {
            'name': getattr(name, 'name', symbol_name),
            'context_name_requested': symbol_name,
            'file_path': module_path,
            'source': source
        }