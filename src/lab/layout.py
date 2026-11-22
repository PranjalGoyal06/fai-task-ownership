"""Classify production paths without importing or executing repository code."""
from pathlib import PurePosixPath

AUXILIARY = {"test", "tests", "testing", "docs", "doc", "documentation", "examples", "example",
             "benchmarks", "benchmark", "scripts", "ci", "build", "dist", "node_modules",
             "__pycache__", "venv", "packaging", "changelog.d"}
OTHER_STACK = {".pyx", ".pxd", ".pxi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".rs", ".go",
               ".java", ".kt", ".swift", ".scala", ".rb", ".cs", ".php", ".js", ".jsx", ".ts", ".tsx",
               ".vue", ".svelte", ".mjs", ".cjs", ".mts", ".cts", ".ex", ".exs", ".hs", ".f", ".f90",
               ".mm", ".dart", ".lua", ".zig", ".clj", ".fs", ".pl", ".erl", ".ml", ".jl"}
CONFIG = {"setup.py", "conftest.py", "noxfile.py", "tasks.py", "fabfile.py", "manage.py"}


def auxiliary(path):
    parts = PurePosixPath(path).parts
    return parts[0].lower() == "tools" or any(p.startswith(".") or p.lower() in AUXILIARY for p in parts[:-1])


def python_source(path):
    name = PurePosixPath(path).name
    return (path.endswith(".py") and not auxiliary(path) and name not in CONFIG and
            not name.startswith("test_") and not name.endswith("_test.py"))


def under(path, roots):
    return any(path == r or path.startswith(r + "/") for r in roots)


def unsupported_source(path):
    file = PurePosixPath(path)
    return file.suffix.lower() in OTHER_STACK and not auxiliary(path) and ".config." not in file.name


def discover(paths):
    roots = set()
    for path in paths:
        if python_source(path):
            parts = path.split("/")
            if "src" in parts[:-1]:
                index = parts.index("src")
                roots.add("/".join(parts[:index + 2]))
            else:
                roots.add(parts[0])
    if not roots:
        raise ValueError("No eligible production Python files found.")
    return sorted(roots)


def import_path(path, roots):
    root = max((r for r in roots if path == r or path.startswith(r + "/")), key=len)
    parent = root.rpartition("/")[0]
    return path.removeprefix(parent + "/") if parent else path
