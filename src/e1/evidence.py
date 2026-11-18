"""Extract module-local evidence without opening outcome records."""
import argparse
import ast
import functools
import json
import re
import subprocess
import warnings
from pathlib import Path

from src.e1.dataset import MODULES, dump, module

WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]{2,}")
STOP = set("the and for from this that with are not self def class django return import None".lower().split())
DOCS = ("docs/ref/models/querysets.txt", "docs/ref/databases.txt",
        "docs/ref/contrib/admin/index.txt", "docs/ref/contrib/index.txt",
        "docs/ref/checks.txt", "docs/ref/forms/api.txt", "docs/ref/templates/api.txt",
        "docs/ref/utils.txt", "docs/ref/request-response.txt")


def words(text):
    return {s.lower() for s in WORD.findall(text)} - STOP


class Snapshot:
    def __init__(self, repository):
        self.git = ["git", "--git-dir=" + str(repository)]
        self.batch = subprocess.Popen(self.git + ["cat-file", "--batch"],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def close(self):
        self.batch.stdin.close()
        self.batch.wait()
        self.batch.stdout.close()

    def blob(self, ref):
        self.batch.stdin.write((ref + "\n").encode())
        self.batch.stdin.flush()
        header = self.batch.stdout.readline().split()
        if header[-1] == b"missing":
            raise ValueError(f"Missing git object {ref}")
        raw = self.batch.stdout.read(int(header[-1]))
        self.batch.stdout.read(1)
        return raw.decode("utf-8", errors="replace")

    @functools.lru_cache(maxsize=16000)
    def parse(self, oid):
        source = self.blob(oid)
        terms = words(source)
        names, imports = {}, []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(source)
        except SyntaxError:
            return terms, names, imports, 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names[node.name] = node.lineno
            if isinstance(node, ast.Import):
                imports.extend((a.name, node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.append((node.module, node.lineno))
        return terms, names, imports, 0

    def evidence(self, issue):
        ref, text = issue["base"], issue["text"]
        terms, exact = words(text), set(WORD.findall(text))
        fields = [[0., 0., 0., 0.] for _ in MODULES]
        citations, edges = {}, {}
        failures = 0
        tree = subprocess.check_output(self.git + ["ls-tree", "-r", ref, "django"]).decode()
        for line in tree.splitlines():
            meta, path = line.split("\t", 1)
            i = module(path)
            if i is None:
                continue
            terms_local, names, imports, failure = self.parse(meta.split()[2])
            failures += failure
            hits = sorted(exact.intersection(names) - {"__init__", "get", "set"})
            path_hit = path in text or path[:-3].replace("/", ".") in text
            lexical = len(terms & terms_local) / max(1, len(terms | terms_local)) ** 0.5
            f = fields[i]
            f[0] = max(f[0], lexical)
            f[1] = max(f[1], len(hits))
            f[2] += int(path_hit)
            prefix = MODULES[i]
            f[3] = float(prefix != "other" and prefix.replace("/", ".") in text.lower())
            if path_hit or hits:
                citations.setdefault(i, []).append([path, hits[:3], [names[n] for n in hits[:3]]])
            for target, lineno in imports:
                j = module(target.replace(".", "/") + "/__init__.py")
                if j is not None and i != j:
                    edges.setdefault((i, j), [path, lineno, target])
        return dict(id=issue["id"], base=ref, features=fields, citations=citations,
                    edges=[[i, j, *v] for (i, j), v in sorted(edges.items())],
                    syntax_failures=failures)

    def descriptions(self, ref):
        return [self.blob(ref + ":" + path) for path in DOCS]


def extract(directory, repository):
    issues = json.loads((directory / "issues.json").read_text())
    cache = directory / "evidence"
    cache.mkdir(exist_ok=True)
    source = Snapshot(repository)
    try:
        cutoff = next(r["base"] for r in reversed(issues) if r["split"] == "train")
        dump(directory / "descriptions.json", dict(base=cutoff, docs=source.descriptions(cutoff)))
        for r in issues:
            path = cache / (r["id"] + ".json")
            if not path.exists():
                dump(path, source.evidence(r))
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    extract(args.directory, args.repository)
