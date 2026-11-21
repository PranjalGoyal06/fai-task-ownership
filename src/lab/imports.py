"""Project a pre-fix snapshot onto a stable, ordered set of module owners."""
from urllib.parse import quote


def import_graph(modules, snapshot, repository, assigned=(), patch_owners=()):
    nodes = [{"name": name, "assigned": name in assigned, "owner": name in patch_owners}
             for name in modules]
    connections = {}
    for source, target, path, line in snapshot["edges"]:
        pair = tuple(sorted((source, target)))
        citations = connections.setdefault(pair, [])
        citations.append({
            "source": source, "target": target, "location": f"{path}#L{line}",
            "url": f"https://github.com/{repository}/blob/{snapshot['base']}/{quote(path)}#L{line}",
        })
    return {"nodes": nodes, "connections": list(connections.values())}
