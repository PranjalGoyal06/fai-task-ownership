"""Interactive pre-fix import graph shared by setup and replay."""
import streamlit as st

from src.lab.preset import ROOT
from src.lab.imports import import_graph

ASSETS = ROOT / "app/lab"
render_graph = st.components.v2.component(
    "ownership_import_graph",
    html='<div class="imports"></div>',
    css=(ASSETS / "import_graph.css").read_text(),
    js=(ASSETS / "import_graph.js").read_text(),
)


def graph(snapshot, modules, repository, frame=None):
    assigned = [] if frame is None else frame.index[frame.assigned]
    owners = [] if frame is None else frame.index[frame.patch_owner]
    data = import_graph(modules, snapshot, repository, assigned, owners)
    st.subheader("Project import graph")
    st.caption("An arrow points to the imported module. A double arrow means both modules import each other.")
    if frame is not None:
        st.caption("Teal fill = assigned by the selected policy · diamond = changed in the patch.")
    render_graph(data=data, key="historical-import-graph")
    st.caption(f"Pre-fix revision `{snapshot['base'][:12]}` · {len(data['nodes'])} modules · "
               f"{len(snapshot['edges'])} directed imports")
