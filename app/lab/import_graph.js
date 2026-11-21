// The node order alone determines positions. Selecting a mark never relays out the graph.
export default function render({parentElement, data}) {
    const root = parentElement.querySelector(".imports");
    root.replaceChildren();
    const svgNS = "http://www.w3.org/2000/svg";
    function svg(tag, attrs, parent) {
        const element = document.createElementNS(svgNS, tag);
        for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
        parent.append(element);
        return element;
    }
    const diagram = svg("svg", {viewBox: "0 0 610 320", role: "group",
        "aria-label": "Directed module import graph"}, root);
    const defs = svg("defs", {}, diagram);
    for (const [id, color] of [["idle", "#80949e"], ["active", "#087778"]]) {
        const marker = svg("marker", {id, viewBox: "0 0 10 10", refX: 9, refY: 5,
            markerWidth: 8, markerHeight: 8, markerUnits: "userSpaceOnUse", orient: "auto-start-reverse"}, defs);
        svg("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: color}, marker);
    }
    const detail = document.createElement("div");
    detail.setAttribute("aria-live", "polite");
    root.append(detail);
    const positions = data.nodes.map((_, i) => {
        const angle = 2 * Math.PI * i / data.nodes.length - Math.PI / 2;
        return {x: 305 + 195 * Math.cos(angle), y: 157 + 121 * Math.sin(angle)};
    });
    let selected = null;
    function button(group, label, action) {
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-label", label);
        svg("title", {}, group).textContent = label;
        group.onclick = action;
        group.onkeydown = event => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                action();
            }
        };
    }
    const edges = data.connections.map(citations => {
        const {source, target} = citations[0];
        const both = citations.some(c => c.source === target && c.target === source);
        const a = positions[source], b = positions[target];
        const length = Math.hypot(b.x - a.x, b.y - a.y);
        const dx = (b.x - a.x) / length, dy = (b.y - a.y) / length;
        const d = `M ${a.x + dx * 13} ${a.y + dy * 13} L ${b.x - dx * 13} ${b.y - dy * 13}`;
        const group = svg("g", {class: "connection"}, diagram);
        svg("path", {d, stroke: "transparent", "stroke-width": 10, fill: "none"}, group);
        const line = svg("path", {d, fill: "none"}, group);
        const edge = {group, line, source, target, both, citations};
        const verb = both ? " exchanges imports with " : " imports ";
        button(group, data.nodes[source].name + verb + data.nodes[target].name, () => select(edge));
        return edge;
    });
    const nodes = data.nodes.map((node, i) => {
        const {x, y} = positions[i];
        const group = svg("g", {class: "module", transform: `translate(${x},${y})`}, diagram);
        const ring = svg("circle", {r: 13, fill: "none", stroke: "#087778", "stroke-width": 2}, group);
        const attrs = {fill: node.assigned ? "#097574" : "#f3f7f8", stroke: "#164e59", "stroke-width": 2};
        if (node.owner) svg("path", {...attrs, d: "M0 -10 L10 0 L0 10 L-10 0 Z"}, group);
        else svg("circle", {...attrs, r: 8}, group);
        const side = x < 275 ? -1 : x > 335 ? 1 : 0;
        const label = svg("text", {x: side * 14, y: side ? 5 : (y < 160 ? -18 : 23),
            "text-anchor": side < 0 ? "end" : side > 0 ? "start" : "middle"}, group);
        label.textContent = node.name;
        const item = {group, ring, index: i};
        button(group, "Highlight imports for " + node.name, () => select(item));
        return item;
    });
    function select(item) {
        selected = selected === item ? null : item;
        const focus = selected?.index;
        for (const edge of edges) {
            const active = edge === selected || focus === edge.source || focus === edge.target;
            const marker = active ? "url(#active)" : "url(#idle)";
            edge.group.style.opacity = !selected || active ? 1 : .17;
            edge.line.setAttribute("stroke", active ? "#087778" : "#80949e");
            edge.line.setAttribute("stroke-width", active ? 2.5 : 1.25);
            edge.line.setAttribute("marker-end", marker);
            edge.line.setAttribute("marker-start", edge.both ? marker : "none");
            edge.group.setAttribute("aria-pressed", String(edge === selected));
        }
        for (const node of nodes) {
            node.ring.style.opacity = node === selected ? 1 : 0;
            node.group.setAttribute("aria-pressed", String(node === selected));
        }
        detail.replaceChildren();
        if (selected?.citations) {
            for (const c of selected.citations) {
                const row = document.createElement("p");
                row.append(data.nodes[c.source].name + " → " + data.nodes[c.target].name + " · ");
                const link = document.createElement("a");
                link.href = c.url;
                link.textContent = c.location;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                row.append(link);
                detail.append(row);
            }
        } else if (selected) {
            const outgoing = edges.filter(e => e.citations.some(c => c.source === focus)).length;
            const incoming = edges.filter(e => e.citations.some(c => c.target === focus)).length;
            detail.textContent = `${data.nodes[focus].name} · ${outgoing} outgoing and ${incoming} incoming connections`;
        } else {
            detail.textContent = "Select a module to highlight its imports, or a connection to open its source lines.";
        }
    }
    select(null);
}
