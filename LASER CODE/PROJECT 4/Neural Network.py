import matplotlib.pyplot as plt
import networkx as nx

# ============ Font Configuration ============
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def draw_network(layers, colors, title):
    pos = {}
    x_coords = {layer: i for i, layer in enumerate(layers.keys())}

    for layer, nodes in layers.items():
        y_spacing = 1.0 / (len(nodes) + 1)
        for i, node in enumerate(nodes):
            pos[node] = (x_coords[layer], 1 - (i + 1) * y_spacing)

    G = nx.DiGraph()
    layer_names = list(layers.keys())
    for i in range(len(layer_names) - 1):
        for u in layers[layer_names[i]]:
            for v in layers[layer_names[i + 1]]:
                G.add_edge(u, v)

    node_color_map = {node: colors[layer] for layer, nodes in layers.items() for node in nodes}
    plt.figure(figsize=(12, 7))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1800,
        node_color=[node_color_map[n] for n in G.nodes()]
    )

    nx.draw_networkx_labels(
        G, pos,
        labels={node: node for node in G.nodes()},
        font_size=11, font_weight="bold", font_color='black', font_family='DejaVu Sans'
    )

    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowstyle="-|>", arrowsize=20,
        width=1.5, connectionstyle="arc3,rad=0.0"
    )

    for layer, nodes in layers.items():
        x = x_coords[layer]
        y = 1.05
        title_text = f"{layer} ({len(nodes)})" if "Hidden Layer" in layer else layer
        plt.text(x, y, title_text,
                 fontsize=14, fontweight='bold',
                 ha='center', va='bottom',
                 color=colors[layer],
                 fontfamily='DejaVu Sans')

    plt.title(title, fontsize=18, fontweight="bold", fontfamily='DejaVu Sans')
    plt.axis("off")
    plt.show()

# ============ Colors ============
colors = {
    "Input": "SteelBlue",
    "Hidden Layer 1": "green",
    "Hidden Layer 2": "red",
    "Hidden Layer 3": "purple",
    "Output": "orange"
}

# ============ Define Layers ============
layers_esr = {
    "Input": ["Laser\nPower", "Scanning\nSpeed ", "Number of\npasses", "Laser\nFrequency"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(6)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(4)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(3)],
    "Output": ["ESR"]
}

layers_mrq = {
    "Input": ["Laser\nPower", "Scanning\nSpeed ", "Number of\npasses", "Laser\nFrequency"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(8)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(5)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(4)],
    "Output": ["MRQ"]
}

layers_haz= {
    "Input": ["Laser\nPower", "Scanning\nSpeed ", "Number of\npasses", "Laser\nFrequency"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(6)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(4)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(2)],
    "Output": ["HAZ"]
}


# ============ Draw All 3 Networks ============
draw_network(layers_esr, colors, "Neural Network Architecture: External Surface Roughness")
draw_network(layers_mrq, colors, "Neural Network Architecture: Material Removal Quantity")
draw_network(layers_haz, colors, "Neural Network Architecture: Heat Affected Zone")
