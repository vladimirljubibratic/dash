import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import logging
import io
import base64
import tempfile
import os

# Set up logging for production with INFO level to reduce verbosity
logging.basicConfig(level=logging.WARNING)

# Example data source to be used initially
initial_data = {
    'Name': ['HQ','Corporate', 'Finance', 'HR', 'Recruitment', 'Training', 'IT', 'Development', 'Operations'],
    'Parent Business': ['','HQ', 'Corporate', 'Corporate', 'HR', 'HR', 'Corporate', 'IT', 'Corporate']
}
df = pd.DataFrame(initial_data)

# Global variable to track the maximum depth of the organization
ROOT_NODE = None
ORGANIZATION_LEVEL_DEPTH = 0

# Create a directed graph from the business hierarchy
def build_graph():
    """
    Builds a directed graph representing the business unit hierarchy.
    Returns:
        G (nx.DiGraph): Directed graph representing the business units.
    """
    G = nx.DiGraph()
    global ORGANIZATION_LEVEL_DEPTH
    global ROOT_NODE
    max_depth = 0

    # Add nodes and edges based on the data
    for _, row in df.iterrows():
        child = row['Name']
        parent = row['Parent Business']
        G.add_node(child)
        if parent:
            G.add_node(parent)
            G.add_edge(parent, child)

    # Identify root nodes (nodes without any parents)
    root_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
    if root_nodes:
        ROOT_NODE = root_nodes[0]
    if not root_nodes:
        logging.warning("No root node found in the graph.")
        return G

    # Calculate the maximum depth of the organization
    for root_node in root_nodes:
        for node in G.nodes:
            try:
                depth = len(nx.shortest_path(G, source=root_node, target=node)) - 1
                if depth > max_depth:
                    max_depth = depth
            except nx.NetworkXNoPath:
                continue
    ORGANIZATION_LEVEL_DEPTH = max_depth + 1
    logging.info(f"Organization maximum depth: {ORGANIZATION_LEVEL_DEPTH}")
    return G

# Function to create the figure based on the current root
def create_figure(G, current_root=None, expand_children=False):
    """
    Creates a Plotly figure representing the hierarchy graph.
    Args:
        G (nx.DiGraph): The full business unit graph.
        current_root (str): The current root node for visualization.
        expand_children (bool): Whether to expand all children for the current root.
    Returns:
        fig (go.Figure): Plotly figure representing the graph.
    """
    # Create a copy of the graph to work with
    subG = G.copy()

    # Remove child nodes for nodes with more than 20 children unless expanding
    nodes_to_remove = []
    for node in list(subG.nodes):
        if len(list(subG.successors(node))) > 20 and not (current_root == node and expand_children):
            subG.nodes[node]['is_large'] = True
            nodes_to_remove.extend(subG.successors(node))
        else:
            subG.nodes[node]['is_large'] = False
    if current_root and subG.nodes[current_root].get('is_large', False):
        # If current_root is a node with removed children, expand them
        nodes_to_remove = []
    subG.remove_nodes_from(nodes_to_remove)

    # Include only nodes connected to the current root for visualization
    if current_root:
        if current_root not in subG:
            logging.warning(f"{current_root} is not found in the graph. Reverting to initial view.")
            current_root = None
        else:
            visible_nodes = [current_root]
            level = [current_root]
            while level:
                next_level = []
                for node in sorted(level):
                    sorted_children = sorted(subG.successors(node))
                    visible_nodes.extend(sorted_children)
                    next_level.extend(sorted_children)
                level = sorted(next_level)
            subG = subG.subgraph(visible_nodes)

    # Calculate the maximum depth of the subgraph
    subgraph_root_nodes = [node for node in subG.nodes if subG.in_degree(node) == 0]
    subgraph_max_depth = 0
    for root_node in subgraph_root_nodes:
        for node in subG.nodes:
            try:
                depth = len(nx.shortest_path(subG, source=root_node, target=node)) - 1
                if depth > subgraph_max_depth:
                    subgraph_max_depth = depth
            except nx.NetworkXNoPath:
                continue
    subgraph_max_depth += 1
    logging.info(f"Subgraph maximum depth: {subgraph_max_depth}")

    # Generate positions using the dot layout
    pos = nx.nx_pydot.pydot_layout(subG, prog="dot")

    # Scale positions to increase spacing for better visualization
    for key in pos:
        pos[key] = (pos[key][0] * 2, pos[key][1] * 2)

    # Create edge traces for Plotly visualization
    edge_x = []
    edge_y = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Create node traces for Plotly visualization
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_sizes = []
    node_colors = []
    node_textpositions = []
    node_textsize = []

    # Define node sizes and colors
    base_size = 20
    size_decrement = 2

    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Truncate long labels into multiple lines if needed
        if node == current_root or node == "":
            truncated_label = node  # Do not truncate the initial root or current root label
        elif len(node) > 12:
            truncated_label = node[:4] + '<br>' + node[4:8] + '<br>' + node[8:]  # Split into three lines
        elif len(node) > 6:
            truncated_label = node[:7] + '<br>' + node[7:]  # Split into two lines
        else:
            truncated_label = node  # Keep the original label
        node_text.append(truncated_label)
        # Add hover text with the full node name and number of children
        num_children = len(list(G.successors(node))) if node in G else 0
        hover_text = f"{node} ({num_children} children)" if num_children > 0 else node
        node_hovertext.append(hover_text)
        # Use a different color for nodes with many children
        is_large = subG.nodes[node].get("is_large", False)
        node_colors.append("royalblue" if is_large else "skyblue")
        # Adjust node size based on level
        try:
            level = len(nx.shortest_path(G, source=current_root, target=node)) if node in G else 0
        except nx.NetworkXNoPath:
            level = 0
        node_sizes.append(max(base_size - level * size_decrement, 6))
        node_textpositions.append("middle center" if level < 2 else "top center" if level == subgraph_max_depth else "bottom center")
        node_textsize.append(int(12 if level < 2 else 8 if level == subgraph_max_depth else 10))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,  # Let Plotly handle dynamic text wrapping
        hovertext=node_hovertext,
        marker=dict(size=node_sizes, color=node_colors, line_width=1),
        hoverinfo="text",
        textposition=node_textpositions,  # Adjust text position to be managed by level
        textfont=dict(size=node_textsize),
    )

    # Create the Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            uirevision='constant',
            title="Business Unit Hierarchy Tree" if not current_root else f"{current_root} - Business Unit Subtree",
            title_font=dict(size=13),
            title_x=0,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=30),
            autosize=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    return fig

# Set up the Dash app for production
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for production deployment
app.title = "Business Unit Hierarchy"

# App layout with a hidden store for the current root
app.layout = html.Div([
    html.Div([
        html.H4("Hierarchical Business Unit Structure", style={'display': 'inline-block', 'margin-right': '10px'}),
        html.Button("Download HTML", id="export-html-btn", style={'display': 'inline-block', 'margin-right': '10px'}),
        dcc.Upload(html.Button('Upload Data', id='upload-data-btn', style={'display': 'inline-block', 'margin-right': '10px'}), id='upload-data', multiple=False),
    ], style={'display': 'flex', 'align-items': 'center'}),
    dcc.Download(id="download-html"),
    dcc.Store(id="current-root", data=None),
    dcc.Graph(id="business-unit-hierarchical-tree", config={'responsive': True}),
])

# Combined callback to update the graph based on node clicks or data upload
@app.callback(
    [Output("business-unit-hierarchical-tree", "figure"),
     Output("current-root", "data"),
     Output("download-html", "data")],
    [Input("business-unit-hierarchical-tree", "clickData"),
     Input("upload-data", "contents"),
     Input("export-html-btn", "n_clicks")],
    [State("upload-data", "filename"),
     State("current-root", "data")]
)
def update_figure(clickData, contents, n_clicks_export, filename, current_root):
    """
    Combined callback function to handle both node clicks, data upload, and HTML export, updating the graph accordingly.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    G = build_graph()

    # Handle data upload
    if triggered_id == 'upload-data' and contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Use Pandas to read the uploaded Excel file
            uploaded_df = pd.read_excel(io.BytesIO(decoded))
            if 'Name' in uploaded_df.columns and 'Parent Business' in uploaded_df.columns:
                global df
                df = uploaded_df.fillna('')
                G = build_graph()
                return create_figure(G), None, None
            else:
                logging.warning("Uploaded file does not contain required columns 'Name' and 'Parent Business'.")
        except Exception as e:
            logging.error(f"There was an error processing the uploaded file: {e}")

    # Handle node click
    elif triggered_id == 'business-unit-hierarchical-tree' and clickData:
        if "points" in clickData and len(clickData["points"]) > 0 and "hovertext" in clickData["points"][0]:
            clicked_node = clickData["points"][0]["hovertext"].split(" (")[0]
            logging.debug(f"Clicked node: {clicked_node}")

            if clicked_node in G:
                # Check if clicked node has children
                has_children = len(list(G.successors(clicked_node))) > 0
                if not has_children:
                    return dash.no_update

                # Check if clicked node is root in the visualization
                is_visualization_root = (current_root == clicked_node)

                if is_visualization_root:
                    # If clicked node is root, return parent of clicked node if available
                    parent_node = next((n for n in G.predecessors(clicked_node)), None)
                    if parent_node:
                        return create_figure(G, current_root=parent_node), parent_node, None
                else:
                    # If clicked node is not root, re-render with clicked node as new root
                    return create_figure(G, current_root=clicked_node), clicked_node, None

    # Handle export button click
    elif triggered_id == 'export-html-btn' and n_clicks_export is not None:
        if current_root:
            fig = create_figure(G, current_root)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
                fig.write_html(tmpfile.name)
                return dash.no_update, current_root, dcc.send_file(tmpfile.name)

    # Render the main view with the full hierarchy at the initial load or if an invalid click occurs
    return create_figure(G, current_root if current_root else ROOT_NODE), ROOT_NODE, None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
