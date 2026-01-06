"""
Visualizer
Tools for visualizing the graph network structure.
"""

import json
import os
from typing import Dict, List, Optional, Set
import numpy as np

from core.graph_network import GraphNetwork, DomainType


class GraphVisualizer:
    """
    Visualizes graph network structure and exports to various formats.
    Implements Phase 3: Visualization capabilities
    """

    def __init__(self, graph: GraphNetwork):
        self.graph = graph

    def export_to_json(self, filepath: str, max_nodes: int = 1000):
        """
        Export graph to JSON format for D3.js visualization.

        Args:
            filepath: Output file path
            max_nodes: Maximum number of nodes to export
        """
        print(f"Exporting graph to JSON: {filepath}")

        # Select top nodes by frequency
        nodes_list = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.frequency,
            reverse=True
        )[:max_nodes]

        node_ids = {node.id for node in nodes_list}

        # Create nodes array
        nodes = []
        for i, node in enumerate(nodes_list):
            nodes.append({
                'id': node.id,
                'label': node.value,
                'type': node.node_type.value,
                'domain': node.domain.value,
                'frequency': node.frequency,
                'index': i
            })

        # Create edges array (only for included nodes)
        edges = []
        for edge in self.graph.edges.values():
            if edge.source in node_ids and edge.target in node_ids:
                edges.append({
                    'source': edge.source,
                    'target': edge.target,
                    'weight': float(edge.weight),
                    'type': edge.edge_type
                })

        # Create final structure
        data = {
            'nodes': nodes,
            'links': edges,
            'metadata': {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'exported_nodes': len(nodes),
                'exported_edges': len(edges)
            }
        }

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(nodes)} nodes and {len(edges)} edges")
        return data

    def export_to_graphml(self, filepath: str, max_nodes: int = 1000):
        """
        Export graph to GraphML format for Gephi/Cytoscape.

        Args:
            filepath: Output file path
            max_nodes: Maximum number of nodes to export
        """
        print(f"Exporting graph to GraphML: {filepath}")

        # Select top nodes
        nodes_list = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.frequency,
            reverse=True
        )[:max_nodes]

        node_ids = {node.id for node in nodes_list}

        # Build GraphML XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        lines.append('  <key id="label" for="node" attr.name="label" attr.type="string"/>')
        lines.append('  <key id="frequency" for="node" attr.name="frequency" attr.type="int"/>')
        lines.append('  <key id="domain" for="node" attr.name="domain" attr.type="string"/>')
        lines.append('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>')
        lines.append('  <graph id="G" edgedefault="directed">')

        # Add nodes
        for node in nodes_list:
            lines.append(f'    <node id="{node.id}">')
            lines.append(f'      <data key="label">{self._escape_xml(node.value)}</data>')
            lines.append(f'      <data key="frequency">{node.frequency}</data>')
            lines.append(f'      <data key="domain">{node.domain.value}</data>')
            lines.append('    </node>')

        # Add edges
        edge_count = 0
        for edge in self.graph.edges.values():
            if edge.source in node_ids and edge.target in node_ids:
                lines.append(f'    <edge id="e{edge_count}" source="{edge.source}" target="{edge.target}">')
                lines.append(f'      <data key="weight">{edge.weight}</data>')
                lines.append('    </edge>')
                edge_count += 1

        lines.append('  </graph>')
        lines.append('</graphml>')

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Exported {len(nodes_list)} nodes and {edge_count} edges")

    def export_to_dot(self, filepath: str, max_nodes: int = 100):
        """
        Export graph to DOT format for Graphviz.

        Args:
            filepath: Output file path
            max_nodes: Maximum number of nodes to export
        """
        print(f"Exporting graph to DOT: {filepath}")

        # Select top nodes
        nodes_list = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.frequency,
            reverse=True
        )[:max_nodes]

        node_ids = {node.id for node in nodes_list}

        # Build DOT format
        lines = ['digraph G {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=rounded];')

        # Domain colors
        domain_colors = {
            'text': 'lightblue',
            'math': 'lightgreen',
            'code': 'lightyellow',
            'general': 'lightgray'
        }

        # Add nodes
        for node in nodes_list:
            color = domain_colors.get(node.domain.value, 'white')
            label = self._escape_dot(node.value)
            lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}", style="filled"];')

        # Add edges
        for edge in self.graph.edges.values():
            if edge.source in node_ids and edge.target in node_ids:
                width = max(0.5, edge.weight * 3)
                lines.append(f'  "{edge.source}" -> "{edge.target}" [penwidth={width:.1f}];')

        lines.append('}')

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Exported {len(nodes_list)} nodes")

    def create_html_visualization(self, output_file: str = "graph_visualization.html", max_nodes: int = 500):
        """
        Create a standalone HTML file with interactive D3.js visualization.

        Args:
            output_file: Output HTML file path
            max_nodes: Maximum number of nodes to visualize
        """
        print(f"Creating HTML visualization: {output_file}")

        # Export graph data
        json_file = output_file.replace('.html', '_data.json')
        graph_data = self.export_to_json(json_file, max_nodes)

        # Create HTML with embedded D3.js visualization
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Graph Network Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
        }
        #graph {
            width: 100vw;
            height: 100vh;
        }
        .node {
            stroke: #fff;
            stroke-width: 1.5px;
            cursor: pointer;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .node-label {
            font-size: 10px;
            pointer-events: none;
            fill: #fff;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 5px;
            max-width: 300px;
        }
        .stat {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>Graph Network Visualization</h3>
        <div class="stat">Nodes: <span id="node-count">0</span></div>
        <div class="stat">Edges: <span id="edge-count">0</span></div>
        <div class="stat">Selected: <span id="selected">None</span></div>
    </div>
    <svg id="graph"></svg>
    
    <script>
        const graphData = """ + json.dumps(graph_data) + """;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height);
        
        // Color scale for domains
        const colorScale = d3.scaleOrdinal()
            .domain(["text", "math", "code", "general"])
            .range(["#4285f4", "#34a853", "#fbbc04", "#ea4335"]);
        
        // Create force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(50))
            .force("charge", d3.forceManyBody().strength(-100))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(10));
        
        // Create links
        const link = svg.append("g")
            .selectAll("line")
            .data(graphData.links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight * 2));
        
        // Create nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .join("circle")
            .attr("class", "node")
            .attr("r", d => 3 + Math.log(d.frequency + 1) * 2)
            .attr("fill", d => colorScale(d.domain))
            .call(drag(simulation))
            .on("click", showNodeInfo);
        
        // Create labels
        const label = svg.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .join("text")
            .attr("class", "node-label")
            .text(d => d.label)
            .attr("x", 8)
            .attr("y", 3);
        
        // Update positions on tick
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x + 8)
                .attr("y", d => d.y + 3);
        });
        
        // Update info
        document.getElementById("node-count").textContent = graphData.nodes.length;
        document.getElementById("edge-count").textContent = graphData.links.length;
        
        function showNodeInfo(event, d) {
            document.getElementById("selected").textContent = 
                `${d.label} (${d.type}, freq: ${d.frequency})`;
        }
        
        function drag(simulation) {
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }
    </script>
</body>
</html>"""

        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

        print(f"HTML visualization created: {output_file}")
        print(f"Open {output_file} in a web browser to view the interactive graph")

    def generate_statistics_report(self, output_file: str = "graph_statistics.txt"):
        """Generate a detailed statistics report"""
        stats = self.graph.get_statistics()

        lines = []
        lines.append("="*60)
        lines.append("GRAPH NETWORK STATISTICS REPORT")
        lines.append("="*60)
        lines.append("")
        lines.append(f"Total Nodes: {stats['num_nodes']}")
        lines.append(f"Total Edges: {stats['num_edges']}")
        lines.append(f"Average Edge Weight: {stats['avg_edge_weight']:.4f}")
        lines.append(f"Average Node Degree: {stats['avg_node_degree']:.2f}")
        lines.append(f"Maximum Node Degree: {stats['max_node_degree']}")
        lines.append("")
        lines.append("Domain Distribution:")
        for domain, count in stats['domain_distribution'].items():
            lines.append(f"  {domain}: {count}")
        lines.append("")

        # Top nodes by frequency
        lines.append("Top 20 Nodes by Frequency:")
        top_nodes = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.frequency,
            reverse=True
        )[:20]
        for i, node in enumerate(top_nodes, 1):
            lines.append(f"  {i}. {node.value} (freq: {node.frequency}, domain: {node.domain.value})")

        lines.append("")
        lines.append("="*60)

        report = '\n'.join(lines)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Statistics report generated: {output_file}")
        return report

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))

    def _escape_dot(self, text: str) -> str:
        """Escape DOT special characters"""
        return text.replace('"', '\\"').replace('\n', '\\n')

