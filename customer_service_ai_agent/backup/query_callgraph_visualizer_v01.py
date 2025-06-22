#!/usr/bin/env python3
"""
Query Execution Call Graph Visualizer
Tracks and visualizes the complete query execution process including thinking, tool selection,
logical inference, and results in interactive HTML graphs.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

class NodeType(Enum):
    """Types of nodes in the execution graph"""
    QUERY_START = "query_start"
    THINKING = "thinking"
    DECOMPOSITION = "decomposition"
    SUBGOAL = "subgoal"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    LOGICAL_INFERENCE = "logical_inference"
    RESULT_SYNTHESIS = "result_synthesis"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"
    CACHE_HIT = "cache_hit"

class ExecutionStatus(Enum):
    """Status of execution nodes"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class ExecutionNode:
    """Represents a node in the execution graph"""
    id: str
    node_type: NodeType
    title: str
    description: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)

@dataclass
class ExecutionEdge:
    """Represents an edge in the execution graph"""
    id: str
    source: str
    target: str
    label: str = ""
    edge_type: str = "flow"
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryExecutionTracker:
    """Tracks query execution and generates call graphs"""

    def __init__(self, output_dir: str = "./generated_callgraphs"):
        self.output_dir = output_dir
        self.current_query_id = None
        self.nodes: Dict[str, ExecutionNode] = {}
        self.edges: Dict[str, ExecutionEdge] = {}
        self.current_context_stack: List[str] = []
        self.query_start_time = None

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def start_query_tracking(self, query: str) -> str:
        """Start tracking a new query execution"""
        self.query_start_time = datetime.now()
        self.current_query_id = f"query_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        self.nodes = {}
        self.edges = {}
        self.current_context_stack = []

        # Create initial query node
        query_node = self._create_node(
            node_type=NodeType.QUERY_START,
            title="Query Received",
            description=f"Initial query: {query[:100]}{'...' if len(query) > 100 else ''}",
            input_data={"query": query}
        )

        self._start_node(query_node.id)
        self.current_context_stack.append(query_node.id)

        print(f"🎯 Started tracking query: {self.current_query_id}")
        return self.current_query_id

    def add_thinking_process(self, thought: str, context: str = "") -> str:
        """Add a thinking/reasoning step"""
        thinking_node = self._create_node(
            node_type=NodeType.THINKING,
            title="AI Thinking Process",
            description=f"Reasoning: {thought[:150]}{'...' if len(thought) > 150 else ''}",
            input_data={"thought": thought, "context": context}
        )

        self._start_node(thinking_node.id)
        self._connect_to_current_context(thinking_node.id, "thinks")

        return thinking_node.id

    def add_query_decomposition(self, complexity_score: int, subgoals: List[Dict]) -> str:
        """Add query decomposition step"""
        decomp_node = self._create_node(
            node_type=NodeType.DECOMPOSITION,
            title="Query Decomposition",
            description=f"Breaking query into {len(subgoals)} subgoals (complexity: {complexity_score})",
            input_data={"complexity_score": complexity_score, "subgoals": subgoals}
        )

        self._start_node(decomp_node.id)
        self._connect_to_current_context(decomp_node.id, "decomposes")

        # Create subgoal nodes
        for i, subgoal in enumerate(subgoals):
            subgoal_node = self._create_node(
                node_type=NodeType.SUBGOAL,
                title=f"Subgoal {i+1}",
                description=subgoal.get('subgoal', '')[:100],
                input_data=subgoal
            )
            self._connect_nodes(decomp_node.id, subgoal_node.id, f"creates subgoal {i+1}")

        self._complete_node(decomp_node.id, {"subgoal_count": len(subgoals)})
        return decomp_node.id

    def add_tool_selection(self, available_tools: List[str], selected_tool: str, reasoning: str) -> str:
        """Add tool selection step"""
        selection_node = self._create_node(
            node_type=NodeType.TOOL_SELECTION,
            title="Tool Selection",
            description=f"Selected: {selected_tool} from {len(available_tools)} available tools",
            input_data={
                "available_tools": available_tools,
                "selected_tool": selected_tool,
                "reasoning": reasoning
            }
        )

        self._start_node(selection_node.id)
        self._connect_to_current_context(selection_node.id, "selects tool")
        self._complete_node(selection_node.id, {"selected_tool": selected_tool})

        return selection_node.id

    def add_tool_execution(self, tool_name: str, input_params: Any, is_cached: bool = False) -> str:
        """Add tool execution step"""
        node_type = NodeType.CACHE_HIT if is_cached else NodeType.TOOL_EXECUTION

        execution_node = self._create_node(
            node_type=node_type,
            title=f"Execute: {tool_name}" + (" (Cached)" if is_cached else ""),
            description=f"Running {tool_name} with parameters: {str(input_params)[:100]}",
            input_data={"tool_name": tool_name, "parameters": input_params, "cached": is_cached}
        )

        self._start_node(execution_node.id)
        self._connect_to_current_context(execution_node.id, "executes" if not is_cached else "retrieves from cache")

        return execution_node.id

    def complete_tool_execution(self, node_id: str, result: Any, success: bool = True):
        """Complete a tool execution"""
        status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self._complete_node(node_id, {"result": result, "success": success}, status)

    def add_logical_inference(self, reasoning: str, evidence: List[str], conclusion: str) -> str:
        """Add logical inference step"""
        inference_node = self._create_node(
            node_type=NodeType.LOGICAL_INFERENCE,
            title="Logical Inference",
            description=f"Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}",
            input_data={
                "reasoning": reasoning,
                "evidence": evidence,
                "conclusion": conclusion
            }
        )

        self._start_node(inference_node.id)
        self._connect_to_current_context(inference_node.id, "infers")
        self._complete_node(inference_node.id, {"conclusion": conclusion})

        return inference_node.id

    def add_result_synthesis(self, partial_results: List[Any], synthesis_method: str) -> str:
        """Add result synthesis step"""
        synthesis_node = self._create_node(
            node_type=NodeType.RESULT_SYNTHESIS,
            title="Result Synthesis",
            description=f"Combining {len(partial_results)} results using {synthesis_method}",
            input_data={
                "partial_results": partial_results,
                "synthesis_method": synthesis_method
            }
        )

        self._start_node(synthesis_node.id)
        self._connect_to_current_context(synthesis_node.id, "synthesizes")

        return synthesis_node.id

    def complete_result_synthesis(self, node_id: str, final_result: Any):
        """Complete result synthesis"""
        self._complete_node(node_id, {"final_result": final_result})

    def add_error(self, error_message: str, error_type: str = "general") -> str:
        """Add an error node"""
        error_node = self._create_node(
            node_type=NodeType.ERROR,
            title=f"Error: {error_type}",
            description=f"Error occurred: {error_message[:100]}",
            input_data={"error_message": error_message, "error_type": error_type}
        )

        self._start_node(error_node.id)
        self._connect_to_current_context(error_node.id, "encounters error")
        self._complete_node(error_node.id, {"error": error_message}, ExecutionStatus.FAILED)

        return error_node.id

    def finalize_query(self, final_response: str, success: bool = True) -> str:
        """Finalize the query with the final response"""
        final_node = self._create_node(
            node_type=NodeType.FINAL_RESPONSE,
            title="Final Response",
            description=f"Query completed: {final_response[:100]}{'...' if len(final_response) > 100 else ''}",
            input_data={"response": final_response, "success": success}
        )

        self._start_node(final_node.id)
        self._connect_to_current_context(final_node.id, "produces")

        status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self._complete_node(final_node.id, {"final_response": final_response}, status)

        # Generate the HTML graph
        html_file = self._generate_html_graph()
        print(f"📊 Query execution graph saved to: {html_file}")

        return final_node.id

    def _create_node(self, node_type: NodeType, title: str, description: str, input_data: Any = None) -> ExecutionNode:
        """Create a new execution node"""
        node_id = f"{node_type.value}_{uuid.uuid4().hex[:8]}"

        node = ExecutionNode(
            id=node_id,
            node_type=node_type,
            title=title,
            description=description,
            input_data=input_data or {}
        )

        self.nodes[node_id] = node
        return node

    def _start_node(self, node_id: str):
        """Mark a node as started"""
        if node_id in self.nodes:
            self.nodes[node_id].status = ExecutionStatus.IN_PROGRESS
            self.nodes[node_id].start_time = datetime.now()

    def _complete_node(self, node_id: str, output_data: Any = None, status: ExecutionStatus = ExecutionStatus.COMPLETED):
        """Mark a node as completed"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = status
            node.end_time = datetime.now()
            if node.start_time:
                node.execution_time = (node.end_time - node.start_time).total_seconds()
            node.output_data = output_data or {}

    def _connect_nodes(self, source_id: str, target_id: str, label: str = "", edge_type: str = "flow"):
        """Connect two nodes with an edge"""
        edge_id = f"{source_id}_to_{target_id}"

        edge = ExecutionEdge(
            id=edge_id,
            source=source_id,
            target=target_id,
            label=label,
            edge_type=edge_type
        )

        self.edges[edge_id] = edge

        # Update node relationships
        if source_id in self.nodes:
            self.nodes[source_id].children.append(target_id)
        if target_id in self.nodes:
            self.nodes[target_id].parents.append(source_id)

    def _connect_to_current_context(self, node_id: str, label: str = ""):
        """Connect a node to the current context"""
        if self.current_context_stack:
            current_context = self.current_context_stack[-1]
            self._connect_nodes(current_context, node_id, label)
            # Update context stack
            self.current_context_stack.append(node_id)

    def _generate_html_graph(self) -> str:
        """Generate interactive HTML graph"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_query_id}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for visualization
        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "query_id": self.current_query_id,
                "generated_at": datetime.now().isoformat(),
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "execution_time": (datetime.now() - self.query_start_time).total_seconds() if self.query_start_time else 0
            }
        }

        # Convert nodes to vis.js format
        for node in self.nodes.values():
            color = self._get_node_color(node.node_type, node.status)
            shape = self._get_node_shape(node.node_type)

            graph_data["nodes"].append({
                "id": node.id,
                "label": node.title,
                "title": self._create_node_tooltip(node),
                "color": color,
                "shape": shape,
                "font": {"size": 12},
                "physics": True
            })

        # Convert edges to vis.js format
        for edge in self.edges.values():
            graph_data["edges"].append({
                "from": edge.source,
                "to": edge.target,
                "label": edge.label,
                "arrows": "to",
                "font": {"size": 10},
                "color": {"color": "#848484"},
                "smooth": {"type": "continuous"}
            })

        # Generate HTML content
        html_content = self._create_html_template(graph_data)

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filepath

    def _get_node_color(self, node_type: NodeType, status: ExecutionStatus) -> Dict[str, str]:
        """Get color scheme for node based on type and status"""
        base_colors = {
            NodeType.QUERY_START: "#4CAF50",
            NodeType.THINKING: "#2196F3",
            NodeType.DECOMPOSITION: "#FF9800",
            NodeType.SUBGOAL: "#FFC107",
            NodeType.TOOL_SELECTION: "#9C27B0",
            NodeType.TOOL_EXECUTION: "#795548",
            NodeType.LOGICAL_INFERENCE: "#607D8B",
            NodeType.RESULT_SYNTHESIS: "#E91E63",
            NodeType.FINAL_RESPONSE: "#4CAF50",
            NodeType.ERROR: "#F44336",
            NodeType.CACHE_HIT: "#00BCD4"
        }

        base_color = base_colors.get(node_type, "#9E9E9E")

        if status == ExecutionStatus.FAILED:
            return {"background": "#F44336", "border": "#D32F2F"}
        elif status == ExecutionStatus.IN_PROGRESS:
            return {"background": "#FFE082", "border": "#FFA000"}
        elif status == ExecutionStatus.CACHED:
            return {"background": "#B2DFDB", "border": "#00695C"}
        else:
            return {"background": base_color, "border": base_color}

    def _get_node_shape(self, node_type: NodeType) -> str:
        """Get shape for node based on type"""
        shapes = {
            NodeType.QUERY_START: "star",
            NodeType.THINKING: "ellipse",
            NodeType.DECOMPOSITION: "diamond",
            NodeType.SUBGOAL: "box",
            NodeType.TOOL_SELECTION: "triangle",
            NodeType.TOOL_EXECUTION: "dot",
            NodeType.LOGICAL_INFERENCE: "hexagon",
            NodeType.RESULT_SYNTHESIS: "square",
            NodeType.FINAL_RESPONSE: "star",
            NodeType.ERROR: "triangleDown",
            NodeType.CACHE_HIT: "circle"
        }
        return shapes.get(node_type, "ellipse")

    def _create_node_tooltip(self, node: ExecutionNode) -> str:
        """Create detailed tooltip for node"""
        tooltip = f"<b>{node.title}</b><br/>"
        tooltip += f"Type: {node.node_type.value}<br/>"
        tooltip += f"Status: {node.status.value}<br/>"
        tooltip += f"Description: {node.description}<br/>"

        if node.execution_time > 0:
            tooltip += f"Execution Time: {node.execution_time:.3f}s<br/>"

        if node.input_data:
            tooltip += f"Input: {str(node.input_data)[:200]}...<br/>"

        if node.output_data:
            tooltip += f"Output: {str(node.output_data)[:200]}...<br/>"

        return tooltip

    def _create_html_template(self, graph_data: Dict) -> str:
        """Create the complete HTML template with embedded vis.js"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Execution Graph - {graph_data['metadata']['query_id']}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        
        .metadata {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }}
        
        .metadata div {{
            background: rgba(255,255,255,0.2);
            padding: 10px 15px;
            border-radius: 5px;
        }}
        
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .controls button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .controls button:hover {{
            background: #45a049;
        }}
        
        #network {{
            width: 100%;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .legend h3 {{
            margin-top: 0;
        }}
        
        .legend-item {{
            display: inline-block;
            margin: 5px 15px 5px 0;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 10px;
            margin-right: 8px;
            vertical-align: middle;
        }}
        
        .info-panel {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .info-panel h3 {{
            margin-top: 0;
            color: #333;
        }}
        
        .node-details {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            border-left: 4px solid #007bff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Query Execution Call Graph</h1>
        <div class="metadata">
            <div><strong>Query ID:</strong> {graph_data['metadata']['query_id']}</div>
            <div><strong>Generated:</strong> {graph_data['metadata']['generated_at']}</div>
            <div><strong>Nodes:</strong> {graph_data['metadata']['total_nodes']}</div>
            <div><strong>Edges:</strong> {graph_data['metadata']['total_edges']}</div>
            <div><strong>Total Time:</strong> {graph_data['metadata']['execution_time']:.2f}s</div>
        </div>
    </div>
    
    <div class="controls">
        <button onclick="fitNetwork()">🔍 Fit to Screen</button>
        <button onclick="togglePhysics()">⚡ Toggle Physics</button>
        <button onclick="exportImage()">📷 Export Image</button>
        <button onclick="showStatistics()">📊 Show Statistics</button>
        <button onclick="resetView()">🔄 Reset View</button>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <h3>🎨 Node Types Legend</h3>
        <div class="legend-item"><span class="legend-color" style="background: #4CAF50;"></span>Query Start/Final Response</div>
        <div class="legend-item"><span class="legend-color" style="background: #2196F3;"></span>Thinking Process</div>
        <div class="legend-item"><span class="legend-color" style="background: #FF9800;"></span>Decomposition</div>
        <div class="legend-item"><span class="legend-color" style="background: #FFC107;"></span>Subgoal</div>
        <div class="legend-item"><span class="legend-color" style="background: #9C27B0;"></span>Tool Selection</div>
        <div class="legend-item"><span class="legend-color" style="background: #795548;"></span>Tool Execution</div>
        <div class="legend-item"><span class="legend-color" style="background: #607D8B;"></span>Logical Inference</div>
        <div class="legend-item"><span class="legend-color" style="background: #E91E63;"></span>Result Synthesis</div>
        <div class="legend-item"><span class="legend-color" style="background: #00BCD4;"></span>Cache Hit</div>
        <div class="legend-item"><span class="legend-color" style="background: #F44336;"></span>Error</div>
    </div>
    
    <div class="info-panel">
        <h3>📋 Node Information</h3>
        <div id="node-info">Click on a node to see detailed information</div>
    </div>
    
    <script>
        // Graph data
        const graphData = {json.dumps(graph_data, indent=2)};
        
        // Create network
        const container = document.getElementById('network');
        
        const options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                    nodeSpacing: 150
                }}
            }},
            physics: {{
                enabled: true,
                hierarchicalRepulsion: {{
                    centralGravity: 0.0,
                    springLength: 100,
                    springConstant: 0.01,
                    nodeDistance: 120,
                    damping: 0.09
                }}
            }},
            nodes: {{
                font: {{ size: 12, color: '#333' }},
                borderWidth: 2,
                shadow: true,
                chosen: true
            }},
            edges: {{
                font: {{ size: 10, color: '#333' }},
                width: 2,
                shadow: true,
                smooth: {{
                    type: 'continuous'
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 300,
                hideEdgesOnDrag: false,
                hideNodesOnDrag: false
            }}
        }};
        
        const network = new vis.Network(container, graphData, options);
        
        // Event listeners
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = graphData.nodes.find(n => n.id === nodeId);
                showNodeDetails(node);
            }}
        }});
        
        network.on("hoverNode", function(params) {{
            container.style.cursor = 'pointer';
        }});
        
        network.on("blurNode", function(params) {{
            container.style.cursor = 'default';
        }});
        
        // Control functions
        function fitNetwork() {{
            network.fit();
        }}
        
        let physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: {{enabled: physicsEnabled}}}});
        }}
        
        function exportImage() {{
            const canvas = container.querySelector('canvas');
            const link = document.createElement('a');
            link.download = 'query_execution_graph.png';
            link.href = canvas.toDataURL();
            link.click();
        }}
        
        function showStatistics() {{
            const stats = calculateStatistics();
            alert(`Graph Statistics:\\n\\nTotal Nodes: ${{stats.totalNodes}}\\nTotal Edges: ${{stats.totalEdges}}\\nAvg Execution Time: ${{stats.avgExecutionTime}}s\\nSuccess Rate: ${{stats.successRate}}%\\nMost Common Node Type: ${{stats.mostCommonType}}`);
        }}
        
        function resetView() {{
            network.fit();
            network.setOptions({{physics: {{enabled: true}}}});
            physicsEnabled = true;
        }}
        
        function showNodeDetails(node) {{
            const infoDiv = document.getElementById('node-info');
            infoDiv.innerHTML = `
                <div class="node-details">
                    <h4>${{node.label}}</h4>
                    <p><strong>ID:</strong> ${{node.id}}</p>
                    <p><strong>Type:</strong> ${{node.title}}</p>
                    <div><strong>Details:</strong><br>${{node.title}}</div>
                </div>
            `;
        }}
        
        function calculateStatistics() {{
            const totalNodes = graphData.nodes.length;
            const totalEdges = graphData.edges.length;
            
            // Calculate node type distribution
            const typeCount = {{}};
            graphData.nodes.forEach(node => {{
                const type = node.shape;
                typeCount[type] = (typeCount[type] || 0) + 1;
            }});
            
            const mostCommonType = Object.keys(typeCount).reduce((a, b) => 
                typeCount[a] > typeCount[b] ? a : b
            );
            
            return {{
                totalNodes,
                totalEdges,
                avgExecutionTime: {graph_data['metadata']['execution_time']:.3f},
                successRate: 95, // Placeholder - could be calculated from actual data
                mostCommonType
            }};
        }}
        
        // Initialize
        fitNetwork();
    </script>
</body>
</html>
        """

# Global tracker instance
execution_tracker = QueryExecutionTracker()

# Integration decorators and functions for the main agent
def track_execution(node_type: NodeType, title: str = "", description: str = ""):
    """Decorator to automatically track function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create tracking node
            func_title = title or f"{func.__name__}"
            func_desc = description or f"Executing {func.__name__}"

            node_id = None
            if node_type == NodeType.TOOL_EXECUTION:
                tool_name = getattr(func, '__name__', 'unknown_tool')
                input_params = {'args': str(args)[:200], 'kwargs': str(kwargs)[:200]}
                node_id = execution_tracker.add_tool_execution(tool_name, input_params)
            elif node_type == NodeType.THINKING:
                node_id = execution_tracker.add_thinking_process(func_desc)
            elif node_type == NodeType.LOGICAL_INFERENCE:
                node_id = execution_tracker.add_logical_inference(func_desc, [], "Processing logic")

            try:
                result = func(*args, **kwargs)

                if node_id and node_type == NodeType.TOOL_EXECUTION:
                    execution_tracker.complete_tool_execution(node_id, result, success=True)

                return result

            except Exception as e:
                if node_id and node_type == NodeType.TOOL_EXECUTION:
                    execution_tracker.complete_tool_execution(node_id, str(e), success=False)

                execution_tracker.add_error(str(e), error_type=func.__name__)
                raise

        return wrapper
    return decorator


class EnhancedQueryExecutionTracker(QueryExecutionTracker):
    """Enhanced tracker that integrates with the customer service agent"""

    def __init__(self, output_dir: str = "./generated_callgraphs"):
        super().__init__(output_dir)
        self.agent_context = {}
        self.tool_call_sequence = []

    def track_agent_query_start(self, query: str, agent_type: str = "CustomerServiceAgent"):
        """Enhanced query start tracking with agent context"""
        query_id = self.start_query_tracking(query)

        # Add agent thinking process
        thinking_node = self.add_thinking_process(
            f"Customer service agent received query: {query}",
            f"Agent type: {agent_type}, preparing to process customer inquiry"
        )

        return query_id

    def track_complexity_assessment(self, query: str, complexity_score: int, patterns: List[str]):
        """Track query complexity assessment"""
        thinking_node = self.add_thinking_process(
            f"Analyzing query complexity. Score: {complexity_score}/10",
            f"Detected patterns: {', '.join(patterns) if patterns else 'None'}"
        )

        if complexity_score >= 3:
            return self.add_query_decomposition(complexity_score, [])

        return thinking_node

    def track_tool_selection_process(self, available_tools: List[str], query_context: str):
        """Track the tool selection reasoning process"""
        reasoning = f"Analyzing query context: {query_context[:100]}..."
        thinking_node = self.add_thinking_process(
            f"Evaluating {len(available_tools)} available tools for this query",
            reasoning
        )
        return thinking_node

    def track_database_operation(self, operation_type: str, query: str, params: Any = None):
        """Track database operations specifically"""
        db_node = self.add_tool_execution(
            tool_name=f"Database_{operation_type}",
            input_params={"sql_query": query[:200], "parameters": params}
        )
        return db_node

    def track_cache_operation(self, operation: str, cache_key: str, hit: bool):
        """Track cache hits and misses"""
        if hit:
            cache_node = self._create_node(
                node_type=NodeType.CACHE_HIT,
                title=f"Cache Hit: {operation}",
                description=f"Retrieved cached result for key: {cache_key[:50]}",
                input_data={"cache_key": cache_key, "operation": operation}
            )
        else:
            cache_node = self._create_node(
                node_type=NodeType.TOOL_EXECUTION,
                title=f"Cache Miss: {operation}",
                description=f"Cache miss, executing operation: {operation}",
                input_data={"cache_key": cache_key, "operation": operation}
            )

        self._start_node(cache_node.id)
        self._connect_to_current_context(cache_node.id, "cache check")
        self._complete_node(cache_node.id, {"cache_hit": hit})

        return cache_node.id

    def track_agent_reasoning(self, step: str, evidence: List[str], intermediate_conclusion: str):
        """Track agent's reasoning steps"""
        return self.add_logical_inference(
            reasoning=step,
            evidence=evidence,
            conclusion=intermediate_conclusion
        )

    def track_response_synthesis(self, partial_results: List[Dict], method: str = "comprehensive_analysis"):
        """Track how the agent synthesizes multiple results"""
        synthesis_node = self.add_result_synthesis(partial_results, method)

        # Add thinking about synthesis strategy
        thinking_node = self.add_thinking_process(
            f"Synthesizing {len(partial_results)} partial results using {method}",
            "Combining database results, policy information, and logical inferences"
        )

        return synthesis_node

    def export_execution_summary(self) -> Dict[str, Any]:
        """Export a summary of the execution for logging"""
        if not self.nodes:
            return {}

        summary = {
            "query_id": self.current_query_id,
            "total_execution_time": (datetime.now() - self.query_start_time).total_seconds() if self.query_start_time else 0,
            "node_counts": {},
            "tool_calls": len([n for n in self.nodes.values() if n.node_type == NodeType.TOOL_EXECUTION]),
            "cache_hits": len([n for n in self.nodes.values() if n.node_type == NodeType.CACHE_HIT]),
            "errors": len([n for n in self.nodes.values() if n.node_type == NodeType.ERROR]),
            "reasoning_steps": len([n for n in self.nodes.values() if n.node_type == NodeType.LOGICAL_INFERENCE]),
            "thinking_steps": len([n for n in self.nodes.values() if n.node_type == NodeType.THINKING])
        }

        # Count node types
        for node in self.nodes.values():
            node_type = node.node_type.value
            summary["node_counts"][node_type] = summary["node_counts"].get(node_type, 0) + 1

        return summary


# Enhanced execution tracker instance
execution_tracker = EnhancedQueryExecutionTracker()


# Integration functions for the main customer service agent
def integrate_with_agent():
    """Integration functions to add to the CustomerServiceAgent class"""

    def enhanced_run_query(self, query: str):
        """Enhanced query execution with full call graph tracking"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        # Start tracking
        query_id = execution_tracker.track_agent_query_start(query)

        try:
            print(f"🤖 Processing query with call graph tracking: '{query}'")
            print(f"📊 Tracking ID: {query_id}")
            print("-" * 70)

            # Track initial agent thinking
            execution_tracker.add_thinking_process(
                "Received customer query, initializing response process",
                f"Query length: {len(query)} characters, complexity assessment needed"
            )

            # Assess query complexity
            if hasattr(self, 'query_decomposer') and self.query_decomposer:
                assessment = self.query_decomposer.assess_query_complexity(query)
                execution_tracker.track_complexity_assessment(
                    query, assessment['complexity_score'], assessment['detected_patterns']
                )

                if assessment['requires_decomposition']:
                    execution_tracker.add_thinking_process(
                        "Query is complex, will decompose into subgoals",
                        f"Complexity score: {assessment['complexity_score']}, patterns: {assessment['detected_patterns']}"
                    )

            # Track tool availability assessment
            available_tools = [tool.metadata.name for tool in self.tools] if self.tools else []
            execution_tracker.track_tool_selection_process(available_tools, query)

            # Execute the actual query with timeout
            import signal
            def timeout_handler(signum, frame):
                execution_tracker.add_error("Query execution timeout", "timeout")
                raise TimeoutError("Query execution timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                # Track the main agent execution
                execution_tracker.add_thinking_process(
                    "Executing main agent query processing",
                    "Agent will now process the query using available tools and knowledge"
                )

                response = self.agent.query(query)
                signal.alarm(0)

                # Track successful completion
                execution_tracker.add_thinking_process(
                    "Query processing completed successfully",
                    f"Generated response of {len(str(response))} characters"
                )

                # Finalize tracking
                final_response = str(response)
                execution_tracker.finalize_query(final_response, success=True)

                print("\n✅ Agent Response:")
                print("=" * 30)
                print(response)
                print()

                # Show execution summary
                summary = execution_tracker.export_execution_summary()
                print(f"📊 Execution Summary:")
                print(f"   Total time: {summary['total_execution_time']:.2f}s")
                print(f"   Tool calls: {summary['tool_calls']}")
                print(f"   Cache hits: {summary['cache_hits']}")
                print(f"   Reasoning steps: {summary['reasoning_steps']}")
                print(f"   Graph saved to: ./generated_callgraphs/")

            except TimeoutError:
                signal.alarm(0)
                execution_tracker.add_error("Query timeout exceeded", "timeout")
                execution_tracker.finalize_query("Query timed out", success=False)
                print("\n⏰ Query timeout! The query took too long to process.")

            except Exception as query_error:
                signal.alarm(0)
                execution_tracker.add_error(str(query_error), "execution_error")
                execution_tracker.finalize_query(f"Query failed: {str(query_error)}", success=False)
                print(f"\n❌ Query execution error: {query_error}")

        except Exception as e:
            execution_tracker.add_error(str(e), "critical_error")
            execution_tracker.finalize_query(f"Critical error: {str(e)}", success=False)
            print(f"❌ Critical error during query execution: {e}")

    def enhanced_run_intelligent_query(self, query: str):
        """Enhanced intelligent query with decomposition tracking"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            return self.enhanced_run_query(query)

        # Start tracking
        query_id = execution_tracker.track_agent_query_start(query)

        try:
            print(f"🤖 Processing query with intelligent decomposition and tracking: '{query}'")
            print("=" * 70)

            # Track complexity assessment
            assessment = self.query_decomposer.assess_query_complexity(query)
            execution_tracker.track_complexity_assessment(
                query, assessment['complexity_score'], assessment['detected_patterns']
            )

            # Track decomposition process
            execution_tracker.add_thinking_process(
                "Starting intelligent query decomposition",
                f"Complexity: {assessment['complexity_score']}, indicators: {assessment['indicators']}"
            )

            # Get subgoals and track them
            subgoals = self.query_decomposer.decompose_query(query)
            decomp_node = execution_tracker.add_query_decomposition(
                assessment['complexity_score'], subgoals
            )

            print(f"\n📝 Query decomposed into {len(subgoals)} subgoals:")
            for i, subgoal in enumerate(subgoals, 1):
                print(f"   {i}. {subgoal['subgoal']} [{subgoal['type']}]")

            # Execute subgoals with tracking
            results = []
            context = []

            for i, subgoal in enumerate(subgoals, 1):
                print(f"\n🎯 Executing subgoal {i}: {subgoal['subgoal']}")

                # Track subgoal execution start
                subgoal_thinking = execution_tracker.add_thinking_process(
                    f"Processing subgoal {i}/{len(subgoals)}",
                    f"Type: {subgoal['type']}, Priority: {subgoal['priority']}"
                )

                # Add context from previous subgoals
                contextual_query = subgoal['subgoal']
                if context:
                    contextual_query = f"Based on previous analysis: {' '.join(context[-2:])}. Now: {subgoal['subgoal']}"
                    execution_tracker.add_logical_inference(
                        f"Incorporating context from {len(context)} previous subgoals",
                        context[-2:] if len(context) >= 2 else context,
                        f"Enhanced query: {contextual_query[:100]}..."
                    )

                try:
                    # Track the individual subgoal execution
                    execution_tracker.add_thinking_process(
                        f"Executing subgoal via agent: {contextual_query[:100]}...",
                        f"Subgoal type: {subgoal['type']}"
                    )

                    result = self._execute_simple_query(contextual_query)

                    results.append({
                        'subgoal': subgoal['subgoal'],
                        'result': result,
                        'type': subgoal['type']
                    })

                    context.append(f"Subgoal {i} found: {str(result)[:200]}...")

                    # Track successful subgoal completion
                    execution_tracker.add_logical_inference(
                        f"Subgoal {i} completed successfully",
                        [f"Query: {subgoal['subgoal'][:50]}..."],
                        f"Result obtained: {str(result)[:100]}..."
                    )

                except Exception as e:
                    error_msg = f"Subgoal {i} failed: {e}"
                    execution_tracker.add_error(error_msg, f"subgoal_{i}_error")

                    results.append({
                        'subgoal': subgoal['subgoal'],
                        'result': f"Failed to execute: {e}",
                        'type': subgoal['type']
                    })

            # Track result synthesis
            execution_tracker.add_thinking_process(
                f"Synthesizing results from {len(results)} subgoals",
                "Combining partial results into comprehensive response"
            )

            synthesis_node = execution_tracker.track_response_synthesis(
                results, "subgoal_based_synthesis"
            )

            # Generate final response
            final_response = self._synthesize_results(query, results)

            execution_tracker.complete_result_synthesis(synthesis_node, final_response)
            execution_tracker.finalize_query(final_response, success=True)

            print(f"\n✅ Complete Response:")
            print("=" * 50)
            print(final_response)

            # Show execution summary
            summary = execution_tracker.export_execution_summary()
            print(f"\n📊 Intelligent Execution Summary:")
            print(f"   Subgoals processed: {len(subgoals)}")
            print(f"   Total time: {summary['total_execution_time']:.2f}s")
            print(f"   Reasoning steps: {summary['reasoning_steps']}")
            print(f"   Thinking processes: {summary['thinking_steps']}")

        except Exception as e:
            execution_tracker.add_error(str(e), "intelligent_query_error")
            execution_tracker.finalize_query(f"Intelligent query failed: {str(e)}", success=False)
            print(f"❌ Intelligent query execution failed: {e}")

    return enhanced_run_query, enhanced_run_intelligent_query


# Tool execution tracking wrapper
class TrackedCustomerServiceTools:
    """Wrapper for CustomerServiceTools that adds execution tracking"""

    def __init__(self, original_tools):
        self.original_tools = original_tools
        self._wrap_all_methods()

    def _wrap_all_methods(self):
        """Wrap all public methods with tracking"""
        for attr_name in dir(self.original_tools):
            if not attr_name.startswith('_'):
                attr = getattr(self.original_tools, attr_name)
                if callable(attr):
                    wrapped_method = self._create_tracked_method(attr, attr_name)
                    setattr(self, attr_name, wrapped_method)

    def _create_tracked_method(self, original_method, method_name):
        """Create a tracked version of a method"""
        def tracked_method(*args, **kwargs):
            # Check for cache first
            cache_key = f"{method_name}_{str(args)[:100]}_{str(kwargs)[:100]}"

            # Determine if this will be a cache hit (simplified logic)
            is_cached = hasattr(original_method, '__wrapped__') and 'cached' in str(original_method)

            # Track the execution
            if is_cached:
                node_id = execution_tracker.track_cache_operation(method_name, cache_key, True)
            else:
                node_id = execution_tracker.add_tool_execution(
                    tool_name=method_name,
                    input_params={"args": args[1:], "kwargs": kwargs},  # Skip 'self'
                    is_cached=False
                )

            try:
                result = original_method(*args, **kwargs)

                if not is_cached:
                    execution_tracker.complete_tool_execution(node_id, result, success=True)

                # Add logical inference about the result
                if isinstance(result, dict) and 'error' not in result:
                    execution_tracker.add_logical_inference(
                        f"Tool {method_name} executed successfully",
                        [f"Input: {str(args[1:])[:100]}"],
                        f"Obtained valid result: {str(result)[:100]}..."
                    )
                elif isinstance(result, list) and result:
                    execution_tracker.add_logical_inference(
                        f"Tool {method_name} returned {len(result)} items",
                        [f"Query parameters: {str(args[1:])[:100]}"],
                        f"Successfully retrieved data: {len(result)} records"
                    )

                return result

            except Exception as e:
                if not is_cached:
                    execution_tracker.complete_tool_execution(node_id, str(e), success=False)

                execution_tracker.add_error(f"Tool {method_name} failed: {str(e)}", f"tool_{method_name}")
                raise

        return tracked_method


# Enhanced integration instructions
def get_integration_instructions():
    """Get instructions for integrating with the main agent"""
    return """
# Integration Instructions for Query Call Graph Visualization

## 1. Add imports to the main agent file:
```python
from query_callgraph_visualizer import (
    execution_tracker, 
    EnhancedQueryExecutionTracker,
    TrackedCustomerServiceTools,
    integrate_with_agent
)
```

## 2. Modify CustomerServiceAgent.__init__:
```python
def __init__(self):
    self.db_manager = DatabaseManager()
    self.tools = None
    self.agent = None
    self.support_index = None
    self.query_decomposer = None
    self.sync_tools = None
    self.async_tools = None
    # ADD THIS LINE:
    self.execution_tracker = EnhancedQueryExecutionTracker()
```

## 3. Wrap tools in create_tools method:
```python
def create_tools(self):
    # ... existing tool creation code ...
    
    # Create regular database-connected tools
    self.sync_tools = CustomerServiceTools(self.db_manager)
    
    # ADD THIS LINE to wrap tools with tracking:
    self.sync_tools = TrackedCustomerServiceTools(self.sync_tools)
    
    # ... rest of method unchanged ...
```

## 4. Replace query execution methods:
```python
# Add these methods to CustomerServiceAgent class:
enhanced_run_query, enhanced_run_intelligent_query = integrate_with_agent()

# Replace existing run_query method
def run_query(self, query: str):
    return enhanced_run_query(self, query)

# Replace existing run_intelligent_query method  
def run_intelligent_query(self, query: str):
    return enhanced_run_intelligent_query(self, query)
```

## 5. Add menu option for viewing graphs:
```python
def main_menu(self):
    # ... existing menu items ...
    print("11. 📊 View Call Graphs")  # Add this option
    
    # In the choice handling:
    elif choice == '11':
        self.view_call_graphs()

def view_call_graphs(self):
    "View generated call graphs"
    import os
    import webbrowser
    
    graph_dir = "./generated_callgraphs"
    if not os.path.exists(graph_dir):
        print("No call graphs generated yet.")
        return
    
    html_files = [f for f in os.listdir(graph_dir) if f.endswith('.html')]
    
    if not html_files:
        print("No HTML call graphs found.")
        return
    
    print(f"\\n📊 Generated Call Graphs ({len(html_files)} files):")
    for i, file in enumerate(html_files, 1):
        print(f"   {i}. {file}")
    
    choice = input("\\nSelect file to open (number) or 'latest' for most recent: ").strip()
    
    if choice.lower() == 'latest':
        latest_file = max(html_files, key=lambda f: os.path.getctime(os.path.join(graph_dir, f)))
        filepath = os.path.join(graph_dir, latest_file)
    elif choice.isdigit() and 1 <= int(choice) <= len(html_files):
        filepath = os.path.join(graph_dir, html_files[int(choice)-1])
    else:
        print("Invalid choice.")
        return
    
    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(filepath)}')
    print(f"✅ Opened {os.path.basename(filepath)} in browser")
```

## 6. The call graph will automatically track:
- Query start and complexity assessment
- Tool selection reasoning
- Database operations
- Cache hits/misses  
- Logical inference steps
- Error handling
- Result synthesis
- Final response generation

Each query execution will generate an interactive HTML file in ./generated_callgraphs/ 
with a complete visualization of the execution flow.
"""

if __name__ == "__main__":
    print("Query Call Graph Visualizer")
    print("=" * 40)
    print(get_integration_instructions())