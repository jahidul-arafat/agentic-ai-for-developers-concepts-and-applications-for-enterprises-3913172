#!/usr/bin/env python3
"""
ENHANCED Query Execution Call Graph Visualizer
Enhanced features:
- Comprehensive node details panel without tooltip duplication
- Color-coded edges based on type and nature
- Better visual hierarchy and information architecture
- Improved data presentation and analysis
"""

import os
import json
import uuid
import time
import html
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re

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
    ACTION_SELECTION = "action_selection"
    ACTION_EXECUTION = "action_execution"

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

class EnhancedQueryExecutionTracker:
    """Enhanced tracker with complete tool and action tracking"""

    def __init__(self, output_dir: str = "./generated_callgraphs"):
        self.output_dir = output_dir
        self.current_query_id = None
        self.nodes: Dict[str, ExecutionNode] = {}
        self.edges: Dict[str, ExecutionEdge] = {}
        self.current_context_stack: List[str] = []
        self.query_start_time = None
        self.agent_context = {}
        self.tool_call_sequence = []
        self.cache_tracker = CacheTracker()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def start_query_tracking(self, query: str) -> str:
        """Start tracking a new query execution"""
        self.query_start_time = datetime.now()
        self.current_query_id = f"query_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        self.nodes = {}
        self.edges = {}
        self.current_context_stack = []
        self.cache_tracker.reset()

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

    def track_agent_query_start(self, query: str, agent_type: str = "CustomerServiceAgent"):
        """Enhanced query start tracking with agent context"""
        query_id = self.start_query_tracking(query)

        # Add agent thinking process
        thinking_node = self.add_thinking_process(
            f"Customer service agent received query: {query}",
            f"Agent type: {agent_type}, preparing to process customer inquiry"
        )

        return query_id

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

    def add_action_selection(self, available_actions: List[str], selected_action: str, reasoning: str) -> str:
        """Track when LLM selects an action from available tools"""
        action_node = self._create_node(
            node_type=NodeType.ACTION_SELECTION,
            title=f"Action Selected: {selected_action}",
            description=f"Selected '{selected_action}' from {len(available_actions)} available actions",
            input_data={
                "available_actions": available_actions,
                "selected_action": selected_action,
                "reasoning": reasoning
            }
        )

        self._start_node(action_node.id)
        self._connect_to_current_context(action_node.id, "selects action")
        self._complete_node(action_node.id, {"selected_action": selected_action})

        return action_node.id

    def add_action_execution(self, action_name: str, input_params: Any, is_cached: bool = False) -> str:
        """Track actual action/tool execution"""
        node_type = NodeType.CACHE_HIT if is_cached else NodeType.ACTION_EXECUTION

        execution_node = self._create_node(
            node_type=node_type,
            title=f"Execute: {action_name}" + (" (Cache Hit)" if is_cached else ""),
            description=f"Running {action_name}" + (" - retrieved from cache" if is_cached else f" with parameters: {str(input_params)[:100]}"),
            input_data={"action_name": action_name, "parameters": input_params, "cached": is_cached}
        )

        self._start_node(execution_node.id)
        self._connect_to_current_context(execution_node.id, "executes" if not is_cached else "cache hit")

        return execution_node.id

    def complete_action_execution(self, node_id: str, result: Any, success: bool = True):
        """Complete an action execution"""
        status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self._complete_node(node_id, {"result": str(result)[:200], "success": success}, status)

    def track_cache_operation(self, operation: str, cache_key: str, hit: bool, execution_time: float = 0.0):
        """Track cache hits and misses with detailed timing"""
        if hit:
            cache_node = self._create_node(
                node_type=NodeType.CACHE_HIT,
                title=f"Cache Hit\n{operation}",  # Add line break in title
                description=f"Retrieved cached result for {operation} (saved ~{execution_time:.3f}s)",
                input_data={"cache_key": cache_key[:50], "operation": operation, "time_saved": execution_time}
            )
        else:
            cache_node = self._create_node(
                node_type=NodeType.TOOL_EXECUTION,
                title=f"Cache Miss: {operation}",
                description=f"Cache miss for {operation}, executing fresh query",
                input_data={"cache_key": cache_key[:50], "operation": operation}
            )

        self._start_node(cache_node.id)
        self._connect_to_current_context(cache_node.id, "cache hit" if hit else "cache miss")
        self._complete_node(cache_node.id, {"cache_hit": hit, "operation": operation})

        return cache_node.id

    def add_tool_execution(self, tool_name: str, input_params: Any, is_cached: bool = False) -> str:
        """Add tool execution step"""
        return self.add_action_execution(tool_name, input_params, is_cached)

    def complete_tool_execution(self, node_id: str, result: Any, success: bool = True):
        """Complete a tool execution"""
        return self.complete_action_execution(node_id, result, success)

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

    def track_response_synthesis(self, partial_results: List[Dict], method: str = "comprehensive_analysis"):
        """Track how the agent synthesizes multiple results"""
        synthesis_node = self.add_result_synthesis(partial_results, method)

        # Add thinking about synthesis strategy
        thinking_node = self.add_thinking_process(
            f"Synthesizing {len(partial_results)} partial results using {method}",
            "Combining database results, policy information, and logical inferences"
        )

        return synthesis_node

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

    def _get_edge_color_and_style(self, edge_type: str, label: str) -> Dict[str, Any]:
        """Get color and style for edge based on type and label"""
        edge_styles = {
            # Thinking and reasoning flows
            "thinks": {"color": "#2196F3", "width": 2, "dashes": False},
            "infers": {"color": "#607D8B", "width": 3, "dashes": False},
            "decomposes": {"color": "#FF9800", "width": 2, "dashes": [5, 5]},

            # Action and execution flows
            "selects action": {"color": "#673AB7", "width": 2, "dashes": False},
            "executes": {"color": "#3F51B5", "width": 3, "dashes": False},
            "cache hit": {"color": "#00BCD4", "width": 2, "dashes": [10, 5]},
            "cache miss": {"color": "#795548", "width": 2, "dashes": False},

            # Result and synthesis flows
            "synthesizes": {"color": "#E91E63", "width": 3, "dashes": False},
            "produces": {"color": "#4CAF50", "width": 3, "dashes": False},

            # Error and control flows
            "encounters error": {"color": "#F44336", "width": 2, "dashes": [3, 3]},
            "creates subgoal": {"color": "#FFC107", "width": 1, "dashes": [2, 2]},

            # Default flow
            "flow": {"color": "#848484", "width": 2, "dashes": False}
        }

        # Match by label first, then by edge_type
        style = edge_styles.get(label, edge_styles.get(edge_type, edge_styles["flow"]))

        return {
            "color": {"color": style["color"]},
            "width": style["width"],
            "dashes": style["dashes"] if style["dashes"] else False
        }

    def _create_enhanced_node_details(self, node: ExecutionNode) -> Dict[str, Any]:
        """Create comprehensive node details for the info panel"""
        details = {
            "basic_info": {
                "id": node.id,
                "type": node.node_type.value,
                "title": node.title,
                "status": node.status.value,
                "description": node.description
            },
            "timing": {
                "start_time": node.start_time.strftime("%H:%M:%S.%f")[:-3] if node.start_time else "N/A",
                "end_time": node.end_time.strftime("%H:%M:%S.%f")[:-3] if node.end_time else "N/A",
                "execution_time": f"{node.execution_time:.3f}s" if node.execution_time > 0 else "N/A"
            },
            "relationships": {
                "parents": len(node.parents),
                "children": len(node.children),
                "parent_ids": node.parents[:3] if node.parents else [],
                "child_ids": node.children[:3] if node.children else []
            },
            "data": {
                "input_summary": self._summarize_data(node.input_data) if node.input_data else "No input data",
                "output_summary": self._summarize_data(node.output_data) if node.output_data else "No output data",
                "metadata_count": len(node.metadata) if node.metadata else 0
            }
        }
        return details

    def _summarize_data(self, data: Any, max_length: int = 100) -> str:
        """Summarize data for display"""
        if not data:
            return "Empty"

        if isinstance(data, dict):
            keys = list(data.keys())[:3]
            return f"Dict with {len(data)} keys: {keys}{'...' if len(data) > 3 else ''}"
        elif isinstance(data, list):
            return f"List with {len(data)} items: {str(data[:2])[:50]}{'...' if len(data) > 2 else ''}"
        elif isinstance(data, str):
            return f"String: {data[:max_length]}{'...' if len(data) > max_length else ''}"
        else:
            return f"{type(data).__name__}: {str(data)[:max_length]}{'...' if len(str(data)) > max_length else ''}"

    def _generate_html_graph(self) -> str:
        """Generate interactive HTML graph with enhanced features"""
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

            # Create shorter, cleaner labels
            label = node.title
            if len(label) > 25:
                label = label[:22] + "..."

            graph_data["nodes"].append({
                "id": node.id,
                "label": label,
                "title": self._create_node_tooltip(node),
                "color": color,
                "shape": shape,
                "font": {"size": 10, "color": "#333"},
                "size": 20,  # Fixed size
                "physics": True,
                "widthConstraint": {"minimum": 100, "maximum": 150}  # Constrain width
            })

        # Convert edges to vis.js format with color coding
        for edge in self.edges.values():
            edge_style = self._get_edge_color_and_style(edge.edge_type, edge.label)

            graph_data["edges"].append({
                "from": edge.source,
                "to": edge.target,
                "label": edge.label,
                "arrows": "to",
                "font": {"size": 10},
                "color": edge_style["color"],
                "width": edge_style["width"],
                "dashes": edge_style["dashes"],
                "smooth": {"type": "continuous"}
            })

        # Add enhanced node details to graph data
        graph_data["nodeDetails"] = {}
        for node in self.nodes.values():
            graph_data["nodeDetails"][node.id] = self._create_enhanced_node_details(node)

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
            NodeType.ACTION_SELECTION: "#673AB7",
            NodeType.ACTION_EXECUTION: "#3F51B5",
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
            NodeType.ACTION_SELECTION: "triangleDown",
            NodeType.ACTION_EXECUTION: "square",
            NodeType.LOGICAL_INFERENCE: "hexagon",
            NodeType.RESULT_SYNTHESIS: "square",
            NodeType.FINAL_RESPONSE: "star",
            NodeType.ERROR: "triangleDown",
            NodeType.CACHE_HIT: "circle"
        }
        return shapes.get(node_type, "ellipse")

    def _create_node_tooltip(self, node: ExecutionNode) -> str:
        """Create detailed tooltip for node with proper text formatting and NO HTML tags for vis.js"""
        def clean_and_truncate(text, max_length=150):
            """Clean text and break into multiple lines if needed"""
            if not text:
                return ""

            # Convert to string and clean
            text_str = str(text)
            # Remove quotes and clean up
            text_str = text_str.replace("'", "").replace('"', "").replace("{", "").replace("}", "")

            # Truncate if too long
            if len(text_str) > max_length:
                text_str = text_str[:max_length] + "..."

            # Break into lines every 50 characters for better readability
            lines = []
            words = text_str.split()
            current_line = ""

            for word in words:
                if len(current_line + " " + word) <= 50:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return "\n".join(lines)

        # For vis.js tooltips, we need to return plain text that will be displayed
        # vis.js will handle the HTML rendering internally
        tooltip_parts = []

        tooltip_parts.append(f"Node: {node.title}")
        tooltip_parts.append(f"Type: {node.node_type.value}")
        tooltip_parts.append(f"Status: {node.status.value}")

        if node.description:
            clean_desc = clean_and_truncate(node.description, 100)
            tooltip_parts.append(f"Description: {clean_desc}")

        if node.execution_time > 0:
            tooltip_parts.append(f"Execution Time: {node.execution_time:.3f}s")

        if node.input_data:
            clean_input = clean_and_truncate(node.input_data, 80)
            tooltip_parts.append(f"Input: {clean_input}")

        if node.output_data:
            clean_output = clean_and_truncate(node.output_data, 80)
            tooltip_parts.append(f"Output: {clean_output}")

        return "\n".join(tooltip_parts)

    def _create_html_template(self, graph_data: Dict) -> str:
        """Create the complete HTML template with enhanced features"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Query Execution Graph - {graph_data['metadata']['query_id']}</title>
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
        
        .legend-line {{
            display: inline-block;
            width: 30px;
            height: 3px;
            margin-right: 8px;
            vertical-align: middle;
            border-radius: 1px;
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
        
        .detail-section {{
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}

        .detail-section h5 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 14px;
        }}

        .detail-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 5px 0;
        }}

        .detail-table td {{
            padding: 4px 8px;
            border-bottom: 1px solid #dee2e6;
            font-size: 13px;
        }}

        .detail-table td:first-child {{
            width: 40%;
            color: #6c757d;
        }}

        .status-completed {{ color: #28a745; font-weight: bold; }}
        .status-failed {{ color: #dc3545; font-weight: bold; }}
        .status-in_progress {{ color: #ffc107; font-weight: bold; }}
        .status-pending {{ color: #6c757d; font-weight: bold; }}
        .status-cached {{ color: #17a2b8; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Enhanced Query Execution Call Graph</h1>
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
        <button onclick="highlightActions()">🎯 Highlight Actions</button>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <h3>🎨 Node Types Legend</h3>
        <div class="legend-item"><span class="legend-color" style="background: #4CAF50;"></span>Query Start/Final Response</div>
        <div class="legend-item"><span class="legend-color" style="background: #2196F3;"></span>Thinking Process</div>
        <div class="legend-item"><span class="legend-color" style="background: #673AB7;"></span>Action Selection</div>
        <div class="legend-item"><span class="legend-color" style="background: #3F51B5;"></span>Action Execution</div>
        <div class="legend-item"><span class="legend-color" style="background: #00BCD4;"></span>Cache Hit</div>
        <div class="legend-item"><span class="legend-color" style="background: #FF9800;"></span>Decomposition</div>
        <div class="legend-item"><span class="legend-color" style="background: #607D8B;"></span>Logical Inference</div>
        <div class="legend-item"><span class="legend-color" style="background: #E91E63;"></span>Result Synthesis</div>
        <div class="legend-item"><span class="legend-color" style="background: #F44336;"></span>Error</div>
    </div>
    
    <div class="legend">
        <h3>🌈 Edge Types Legend</h3>
        <div class="legend-item"><span class="legend-line" style="background: #2196F3;"></span>Thinking Process</div>
        <div class="legend-item"><span class="legend-line" style="background: #607D8B;"></span>Logical Inference</div>
        <div class="legend-item"><span class="legend-line" style="background: #673AB7;"></span>Action Selection</div>
        <div class="legend-item"><span class="legend-line" style="background: #3F51B5;"></span>Action Execution</div>
        <div class="legend-item"><span class="legend-line" style="background: #00BCD4; border-style: dashed;"></span>Cache Hit</div>
        <div class="legend-item"><span class="legend-line" style="background: #4CAF50;"></span>Result Production</div>
        <div class="legend-item"><span class="legend-line" style="background: #F44336; border-style: dashed;"></span>Error Flow</div>
    </div>
    
    <div class="info-panel">
        <h3>📋 Enhanced Node Information</h3>
        <div id="node-info">Click on a node to see comprehensive detailed information</div>
    </div>
    
    <script>
        // Graph data
        const graphData = {json.dumps(graph_data, indent=2, default=str)};
        
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
                font: {{ size: 10, color: '#333' }},
                borderWidth: 2,
                shadow: true,
                chosen: true,
                size: 20,
                scaling: {{
                    min: 15,
                    max: 30
                }},
                widthConstraint: {{
                    minimum: 80,
                    maximum: 120
                }}
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
        
        // Store original node data for highlighting
        let originalNodeData = JSON.parse(JSON.stringify(graphData.nodes));
        let isHighlighted = false;
        
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
            if (canvas) {{
                const link = document.createElement('a');
                link.download = 'enhanced_query_execution_graph.png';
                link.href = canvas.toDataURL();
                link.click();
            }} else {{
                alert('Canvas not found. Please try again.');
            }}
        }}
        
        function highlightActions() {{
            // Toggle highlight state
            isHighlighted = !isHighlighted;
            
            if (isHighlighted) {{
                // Highlight action-related nodes
                const updatedNodes = graphData.nodes.map(node => {{
                    const isActionNode = node.shape === 'triangleDown' || 
                                       node.shape === 'square' || 
                                       node.shape === 'circle';
                    
                    if (isActionNode) {{
                        return {{
                            ...node,
                            borderWidth: 6,
                            color: {{
                                ...node.color,
                                border: '#FF5722'
                            }},
                            font: {{
                                ...node.font,
                                size: 14,
                                color: '#FF5722'
                            }}
                        }};
                    }} else {{
                        return {{
                            ...node,
                            color: {{
                                ...node.color,
                                background: '#E0E0E0'
                            }}
                        }};
                    }}
                }});
                
                // Update the network with highlighted nodes
                network.setData({{
                    nodes: updatedNodes,
                    edges: graphData.edges
                }});
                
                alert('Action nodes highlighted! Click again to reset.');
            }} else {{
                // Reset to original appearance
                network.setData({{
                    nodes: originalNodeData,
                    edges: graphData.edges
                }});
            }}
        }}
        
        function showStatistics() {{
            const stats = calculateStatistics();
            const actionNodes = graphData.nodes.filter(n => 
                n.shape === 'triangleDown' || n.shape === 'square' || n.shape === 'circle'
            ).length;
            
            const cacheHits = graphData.nodes.filter(n => n.shape === 'circle').length;
            const errorNodes = graphData.nodes.filter(n => 
                n.color.background === '#F44336'
            ).length;
            
            const message = `📊 Enhanced Graph Statistics:
            
Total Nodes: ${{stats.totalNodes}}
Total Edges: ${{stats.totalEdges}}
Action Nodes: ${{actionNodes}}
Cache Hits: ${{cacheHits}}
Error Nodes: ${{errorNodes}}
Average Execution Time: ${{stats.avgExecutionTime}}s
Success Rate: ${{stats.successRate}}%
Most Common Node Type: ${{stats.mostCommonType}}`;
            
            alert(message);
        }}
        
        function resetView() {{
            // Reset highlighting
            isHighlighted = false;
            network.setData({{
                nodes: originalNodeData,
                edges: graphData.edges
            }});
            
            // Reset view and physics
            network.fit();
            network.setOptions({{physics: {{enabled: true}}}});
            physicsEnabled = true;
        }}
        
        function showNodeDetails(node) {{
            const infoDiv = document.getElementById('node-info');
            
            if (!node) {{
                infoDiv.innerHTML = '<p>No node selected</p>';
                return;
            }}
            
            // Get enhanced details from the graph metadata
            const nodeDetails = graphData.nodeDetails[node.id];
            
            if (!nodeDetails) {{
                infoDiv.innerHTML = '<p>No detailed information available</p>';
                return;
            }}
            
            const details = `
                <div class="node-details">
                    <h4>🔍 ${{nodeDetails.basic_info.title}}</h4>
                    
                    <div class="detail-section">
                        <h5>📋 Basic Information</h5>
                        <table class="detail-table">
                            <tr><td><strong>Node ID:</strong></td><td>${{nodeDetails.basic_info.id}}</td></tr>
                            <tr><td><strong>Type:</strong></td><td>${{nodeDetails.basic_info.type}}</td></tr>
                            <tr><td><strong>Status:</strong></td><td><span class="status-${{nodeDetails.basic_info.status}}">${{nodeDetails.basic_info.status}}</span></td></tr>
                        </table>
                        <p><strong>Description:</strong> ${{nodeDetails.basic_info.description}}</p>
                    </div>
                    
                    <div class="detail-section">
                        <h5>⏱️ Timing Information</h5>
                        <table class="detail-table">
                            <tr><td><strong>Start Time:</strong></td><td>${{nodeDetails.timing.start_time}}</td></tr>
                            <tr><td><strong>End Time:</strong></td><td>${{nodeDetails.timing.end_time}}</td></tr>
                            <tr><td><strong>Execution Time:</strong></td><td>${{nodeDetails.timing.execution_time}}</td></tr>
                        </table>
                    </div>
                    
                    <div class="detail-section">
                        <h5>🔗 Relationships</h5>
                        <table class="detail-table">
                            <tr><td><strong>Parent Nodes:</strong></td><td>${{nodeDetails.relationships.parents}}</td></tr>
                            <tr><td><strong>Child Nodes:</strong></td><td>${{nodeDetails.relationships.children}}</td></tr>
                        </table>
                        ${{nodeDetails.relationships.parent_ids.length > 0 ? 
                            `<p><strong>Parents:</strong> ${{nodeDetails.relationships.parent_ids.join(', ')}}</p>` : ''}}
                        ${{nodeDetails.relationships.child_ids.length > 0 ? 
                            `<p><strong>Children:</strong> ${{nodeDetails.relationships.child_ids.join(', ')}}</p>` : ''}}
                    </div>
                    
                    <div class="detail-section">
                        <h5>📊 Data Information</h5>
                        <p><strong>Input:</strong> ${{nodeDetails.data.input_summary}}</p>
                        <p><strong>Output:</strong> ${{nodeDetails.data.output_summary}}</p>
                        <p><strong>Metadata:</strong> ${{nodeDetails.data.metadata_count}} items</p>
                    </div>
                </div>
            `;
            
            infoDiv.innerHTML = details;
        }}
        
        function calculateStatistics() {{
            const totalNodes = graphData.nodes.length;
            const totalEdges = graphData.edges.length;
            
            // Calculate node type distribution
            const shapeCount = {{}};
            graphData.nodes.forEach(node => {{
                const shape = node.shape;
                shapeCount[shape] = (shapeCount[shape] || 0) + 1;
            }});
            
            const mostCommonShape = Object.keys(shapeCount).reduce((a, b) => 
                shapeCount[a] > shapeCount[b] ? a : b
            );
            
            // Calculate success rate (nodes without error color)
            const errorNodes = graphData.nodes.filter(n => 
                n.color.background === '#F44336'
            ).length;
            const successRate = ((totalNodes - errorNodes) / totalNodes * 100).toFixed(1);
            
            return {{
                totalNodes,
                totalEdges,
                avgExecutionTime: {graph_data['metadata']['execution_time']:.3f},
                successRate,
                mostCommonType: mostCommonShape
            }};
        }}
        
        // Initialize
        fitNetwork();
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(event) {{
            switch(event.key) {{
                case 'f':
                case 'F':
                    if (event.ctrlKey || event.metaKey) {{
                        event.preventDefault();
                        fitNetwork();
                    }}
                    break;
                case 'h':
                case 'H':
                    if (event.ctrlKey || event.metaKey) {{
                        event.preventDefault();
                        highlightActions();
                    }}
                    break;
                case 'r':
                case 'R':
                    if (event.ctrlKey || event.metaKey) {{
                        event.preventDefault();
                        resetView();
                    }}
                    break;
            }}
        }});
    </script>
</body>
</html>
        """

    def export_execution_summary(self) -> Dict[str, Any]:
        """Export a summary of the execution for logging"""
        if not self.nodes:
            return {}

        summary = {
            "query_id": self.current_query_id,
            "total_execution_time": (datetime.now() - self.query_start_time).total_seconds() if self.query_start_time else 0,
            "node_counts": {},
            "tool_calls": len([n for n in self.nodes.values() if n.node_type in [NodeType.TOOL_EXECUTION, NodeType.ACTION_EXECUTION]]),
            "cache_hits": len([n for n in self.nodes.values() if n.node_type == NodeType.CACHE_HIT]),
            "errors": len([n for n in self.nodes.values() if n.node_type == NodeType.ERROR]),
            "reasoning_steps": len([n for n in self.nodes.values() if n.node_type == NodeType.LOGICAL_INFERENCE]),
            "thinking_steps": len([n for n in self.nodes.values() if n.node_type == NodeType.THINKING]),
            "action_selections": len([n for n in self.nodes.values() if n.node_type == NodeType.ACTION_SELECTION])
        }

        # Count node types
        for node in self.nodes.values():
            node_type = node.node_type.value
            summary["node_counts"][node_type] = summary["node_counts"].get(node_type, 0) + 1

        return summary


class CacheTracker:
    """Dedicated cache tracking system"""

    def __init__(self):
        self.cache_operations = []
        self.hit_count = 0
        self.miss_count = 0

    def reset(self):
        """Reset cache tracking for new query"""
        self.cache_operations = []
        self.hit_count = 0
        self.miss_count = 0

    def record_operation(self, operation: str, hit: bool, execution_time: float = 0.0):
        """Record a cache operation"""
        self.cache_operations.append({
            "operation": operation,
            "hit": hit,
            "execution_time": execution_time,
            "timestamp": datetime.now()
        })

        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1


class EnhancedTrackedCustomerServiceTools:
    """Enhanced wrapper with proper argument handling for caching"""

    def __init__(self, original_tools, execution_tracker):
        self.original_tools = original_tools
        self.execution_tracker = execution_tracker
        self._wrap_all_methods()

    def _wrap_all_methods(self):
        """Wrap all public methods with enhanced tracking"""
        for attr_name in dir(self.original_tools):
            if not attr_name.startswith('_'):
                attr = getattr(self.original_tools, attr_name)
                if callable(attr):
                    wrapped_method = self._create_enhanced_tracked_method(attr, attr_name)
                    setattr(self, attr_name, wrapped_method)

    def _sanitize_args_for_cache(self, args, kwargs):
        """Sanitize arguments to ensure they're cacheable"""
        sanitized_args = []
        for arg in args:
            if isinstance(arg, dict):
                # Convert dict to tuple of sorted items
                sanitized_args.append(tuple(sorted(arg.items())))
            elif isinstance(arg, list):
                # Convert list to tuple
                sanitized_args.append(tuple(arg))
            elif isinstance(arg, set):
                # Convert set to sorted tuple
                sanitized_args.append(tuple(sorted(arg)))
            else:
                sanitized_args.append(arg)

        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                sanitized_kwargs[key] = tuple(sorted(value.items()))
            elif isinstance(value, list):
                sanitized_kwargs[key] = tuple(value)
            elif isinstance(value, set):
                sanitized_kwargs[key] = tuple(sorted(value))
            else:
                sanitized_kwargs[key] = value

        return tuple(sanitized_args), sanitized_kwargs

    def _create_enhanced_tracked_method(self, original_method, method_name):
        """Create an enhanced tracked version of a method with proper error handling"""
        def enhanced_tracked_method(*args, **kwargs):
            import time
            import functools

            start_time = time.time()

            # Sanitize arguments for safer processing
            try:
                sanitized_args, sanitized_kwargs = self._sanitize_args_for_cache(args[1:], kwargs)
                cache_key = f"{method_name}_{hash(str(sanitized_args) + str(sorted(sanitized_kwargs.items())))}"
            except Exception as e:
                # Fallback cache key generation
                cache_key = f"{method_name}_{hash(str(args[1:])[:100])}_{hash(str(kwargs)[:100])}"

            # Simple cache hit detection
            cache_hit = self._detect_cache_hit(method_name, args, kwargs)

            if cache_hit:
                # Track cache hit
                node_id = self.execution_tracker.track_cache_operation(
                    method_name, cache_key, True, execution_time=0.001
                )

                try:
                    result = original_method(*args, **kwargs)
                    execution_time = time.time() - start_time

                    self.execution_tracker.add_logical_inference(
                        f"Cache hit for {method_name} - data retrieved instantly",
                        [f"Cache key: {cache_key[:20]}..."],
                        "Saved execution time, returned cached result"
                    )

                    return result
                except Exception as e:
                    self.execution_tracker.add_error(f"Cached {method_name} failed: {str(e)}", f"cache_error_{method_name}")
                    return {"error": str(e)}
            else:
                # Track cache miss and tool execution
                node_id = self.execution_tracker.add_action_execution(
                    method_name,
                    {"args": str(args[1:])[:100], "kwargs": str(kwargs)[:100]},  # Convert to string to avoid dict issues
                    is_cached=False
                )

                try:
                    result = original_method(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Complete the action execution
                    self.execution_tracker.complete_action_execution(node_id, result, success=True)

                    # Add logical inference about the result
                    if isinstance(result, dict) and 'error' not in result:
                        self.execution_tracker.add_logical_inference(
                            f"Tool {method_name} executed successfully",
                            [f"Input: {str(args[1:])[:50]}..."],
                            f"Retrieved valid data: {str(result)[:100]}..."
                        )
                    elif isinstance(result, list) and result:
                        self.execution_tracker.add_logical_inference(
                            f"Tool {method_name} returned {len(result)} items",
                            [f"Query parameters: {str(args[1:])[:50]}..."],
                            f"Successfully retrieved {len(result)} records"
                        )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.execution_tracker.complete_action_execution(node_id, str(e), success=False)
                    self.execution_tracker.add_error(f"Tool {method_name} failed: {str(e)}", f"tool_{method_name}")
                    return {"error": str(e)}

        return enhanced_tracked_method

    def _detect_cache_hit(self, method_name: str, args: tuple, kwargs: dict) -> bool:
        """Detect if this call would be a cache hit with safer signature generation"""
        try:
            # Create a safer call signature
            sanitized_args, sanitized_kwargs = self._sanitize_args_for_cache(args[1:], kwargs)
            call_signature = f"{method_name}_{hash(str(sanitized_args) + str(sorted(sanitized_kwargs.items())))}"
        except Exception:
            # Fallback signature
            call_signature = f"{method_name}_{hash(str(args[1:])[:50])}_{hash(str(kwargs)[:50])}"

        if not hasattr(self, '_call_history'):
            self._call_history = set()

        if call_signature in self._call_history:
            return True
        else:
            self._call_history.add(call_signature)
            return False


# Global tracker instance
execution_tracker = EnhancedQueryExecutionTracker()

def integrate_with_agent():
    """Enhanced integration functions"""

    def enhanced_run_query(self, query: str):
        """Enhanced query execution with complete tracking"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        # Start tracking
        query_id = execution_tracker.track_agent_query_start(query)

        try:
            print(f"🤖 Processing query with enhanced call graph tracking: '{query}'")
            print(f"📊 Tracking ID: {query_id}")
            print("-" * 70)

            # Track initial agent thinking
            execution_tracker.add_thinking_process(
                "Received customer query, initializing response process",
                f"Query length: {len(query)} characters, analyzing requirements"
            )

            # Track tool availability assessment
            available_tools = [tool.metadata.name for tool in self.tools] if self.tools else []
            execution_tracker.track_tool_selection_process(available_tools, query)

            # Track agent reasoning about tool selection
            execution_tracker.add_thinking_process(
                "Agent analyzing available tools and planning execution strategy",
                f"Available tools: {', '.join(available_tools[:5])}{'...' if len(available_tools) > 5 else ''}"
            )

            try:
                # Execute the actual query
                response = self.agent.query(query)

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
                print(f"📊 Enhanced Execution Summary:")
                print(f"   Total time: {summary['total_execution_time']:.2f}s")
                print(f"   Tool calls: {summary['tool_calls']}")
                print(f"   Action selections: {summary['action_selections']}")
                print(f"   Cache hits: {summary['cache_hits']}")
                print(f"   Reasoning steps: {summary['reasoning_steps']}")
                print(f"   Thinking steps: {summary['thinking_steps']}")
                print(f"   📈 Enhanced call graph saved to: ./generated_callgraphs/")

            except Exception as query_error:
                execution_tracker.add_error(str(query_error), "execution_error")
                execution_tracker.finalize_query(f"Query failed: {str(query_error)}", success=False)
                print(f"\n❌ Query execution error: {query_error}")

        except Exception as e:
            execution_tracker.add_error(str(e), "critical_error")
            execution_tracker.finalize_query(f"Critical error: {str(e)}", success=False)
            print(f"❌ Critical error during query execution: {e}")

    def enhanced_run_intelligent_query(self, query: str):
        """Enhanced intelligent query with complete decomposition tracking"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            return enhanced_run_query(self, query)

        # Use the enhanced run_query for intelligent queries too
        return enhanced_run_query(self, query)

    return enhanced_run_query, enhanced_run_intelligent_query


if __name__ == "__main__":
    print("🔧 ENHANCED Query Call Graph Visualizer")
    print("✅ New Features:")
    print("   - Comprehensive node details panel without tooltip duplication")
    print("   - Color-coded edges based on type and nature")
    print("   - Enhanced visual hierarchy and information architecture")
    print("   - Improved data presentation and analysis")
    print("   - Better relationship tracking and timing information")
    print("   - Structured node information with tables and sections")
    print("=" * 70)