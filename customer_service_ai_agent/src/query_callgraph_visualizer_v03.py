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
            description=f"Running {action_name}" + (
                " - retrieved from cache" if is_cached else f" with parameters: {str(input_params)[:100]}"),
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
                title=f"Subgoal {i + 1}",
                description=subgoal.get('subgoal', '')[:100],
                input_data=subgoal
            )
            self._connect_nodes(decomp_node.id, subgoal_node.id, f"creates subgoal {i + 1}")

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

    def _complete_node(self, node_id: str, output_data: Any = None,
                       status: ExecutionStatus = ExecutionStatus.COMPLETED):
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

    # ADD THIS NEW METHOD HERE ↓
    def _create_shadow_node_data(self, node: ExecutionNode) -> List[Dict]:
        """Create shadow nodes that show detailed breakdown of what's happening inside a node"""
        shadow_nodes = []

        if node.node_type == NodeType.THINKING:
            # Create shadow nodes for thinking process breakdown
            thought_data = node.input_data.get('thought', '') if node.input_data else ''
            context_data = node.input_data.get('context', '') if node.input_data else ''

            if thought_data:
                shadow_nodes.append({
                    "id": f"shadow_thought_{node.id}",
                    "label": "💭 Thought Process",
                    "details": f"Main thought: {thought_data[:150]}...",
                    "type": "thought",
                    "parent": node.id
                })

            if context_data:
                shadow_nodes.append({
                    "id": f"shadow_context_{node.id}",
                    "label": "🧠 Context Analysis",
                    "details": f"Context: {context_data[:150]}...",
                    "type": "context",
                    "parent": node.id
                })

        elif node.node_type == NodeType.ACTION_EXECUTION:
            # Create shadow nodes for action execution breakdown
            action_name = node.input_data.get('action_name', '') if node.input_data else ''
            parameters = node.input_data.get('parameters', {}) if node.input_data else {}
            result = node.output_data.get('result', '') if node.output_data else ''

            if action_name:
                shadow_nodes.append({
                    "id": f"shadow_action_{node.id}",
                    "label": f"⚡ {action_name}",
                    "details": f"Action: {action_name}",
                    "type": "action",
                    "parent": node.id
                })

            if parameters:
                shadow_nodes.append({
                    "id": f"shadow_params_{node.id}",
                    "label": "📝 Parameters",
                    "details": f"Input params: {str(parameters)[:100]}...",
                    "type": "parameters",
                    "parent": node.id
                })

            if result:
                shadow_nodes.append({
                    "id": f"shadow_result_{node.id}",
                    "label": "📊 Result",
                    "details": f"Output: {str(result)[:100]}...",
                    "type": "result",
                    "parent": node.id
                })

        elif node.node_type == NodeType.LOGICAL_INFERENCE:
            # Create shadow nodes for logical inference breakdown
            reasoning = node.input_data.get('reasoning', '') if node.input_data else ''
            evidence = node.input_data.get('evidence', []) if node.input_data else []
            conclusion = node.input_data.get('conclusion', '') if node.input_data else ''

            if reasoning:
                shadow_nodes.append({
                    "id": f"shadow_reasoning_{node.id}",
                    "label": "🤔 Reasoning",
                    "details": f"Logic: {reasoning[:150]}...",
                    "type": "reasoning",
                    "parent": node.id
                })

            if evidence:
                shadow_nodes.append({
                    "id": f"shadow_evidence_{node.id}",
                    "label": "📋 Evidence",
                    "details": f"Evidence points: {len(evidence)} items",
                    "type": "evidence",
                    "parent": node.id
                })

            if conclusion:
                shadow_nodes.append({
                    "id": f"shadow_conclusion_{node.id}",
                    "label": "✅ Conclusion",
                    "details": f"Result: {conclusion[:150]}...",
                    "type": "conclusion",
                    "parent": node.id
                })

        elif node.node_type == NodeType.CACHE_HIT:
            # Create shadow nodes for cache operation breakdown
            operation = node.input_data.get('operation', '') if node.input_data else ''
            cache_key = node.input_data.get('cache_key', '') if node.input_data else ''
            time_saved = node.input_data.get('time_saved', 0) if node.input_data else 0

            if operation:
                shadow_nodes.append({
                    "id": f"shadow_cache_op_{node.id}",
                    "label": "🚀 Cache Operation",
                    "details": f"Operation: {operation}",
                    "type": "cache_operation",
                    "parent": node.id
                })

            if cache_key:
                shadow_nodes.append({
                    "id": f"shadow_cache_key_{node.id}",
                    "label": "🔑 Cache Key",
                    "details": f"Key: {cache_key[:50]}...",
                    "type": "cache_key",
                    "parent": node.id
                })

            if time_saved > 0:
                shadow_nodes.append({
                    "id": f"shadow_time_saved_{node.id}",
                    "label": "⏱️ Time Saved",
                    "details": f"Saved: {time_saved:.3f}s",
                    "type": "time_saved",
                    "parent": node.id
                })

        return shadow_nodes

    # ADD THIS NEW METHOD HERE ↓ (right after _create_shadow_node_data)
    def _extract_query_and_response(self) -> Dict[str, str]:
        """Extract the original query and final response for header display"""
        query_text = "No query found"
        response_text = "No response generated"

        # Find the query start node
        for node in self.nodes.values():
            if node.node_type == NodeType.QUERY_START and node.input_data:
                query_text = node.input_data.get('query', 'No query found')
                break

        # Find the final response node
        for node in self.nodes.values():
            if node.node_type == NodeType.FINAL_RESPONSE and node.input_data:
                response_text = node.input_data.get('response', 'No response generated')
                break

        return {
            "query": query_text[:200] + "..." if len(query_text) > 200 else query_text,
            "response": response_text[:300] + "..." if len(response_text) > 300 else response_text
        }

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
                "execution_time": (
                        datetime.now() - self.query_start_time).total_seconds() if self.query_start_time else 0
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

        # ADD THESE LINES HERE ↓ (after nodeDetails creation, before HTML template)
        # Add shadow node data to graph data
        graph_data["shadowNodes"] = {}
        for node in self.nodes.values():
            shadow_data = self._create_shadow_node_data(node)
            if shadow_data:
                graph_data["shadowNodes"][node.id] = shadow_data

        # Add query and response data
        query_response_data = self._extract_query_and_response()
        graph_data["queryResponse"] = query_response_data

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
            background: linear-gradient(135deg, #b8c6d1 0%, #c7d2dd 100%);
            color: black;
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
        
        .query-response-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}

        .query-box, .response-box {{
            background: rgba(255,255,255,0.15);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}

        .query-box h3, .response-box h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: black;
        }}

        .query-box p, .response-box p {{
            margin: 0;
            font-size: 14px;
            line-height: 1.4;
            color: black;
        }}

        .shadow-node {{
            opacity: 0.7;
            border: 2px dashed #666;
            background: rgba(255,255,255,0.9);
            font-style: italic;
        }}

        .shadow-edge {{
            opacity: 0.5;
            dashes: [5, 5];
            color: #999;
        }}

        /* Animation for shadow nodes */
        @keyframes shadowFadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 0.7; transform: translateY(0); }}
        }}

        .shadow-node {{
            animation: shadowFadeIn 0.3s ease-out;
        }}
        
        /* Developer Information and Tool Description Styles */
        .developer-info {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin: 20px 0;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .developer-badge {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            min-width: 250px;
        }}
        
        .developer-badge h4 {{
            margin: 0 0 8px 0;
            color: black;
            font-size: 14px;
        }}
        
        .developer-link {{
            color: black;
            text-decoration: none;
            font-size: 16px;
            line-height: 1.4;
            transition: all 0.3s ease;
        }}
        
        .developer-link:hover {{
            color: #a8e6cf;
            text-shadow: 0 0 8px rgba(168, 230, 207, 0.8);
        }}
        
        .tool-description {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            flex: 1;
            min-width: 300px;
        }}
        
        .tool-description h4 {{
            margin: 0 0 8px 0;
            color: black;
            font-size: 14px;
        }}
        
        .tool-description p {{
            margin: 0;
            font-size: 13px;
            line-height: 1.5;
            color: black;
        }}
        
        /* Responsive Design for Mobile */
        @media (max-width: 768px) {{
                .developer-info {{
                flex-direction: column;
                gap: 15px;
                }}
        
                .developer-badge,
                .tool-description {{
                min-width: unset;
                width: 100%;
                }}
        
                .tool-description p {{
                font-size: 12px;
                }}
            }}
            
        .callgraph-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .network-column {{
                display: flex;
            flex-direction: column;
        }}
        
        .execution-image-column {{
                display: flex;
            flex-direction: column;
        }}
        
        #network {{
                width: 100%;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .execution-image {{
                width: 100%;
            height: 70vh;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            object-fit: contain;
            padding: 10px;
            box-sizing: border-box;
        }}
        
        .column-header {{
                background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            font-weight: bold;
            color: #333;
        }}
        
        /* Responsive design for mobile */
        @media (max-width: 1024px) {{
                .callgraph-container {{
                grid-template-columns: 1fr;
                gap: 15px;
                }}
        
                .execution-image {{
                height: 50vh;
                }}
                }}
        
        .pseudocode-section {{
                background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .pseudocode-section h3 {{
                margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        
        .pseudocode-container {{
                background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .pseudocode-container::before {{
                content: "/**\n * CORE AGENTIC AI SYSTEM ALGORITHM\n * Autonomous Query Processing with Intelligent Tool Orchestration\n * \n * Author: Jahidul Arafat (ex-L3 Sr Solution Architect, Oracle)\n * Presidential Fellow & PhD Candidate, Auburn University\n */";
            display: block;
            color: #28a745;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 10px;
        }}
        
        .pseudocode-highlight {{
                background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
            border-radius: 5px;
        }}
        
        .toggle-pseudocode {{
                background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        
        .toggle-pseudocode:hover {{
                background: #1976D2;
        }}
        
        /* Syntax highlighting for pseudocode */
        .keyword {{ color: #d73a49; font-weight: bold; }}
        .function-name {{ color: #6f42c1; font-weight: bold; }}
        .comment {{ color: #6a737d; font-style: italic; }}
        .string {{ color: #032f62; }}
        .number {{ color: #005cc5; }}
        .variable {{ color: #e36209; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Enhanced Agentic AI (LLM+Vector Embedding) Query/Goal Execution Call Graph (Autonomous with Intelligent Routing)</h1>
        <p><a href="https://www.linkedin.com/in/jahidul-arafat-presidential-fellow-phd-candidate-791a7490/" target="_blank" class="developer-link">
            <strong>Algorithm and Simulation Designed By Jahidul Arafat (ex-L3 Sr Solution and Cloud Architect, Oracle)</strong><br>
            Presidential Fellow & PhD Candidate, Auburn University (R1 Carnegie Research University), USA
        </a></p>
        <p>This visualization shows the execution flow of an <strong>Agentic AI Customer Service System</strong> that autonomously decomposes complex queries, selects appropriate tools, manages database operations, and synthesizes intelligent responses. The system demonstrates advanced AI agent capabilities including reasoning, tool orchestration, cache optimization, and adaptive query processing.</p>
        <p><a href="https://mermaid.live/view#pako:eNq9fdtuI0mW2K-EBdhF7bDYkqqreloYtcGiqCqhRUlNUlXTOzVIJDNTYk4lM9l5UZW6UcA8-MFY2DBgL_wwfvDaMOAXf4D9O_sD9if4nBP3zMhkStO7DbRKIk9EnIg4ceLc45e9IAujveO9IvqpitIgOo39u9zffEgZ_Lf18zIO4q2fluyG-QW7KaKcnadllN_6QdQEms0QaubHKZtFacUmWVrmWZJEeRN2MkbYSVWU2SbKF1F-HwfR-C5KSwfs2RsCztLb-A76T_07V5cXVwR2kd3B12zxUJSRYybTJQJNP0dBVcZZusz94KOrtx9OEe6HKsofTqMg22yzwgU2f49g82gclO-z3NnTeI4gNLd5laYukOUCQZZZliyiJArKzLVii6VjybBN4ehwQsB8dmG_NlPRaJqufaCFUDRugZ7QZk_8YB217sjyZinndVMASOtin76m3k790l_5RXuHk2tBCCksEuzeNfTsgHpNUHEeVHH5Oo9855gXFzNOLoGfwO9fXW2jdHx-EX90UPZ0Rl1ON6soDOP0bgbHxjHwu3OEekcbuID_o_M0jD47aGuqaGua3sWpY8Qxrdy4eEiDti2e_p4Wdw0TDHEhOE27SGcxnaneFtHG364BtybYNUFdw_HO8g2SwCxLY2eHbyYI-cbPV7BPEzzibpo9I7o-i5Oo7Ti-p4V9H61e59knOmEc5jIrI5bdw0G-Gb5_fcz-33_9y5_Z4sfFEqZyfnm-PB9fnP_teHl-dcmu344XUzY4jUofBgrZ6ywrizL3t-w6z4KoKPZlpzfPv_tuNjtm24dynaUsEMfCK_i58Hw8o6PtA4eezQQ4kHjw0fPT0IvTovSTxAsjIJYQOGYcFYN9Ds5_Jlm2ZWdZzqZwMoAv_FTFOeB0DYQPvXMYq_Myf2AxcJe8hHWpAflJKVuyWVwUQHn6S6sXiZjoYiD-3W8BL6rVlq_NaBtvR6K1BobJ8T_UL8aSADs-hnn6IRtF6T0LiC1XuY8HkoMBhISDvS4i6OY-zrN0A8vL7v089ldJVDRg7_0kDn3Yd6tL_LhyQBdRya4WPbsO4JBgx8iuWAhbgvQam4Byr62hcZZRcwXgpqHxq62XwG0D2yKJAL6R34shb4EmGTDUUF2DTRgYtMg6wHCqMBBLovsoKRhAMTqiJVzFhYYWUxAoqak4JjAZHzvvXjmNyRiAgCkf11nyYJ0V5ZBVcGKGsLdF8SnLwyELV6IhtMHu4cDa7HdwC2cTUPFKYFfFOkvCk5dDVsabKKvKk1cHctzXz-XAAW_PVrwDoPC4jIFAfpbT4UMRrP6O_YRMlQE5l3EBTKbQsGLaoZgQ2_AZiT2w1oimP10eq6uQs-qazCAXa7oU0AYaKVwQcLQBDneiMJifghZ7H8leGUhe2zWgnldBWUn-rMCJ3Njb5eyCwU5FgkCh220C3RQKWkxTd1tybBleEw-NWU5g-cxLXFHATHxpTGoTAdsMnxfbKIhv44CfpqIGz_G8S7KVnxj7oBajBi5WYe0Xazy57GP0YM7PXLmJ2kN-jOUGuicGssdxQ_KQs4MvJYgxPxAEErYGfDMioSxXzNiA5_Pb6ksSjqOfPGhaQ1iBZwnD19e_7Xr7-3_HTsfL8Wu8yyZXl5fTCV1ui-ny5poNtMjD8KpHouL3nb7c9JnlnEkS-mC_cV4025GdbhFTBye3TvX1MZs9LH64sOWvgTGjPL5blyy7ZQiLfXqpv4lOnjVuWvzu2e9W-VffEVQBy3_yUv8NPCIqARiuvCw9WeZVRN_5IN6AKL6JS_3Zqrq9jYDH6U-CNd455cmzqrz97Wb19TOxPtdiDsaOv9TsQC9GUYOH01WyCK9zDaNA1JLW1rJBlMbyY4ceZ1Qn7MPeYnoBmw2bfnO5HPzNPjubX80YsFXk7Xv1DbiDddFjqdMqcS1IfWD-PZAGnScDr9s82xByXchnqz_Bbw2UOTuJBhr1_TpmOVBdnnYht_Hzj-ZYIPgpRNv5ND9wqIElURm1H6D_8Z9JpP9XKKxPT0_PL9_Iw_PWz8NPfh6xqy3cOPHPRN_2ucERxYWebKxbEL8BeQ9u2g0I6l4kVQD4DCnZlv5QYJv5AfsUl2vQgUGoMsVptlj7obqxrRGA2QTr0Qo5RRoWo822GMWFp1ZnsO9owxEgEgJ4SSr1KbHrH5dX88lbb3a98N6ev3nrvR8vp_PZeP69N0cB-uRgdCCEvQSWe3JzOmZje1scqAZV6D8KRWwgceQDXd-wM-C4OOkdTbeVapk2L2rY9WOmtTcnPyKYDWptJ5zLjWCfPfqAc5Zt7CG1md9WOf8O71i8j-BONr82Piawjf_ZKzPYv8IaQn3K-xISj9kP_0j1AecIJdN6J-JjIe1dzOTE4UDC8seFyZ1gj0AKhOuMwYFYlFUYZ7qdWF88K0YL1Y1LEprBAXtbkVR55geRUoOdS03QtLKC-RcgWKJx6TnchGmBhwHOwFew889nwIoBo4tXz--PnqkFSKL0rlyfvDw8oo84IZzgEQxAThPnjq9mDvcKrMwGEPDQlEWXgBCFZq8l6mH2KSV9hbDCxeECRHzL0ihSAr7RwoBGIYQLPKDG8bE1uBS25IKINi2iFnISYqC4pw4-qYhZMrvBh723UZJkH_b2G_tHtxLck1u4sCLVgbG5BFBUAWp5t1XSzjj_01_Y4ub6-mq-ZKdXk5vZ9HK5YNfzq8l0sUAuOuAGDUbGDDZB2aGDfxbVFpVZL8yCCpWywpreDb_BJBAqR4WHF5OHKkWTmU7ESrAF6ALwk643uWFcpRejf8tEp0yNDIp7BsRn8JzXfgFy686-XrAVAXb0xLUn0YkSCOy9prua4GiiXiGBTV6pBuWQakwGh7FsZ3rYQurL9nIO6GeHUWIh1gnNMg60ydrB1dboM-CAR9cBppR1AQlwKKyxwaj8DBriaBvewk-YzZBFZTBy3Q6wPrw1in_tU313fswWMdLBqVDcH-YRXqSDON1WYtYnhI9nzv3duWiMp5nE4S5TzalYd40ntT9bHNNx5ojCoS0tmLPFczGI2rcS1qLWCX4vZG45jCVo1WYNbeSR0oepcBgj9PIIFrvYJjEaBQbBuko_crH68ODo6yHjH-C5T_ztycuD-hohraLSKk6jPr0mCo9fPuy6QKRA6ga-K6-Con2BBIpbsmAg0i7YjQ_doZ8BVFIfd3bHOuKekE5etKr72G3dcqsmT03NJaNbgtg-1wxoAEK3fYmWCDPRMLU7h6QEVEpsAjIghF4cGffNvWF4NS4lnEw7kN4ZnKn4msusPddTNInpRtih2P6Xf_i___s_sOnl2_HlZHrKlldXF2wyn3Lb7QAZfB6tkXncR6Svs_c53_7mDcP3zkO12r5WJovlsdMzMghXnrATSPjFUjYwFEGavFI58ghVSsX3sInAgL6m8TvsRhPDctThgRmgOZ5PZqjtNZ6wFyjj0mSpOv2ERm1Ob7hQ3B7TQXAINDOAGh2KM6cMVWEUZEBhFkWZ8FtUzHPYqAzEvTgFAfO2SrkAWcR3qa_tVo2mfhhaVpMN9y1YJu1GAyGnAdO4M60STtgoz2HiZEJVndYImNpImU1skNhNt8g2hr7Ja7KDtoZM76XYuDFZjKa_P3Z4aAYo5n4id2Fx8rW0SU1__1wMGQk4EyvqbzEF6dDHseJspDw5g0PJzuF72Uchv611IqZPnXTOXR84LgBz26ve8FK7pZwWIDQxk6wIdHX48jdWJ9T2mGT4f_zzf6fLhwwenjAibLMkDh6s7zdVUsYgA3DAwtv6OagQQn1DoMK_jeiEAJUVaB9naM7TB1YB-uE933lluKsB3EUZ2WJBAiSQIq5DwDkIY1iGe5xw8VGBGQBZWAWlRfGNBbNP65lc2KUy0Tj3wpim2gwHOB4Jko-kpNYO5jo5rqElhojgiIsJ0a0P-6KYsUt8-2Gq5Ri6MEZ-wU1IXkTOT9n4h6kkTm4b49-a1FlHyPCgEq_DTZdaQBH5ebCuYSOaH76sH3_FzDusS3BnTZZs_Ab0I-MCI78JG8NYMSqowABBGgNFqPX28huOlvn7Yx5BQH3xMAK-vvymcx0vbEPfnsiZeJpEQb880d4fVDfpY5jPKtM2DkLEEx8q7RtmwVltcXLILbJwjLJP6rzRMB6ayouTMz-Ruif_OX8vpmNcrTDrAng9cIHI8HUrSG7eK7nN9A70jfyhBqFmwhRuoBxt4rJQgJKYaS84Y21yNTLORRs07k9TNFmF7aQ-Wfslh31N9uUBSWceDXtydHBwYOszovGG985N0raAQEqjGP00LtqGFz2cwG6nHSrReH5sBpQM-FbyiQ9FJyf8nyGT2462EXk5zUUnxj7xxcuNCBUFJU2wKXRV8A3gF5-W-AHW2gTeT5clZOuXwZqfBg8OrieFEOto4OgEyDzQRLdC5OHcPdDuLcW26mhQ24g3sCFNDnV6XA_0ESvKZyH506mARbaHtmBupvkclw84GaDNtBYMYB9ZbDpRTZhsoq9BuuI86Supfdy4YvjHaD28zf1NVO8lx8MS-44rq9nVqipgPkXhgZbtS6VUTdpiyqFaol1y_3_8X-x0uhyfX4DQ_8PNdP6jaVN6fzX__uzi6n09MkP4L_hY2VZfWsptDVvCb48BWowsX0TNeUv7LQiM3zdF6eflwGyi3KtKt6rSGAA4Bl4cup22eZaVhosV1cMaIO4LDLfZao90XtZgaB-UzxrdVrD4sGmG41o7dJWMzo8sMBY2kEjuO0naL9CBJqauadWav6JpDgWMHnbwhP81or8GdUiD6osA5csTdtAu1VxzKmfLh62hGejOuLEJoMVxQAcwxhQU9SgY_u1ZVqWhHdTSjthvTthhC6yyJstTOEKBKg0H4m-vBHz3dwfE1CYC0lSMFAGKEYg6cYCqVFGDxNkBrhVwp-_Yi4N_3YW3aiSlX9yZgm7nR7bLeSQS2XQ6m9aPvQHKCQq7-KWNv8WFpDTQleFOj_yU-2RrA-LNUwrzfm0njsnySd_pNTQ-FDMpPMmKaMXVcPUbn1sOOX9nAz7b37Gjuj1yyrVIr1zHKR4zT8RHDT7siR4k_-PoKpO8dakJNykymjT081AcPpja2rT5amY5GQ_xilsIeMA2vUui59ew1DpGVTesDYlNW3ihur9r0ph5iWtIJWvxcC0-VbLYaemNRxy5cKHWy4ViaNopaJtqSMtdCNgN3epqJLKcZqaOZEHnfvpRiOqrB0AqiUiHsyDFFBBUSMOetqE7MNZ3BMEq67za-to0hqy-wPVrodaPvhY0K-cibxIZwospCDq2RN5MntoKEoXMRVXTcVKwarfvGofYNVdizpXoz9BCAWpSLX5QC-2AgNYUHIyWIG1flkYfENtsy1rf9l-Gg7N2S6IF3rSP1qD5KMxfZaD5rtAFBow9z3ypCjqaCJmDVBAMrUNNZxOVlqPecL7h_LnsiRwoLqyD1JyJtTeinXYEaRLjXwGRFcIvJT4ZMtf-uajPd1GdTXl-C9k10UbWSTr1BK0oMo7VhjGmN1lq5ocLOShl8M_QWM_9ZvvmJ5o5YrdDCj-R1lSOkmKM3Gx7VgGGS0u47x5BmA4pnCtHi2eolVohIxrocycT2ob3W7uqsRJtzG309sjFEP1jpBx5E9Ee6oGIZHbp53dANh8_4b-OTlWonbfxP0aejLQbPBJ8d_9KhJ6dvqSAPiestj3LuTxyHXg0NrUeqD56LSY5lMmi_DYum1_bcYkkzy2XF9x8BoJLawN7RiEGr4HU5QZv0AufQbYVrHSwjsuTEhT1IQrCJ13z20nJivNaeA3FwsuBunqWBpjd82p-yr3ktNoYut5z9bxNK2yvpbtFi1S_tescBSOYaRGRw3giFtkTscj88Enba0v3k9eyG6IkGc6MEbHRYxAiohWNJwko3qEbTo-5MHix9NNwptzJfLrRqImti-UQI_pkdDiGXwlZgFhyM1z1KeNxV50Zfyg0_-KnRHDSronw0MRrzPfQVnhP-J0EZ97ZXBLCQGKwY9t3zOh1Zxins4kd2mnI1v3ak_aMt-ZbzABScaGTWihrx_ByFVQLz4xdcP1HZ_-aX9VrvyrKLqK1JvnJj7k10JicjJl7cbD_T4FrGj59J3GoKi8ytJnoMUf8s4EVmrzfu6-RDLbtR-aqNefPBaLC-7mNQMVC2n3E2AGymN0NnPG--tc-9IzRwEZrdySpY3TJ4rhqLSb9dPZCfQlVSfY2sPre7-5CrJ_oIuKRRv22C5iAzIshg9DumfOrUkf1dV7K3dM3RSq6KnlXWqYaMikxKGNmF34qAQSDSvi9WZbJ4MXBAeg7sM_h0y4dgWYbG0ebXUe_1_IKAB4eVJgpYwjnOK1ejastD7NT2SyDv2YqO6S1pphGQs3-zn5blZBG1_YSPHouXLQTMgkGXvej2poUhZJb6pHvudfUUH8m6MGHvUktOQy7-rA3xFhxexDLTPeo_bilZNLWsVq6bb1NdkvZpUh8cR5nZ78t2jjZ8QRDMzoVu7_fqsPzJU6yuxgWwItTEQMl2imvUW9Va07t2AJuwTiIrdhAJ87K0HVLEquMbW47a6odbYy2Rnlo0XI1IrK9jKKQzdAIfJ5yi2arCGQ6oMs4rQwL3E6chLCWRp9LLnrvP2lPCWUe9YrLacR6d6-gXDsezPkxzT4BLd85l7LbdvhhT4wue1RjhI0zYM3H-sOyVAq7tb3JltHa8jwYs6l3yPkQ9yzYBv7vTh5l4Zd92Cb-oXS1ycg45WzoNv6jNTtJYsPvWLf_t3gC0IFyrhuLKZ2aI8uU9laXAHduCR-tUJfqZmvyDtnGWWt6TQNtDWE7rcZGkLflTu0Pe5T-SwkYDNa9kN4k08Pw8vnhAUHLvNKhMFOSnlJUq7sM7r_RaGRmOOnfWi3FtUTaGmwcwgrHtw-UcEr4p-gac8IqwxaiqjCyQcXSyy9badYOJCB3i-3I0nxP_rLvaAl8BZ3FLK02K1QxHFhp12TiFwVOdfXA0K3JBiihglSQCCv0UMWvURxjuY5sNq96wn7ucIvjLMcQBgrFc80rQt4aYGICYeXlfgrcx1o-J3UJqzM7iz7BvcEB2eB3rO6IMEb6vMXCDLLXAWj1oFuftDbAeDcdEEeOjKIDNMizoniu4oF3wv9UQd_lgzD61KCJX-H0Zn76YMzvO3Z40DpBlJwjZOX1Kba3AWpaYeRcEW9Ao8oprpnp7e6cxCaCztHBRhEGKvyxPhM3d9deY3X49Nb84Y-tTEuyZs4treMwqHuOhy4KqrtC7BPl8sPxFEOTcXX44pr-WqqXhD5VuY11p8SE-6nY6ypOQsv-5bg5tOZuLpKOSRN92V_XXHhGeg9HaBCfHKJL9bJGJt2X4e2HPXHFUOEC0dcv8ZfGde_o1UAWzoFKcV7hGnjii4Hocygh95uOJzRVXefRfZxVhVxJhzOhY0iYxmsflHGGkQG_CIAvI9jJT_C3QOGLeaM4l6cpDKOwwHcjToEct5mIS99rm09DymtZN0_FDlC4gbi661Pr3IMGmRI9yD28zTCOs2C36C7j0XFRM8KgBUMdabADIxVzIAhKDt8P7_n7IdcVF_4GT7Go6KMjrYAwCjMcw19Bux3uVxnEKO8ih4qlhU4JpfPNe-ydOL8yhugXReLKgALX7pf97hMjWwt1q6g2G7-xvDuoE8h-oY-sEu9CJM-G-tfCxBtUJPU4KRmw67XvFmw0MUsx4udI2dSUK4J2TuJTuMR0bsVQSoySScTaFKjoy41C94Otizjz4NGMipKr4LRCFHLlnaNspBnybZxgPh0CnzyryU3PXFqAvDLb-pDfOxurmba1VgDPuicsWa6RXOVUqM08AGkxuo8k8bVCagHdQ8PJZgO0xCNHBvudt7yp0xGVq6219UFneLUmqrAmX9ei8iwSxmG5JiWSiYmPpKEKGWBnOLQoC-GMJL3lAJIp26gOpTnWdKdqgUTZHcus9JMW62M9JNQAE2Fxg7aAVc-0itb2zRm8SvopKg-40etyk3iU7NJEBfYZJSePs-GCCqCRYIj7vI63hbNJkMTBR6IMNY5BFkatpCCpQhv7DZY4CAoH9piFW_gS3zLzlBWCjHU0gcKTdb3U_Hl2Lt9BzL-GHlAelcbz5MG5TKBSYK6KEcRhr6qjyJK6LrjdWzRoJp_cHKvCXZqE2W-YYxU68vX_jl2P5-OLi-mFGVJ9djVn099fT-dLtphML8fz86uF7CTblmyGoaDPr1BlA2EVRizZWPCieljh8rgz5Wrwh8ODg2-GoLUc0s-jF380ow51olpAUUNGKtqAJ1BYqosrZ80TTZ0XFA0w1vTs89w7v_hocR-dgCcT5mT6KEBSyDFlUtZ0WP0bzJZdixkzvmxaMq6JD2ooIcQxGqMWE4cJfsLfpBPexOEa4IrW7_qG-1tnqHLppzaAXxc1W_A6ehReh4_H6-hpeL14FF5HLx6N14udwo-iLR890Ypy7ny4ePLB3xCR7btIkavcOktKSj3Og4Hxq7y0kVWL087QFF2GcuKWV7FN8PiZslJ4OwcKCliav1TUNS9vWRRV5G6grhvkdChLmYkkg8Z1fUNsFHiMRzULOSsRM_EtrkMb0crq_i0DhgacbYbJ42x2dXm-vJojtxtfnrJ354sbVQbUZHVmzaUZZ6cUwhiZ0cA3UqpYY3Zb-zXkWm4RjicVFPIM2qug4sjU_c9L4qFDz8OFdELLPeRpTp60iJroOdsZhv8Azc7kwRKV7GoNDOO6Fg25r5Fm0TppcpPxuBVbqsTb0grx1sXy9OxFLTeXZGnAy_lzt5Wffqz5WgxQp_zjIkiz1p-k-ki4pIJm5lathJ_aYkKpwlqCYsqtC0UcCBdKJR05vMZmNT5YTxRP6v7RBlEZbfRhV_ETW4VgvU2toprnlDbcxtjrxc3y_MJVD0xN91rMlpdqRK6TVXkQNXzkypsuK8_y_ETUXrOdsMG2QrSDqBkk0YBNqw2VGfVDFw5Kl7Cw5Qu383owdHaxZ9ZZcHCOJgM1V79DeVJ8VFGg2TD0i_Uq8_OwwUSR_1HM9hsqJ_ouLqqacmMyv_s4-sS5BZehG6iT5B34KRt95ZK5v1LFdB_qRXGwexLXeZGntlQMrfui0L16wKJsyPN8Z4iGVtPjYpv4D7XCSrADaeVcRR02J5SFNnzUtcCzAnhR1ttmqSRcGCR7f1VkCdrtyMmngvexxb5rSRQ8YY6NWlcF78BP0WrFK2OPMPCAijthZTqj7_evBWziV2mwlkn5nmg3cIJiSaRuRVDA8-UzQTW5ePdN6uq8z__n_2Hz6eLqZo6X-fhy_GaK1cboMp9cTMeXN9c2GVP0cC3T2NykAOSonN9ejlM3E5d7UOU5vxXlLRe1XIoKwMzBbOwMXxIOjBik1dbjuapFGx3V4NrY7QTnI0q28OkXLqPhTM6cAocas7dEAx5mJoSP9ggmLbBQx3UBpDkId74TvpSU8ZZLGw50uSRA3TpEktoVzdHVYTj1rHCXkEFdN8SA-iwNhKdAkw_o-rh76ur2mxeBvZkAowhGwmxYB3kzESDcYXrHS-obRkZHA6tWgHebR1HY5-YSZCsIsWaxqNF8xyn-x7_8Nzadz6_m7C2c2wspjM-nk6t3mOU9m07gi_PFzLI-TKnGyAIucD-Ps0YJVFTyalFrhBcP-DIP60KG50t_iTNC32xhBeIrZtBgA7WTKKK-RP1yR5hbrV-8ypGsZD3PgcOvNRcwbAkwbJr4W2c4v-qZcGQn7Nnb8cWZd3U9vXzWAs2FwhJrksLUKInUyIzYdwdo6aD9hVEmsi0WXuLUL_a0OYfJxdVievps2Dfwlc6rRvGMhKCd6KlJiwC-vtjR4mLET8CTpHvgV7dqEMKLMgbZ6zylLQYycO6Xqlhm1chHEpt-DqKtOySNU752j_w6sZBGbTO40oMI95V6dkSw7HBnWmGh8ty4UlY0xcozrFSTVpptq4AtAo3dZ655Ao3ElBadRicSEIYKs6AjYaJZKFt4CloOHndHPOLQ9Yh1N-LKKZLVOKedjIB2jSPU84ipJemI3VXYPOJAOZhq49WItnYUOCSBGJ4iXdf3EQygR4PEL_RckO6gcZp9GjwmILhv5KdBre3sxFXGXR6Iti0ydnMH5GM3sk8QqLjDUT-kB08Ew5H8J2n6HZVzyVK3eTNpEe5Uuqk-wm0eA0bJg2i4geNRs5ZoZbS6u8NKzT76XlNh7QzalfRYBhMrtJT3WniEqD-tInSqSX__b-RTR4u3N8vTq_eXbSoSf1aJLdZViUGWtvoxI34EF4C_3SZCozaxVzV1pEzotN04nEO-rnbYTBiqG_ezfFQI9AZoVK9nQ3EzPzVQMikGQmBUUsOpYwDzivy4ubzulcvm46ip6ElkHCEeDQs9qQDKSm917XJbodue2oUug76xalwdjEKv2nabEcO4EKe68Z6HfSvSNnCNRT9kMdjfHS5m8JiYZ2vWLmuZr2Xmt7mSxBQg1mO_F6Hn21rsextH0EysNgUvqAkQRlqm1GR6yP7iQZGS_IVpyHlXlBc14NrjQ8oUK7ar26xa16w63Qc8hqBucHAa94U_mrdwmRhr9gsO2MOYb3XcsG_3NulTYEOxy0zPu24x_wv8DUx22NgNPz0SjNYam2ZM5ddXURyKuzhsC3a4QYfXQscWSCCKDqIYA1ewqRyMg8lCuu1WZnr4K8nuunAwngj7hPXejK1v0pP57BgxC-xcvDjWABRIi6fEPLl07ehqE4agbH4YzH6VqUMYN2jnhMHDDddi3WuYQBrHs35Dz2ZNw4d1D_Jb_MPemywLVw_Rv2ALvuo8dlrt1sh6h6T1-v73uvLdZDx5O2Wvp2_H786v5vD57OpysZw3XJbc1vk6Wvv3MbDmU5hY05YYwqcpPm1YSpfdSjRwCjH01gT3ucYhiKmHBweHrRfCC6P8T0ugIqXMS5kX_xm5Mii1QccZ5XDo9Nrr-tZbuBV7NazXuGkplPJhr9EbasfU43B_yH750mYJrRVAES16Qrd3K6JQADfOCrAqY6s19k8FXLdhtdkWA9lgyD0l8GfhSixXTRGVJF6NNuFLagt0A0d5FKX4UMrAMY9mZRavWcWlddWNsiweaAkYKdxSgMOuseLFtzQUvTXRYZYm2LJMPFmPZbeyfxbnhXCGPe8sRdK3DEn_EiTtpYLc5Tm4vd1F5p2FMNTjYdlInvMhG41G7P3b6XxqfAoH9l-adL_fZsSQFhYU3KJwR40Jp-7ZaR6pVTgQocCtNQ6Mlo-qbtDjVTKXOtyjrIBRNIBrfDVG1V4zQFjaOJXpKEBXdM8TUvQxQZ1YMtkiuso01fLzX3Xm55sV862JsgE9JyeF3_1e9oxFteIvjfNTWahj6azHZLiw8OkrGaYSNlZhv8eJDmtXwNPOdkvBpF2rRF09MS_EDhi2L15YP30nt7iyKJCTe_OMKB_3tei8E_jbRNxpIoOXGmOJUAXhlCfPnTgJpG7hs3u2nL3TTWWJNyqwpya9dYphf_kzG5--46-OjBc_Xk66ChCjDDaWDwPwUFn55oMjTFYKZB_2xiLnVFZtLlCLHlcrYDpDNr7gyWDGgwKGm7LxYl_for12qt0vquAqrxum66j-4ZldOvrZkD3TqDz745e_NovYzpkp_tCWJozdHY7Yucy5NdZHrRulAB-N2Juo5GdIvcQqQ3UJ4sWIySV3rypBfT1CBsP1VJhIEpPXTejOBPFyROkA-CokbRG3A4Pyho--cpBXiIzI_K0F6vTOwmsKJTI_5-i4TmZXedioOWmb3WR0oIzLVounQ7PVS6vRRrwOW--uEZ-tDVzoRMe6-oMkShs99ar70DM2u70DjLpWK3LoOHruOGKsCh6nnjTyDVR8nJzEYYt93ggybrQRYe-W0XlAq9HWmyGoiVOpe1MsrNU1VBdFNPIdNYWEWEGroGsQGU39ziU0LKRGG3f2m4jl1vtz9Ovtz9Gvuj9H_7z7c_T0_Tl6wv4c9d2fy19vfy5_1f25_Ofdn8un78_lE_bnsmN_ehbysfMQPJGH4LXlIbTmI9RWL3JyYdVKxUdT2QPd2BUj3e4zaV4n7mb1PAf7-YuOdXeGXcgC6uJ6bUvVcVZsEQEz3IGkSxY8ybc5biQrelpQ6c6YsL2bRis7ZFgbxrtjLf-BnoGdT8eLq0sZpXU6nZwv8N2i2fh7_IzbKa1IrVOZdU2tVfn3RuGZWl64rqIjXxNhhYj2Ek8LyqcIbv2AnkXYc5Wql4WrSD-F7QjKegZko6CQqHGvqpGr9ygc8tn7NT7jqWF_x-w3j2iR0oycIsZjyB2j9a19jhOh0jj0cDQKprtz-e06O_REFO7DBh2_ssAGr7TD3cXH7MesYijT-oobsYI_WSfycUajETW4WUxh636hvftCn-DDjAv4iNwvYVQEecxjWvnX1_Ppu_OrmwUvp6DeEJChj6KPt-eX3x8rCR0OACvistK1VjDxGv8l6DFq55N1lhUclMq17yzDvqtQvNKdeLmJ_VZ4gChiSkhqf66g1iS695OKgvhFdQpXklVL9SCjfn-w9uN0sLMAvqA1fC1aRhP8spNKxBYsYT0x6IHFBeiUlOZMNfHl_p_T09dYH4TfKTqCQVbPl5BLuTOykD77lFVJCJ9BawlEW3n-LElw0Kbhn3aVAIkV6K2V9pI4PLG9E85KDsYbFZ65KoO2dwpEmSTg7FkFi6SqnOyCr9X3796q7gpqcmjxdHB3DX_dPCUjfQtwAkCYBS3NMla2-NrRTJYYlL33fABgLOICCv0UQFfNQdoXVZieDD96p1trz2FT9ZK1iAF2uQLcDboH2FHuruM5BvHih8rnwL_bn2JwbaXxHMOgC1zE60la8_i94idtrewnKJw72Tl186kGa3q_8psN9CwDcXNdviV4COD69Qtd2sVRwMU9RKNUpib6tjjkDvpRjNwskNlFcKIErHX1tVwxDTLbWVJzCPig7B1EVEUoSKrCXUS5s9DmOC0-oQGSgjl31adURTk9k_GfMPQo7mgLdx_azXqU6Xxyxc1W9IwXJ1sbY_1GDA2LePlNowMP77ydtTub5T7bS4r2DOTUjJRqjYit6qJPcRHtXuMmS1SRh1L8TB1X3u6L68OesH7S40-EtgoJ3-tCRIrOTmrqo0PRQ53-Z_M9onlETpyWa1vnVvo5FmXrFgzqxYRxKEMDyMVQFEJvVJXFyhWNievyUvWhRf7KZxD10xrJC1-YjM826tK0zE94GruJYddu_iDeTxIloZhVB6V9Xiom0078qKUGtJVTsJ_j9FxFX7sLAPwdm1zNQPd4O71cnL-bisyj5Xw8-b6eedRMN3ornjQGnRZUj9PYvGnafOp833BVRCCvfBfZEXdpPxJiae-w7mXT4eyITIQ29ntBeMtgsEP_3KQF-S7cpj6zvr_Oe4KjeRc5cqDcYiDvH5gXT-HpCBInagNxghakM-2gmbW0OwfEfQOavn4q8dDIDlEZ1PrN-ZZohnqkK0-qdxltum9k_uBGBL-uQcGFzWyxPaoRKbpF5te6A3idDRXTN3Be07iD_ce-MsPxfeiZI9NABZ-JDmxUdrxrIimAI_zQmWLTjX5LEo6ZG0TqSDcB2D1dz2TEmyjibBjh2rJS-2GqU4B00VBxHXQ_cGJM0QpTedyTH_YUozR0T01WNdvxeIlVgkE-lMCbuoKV25F5Qu5SS_5TQeYHb-PHKRx8eSAe3Oyuy4Fg4NHztZHujTdyrfTOn_F0mt47H_hbtPt5xU9JewbMX7nnZztenHBtukwLetSm63AbXUtHzmtgze6xNGQnggJivQnokSlrjfZKt5RZBPxSh2mmoqbx_lMJSEU2csQelfLWwFMjhLYkzH_rd0Ls_LnIzoajZ2J692BEWzkeB4nCXR11Pt60eymts66Ok5DmWqoO9xp656tirshE3IVty1sRvS7-VXSLXjfe8eCf7A2mjkdhblLBaxlI35eZ872xoo8sIWRCAzWZX99rXsIEl2JxG91FfAvrWxTxk-anjS0ih39X-q5D0BFzsOJqd9wWariu5Fxr_saZqsmxEX-Mrc9l6kwq5Z10P9jj5MVGH49lxoqZSiYTb7Z-0MmMW0_mX5WLq5QgVYuhQw-S9KuKQvTUhaQjmhI3HqcG2fUl0F_ziPxye0Apf-1I0bbvD64bdt8gXW-ftVjwO64KtbgC3Z2j2tqpDL3fKWG22DXtxe6ZO19b6F41Kox1XvHXvXpc1voAfS4jKtzbXZ6k71rvOL9GcQst_cRJ0vnY16OP4I7T168gS59jxFu2V2rpOA-KTTx1pQUP2bnBj-M1HQTNS5fEqZ7_dZTH2a5p64TQHGO7yF7-NCKTsl8e_anbfuCgMutROUvvR_t_n-ot_QsXmC9GqmT_mgPJAEWjyB2wm59laQAs5d4Cq30XVLGgedkZsGbRT8kIVZWBFqvjYrmjmoL0abWFcvP0JTUNDIF3lUjZI2e_pcAd08sTyLWQh2GtB4Ix5omd8XCAJMs-VlvJHnln-mBwMVIVaaD6q9GWB22Luo7EG428BZJKu4pd1r23HHWqf64Nwg6Hn-2TFwuTdXjkjcrwiizwwVsR5eQ0NoosYmPpZTJxZ665g1T61sHQTWWh9mYljDZy211SiC_pXQUaDOZBdIXRKflPGmBKNMWRFB8ntSzARvUOeiokvo2BQkz0sas46lvIg3BdR8kW5_HoOh6nP16OZ-cTNrm6PDt_c8MTf4XbYnzxfHk-m7Lx6fh62cwIztLb-K4SYWbzCEs2UmCUqPkxDv1t2VLWMydopFfdRXOlzt6IQpAhYJ7eDxD3PA6juskPACV4Hok3ywA-zrOUzCT3fh6jkmevqG6kdFWhj2mcMJmzam3HS00LalERV-2TOnujY0rNQYI1Pj9WeCK8NY66IzexWsqD5-yi3SejXkK3t21C7UJHbhbJ-8J0ZnJQ99zqpgMJ2KgA3FqVA45iWPlJgmdom8Cx7PIzGPrflgdIqAH7uW8xdrTvSlDYmlgJjOpqD-kxg_3CP4EqS7GUyNlwJSjRLPsYpV3tSABv0KGrgSyBQHFmj5w-z2rsuwATPX-RoheV6HVvz4aWs-fp0FV3-U-8YXi_5MN2wVr1Qx49W7Pcet85Y7i5mDPPEWqbs_GMA5-0qPZDdIlzczaggHSqJPxgxKO751-v0GNaxjmKPcO-VcmOnbyj9tKRKkbdxbBljLjZteLzra-PAL57w70NCJV-HO4d71EA6Ye9ch1tog97JL7xcsGYjfgFQP2qzBawDHvHlE64l2P44t4xZbUP9_h6nMY-sJONBNn66d9mmfnn3vEve5_3jp8ffnMwevnim2-PXh188-Lw5dHhcO9h7_jlq69HB9--OPj25dHBb3_7zctvvwz3fqYODo9GB_Tf0eHR0atvv_nmpRxyGqLjTuGB045yHAalUJjXXVyUgD1fHvy8yhP4eF2W2-L4q6_w69EdrE61GsFqfVXE4drPy_X9t6--enX06rf-0Yvo1Tcv_JcvXoTB6vDb394efX14G35zcHjk730B_CIafwbyIPSKFRRgsLscl1QsEyx0lE_QkgHT-PrbL_8fQ8c-qQ" target="_blank" class="developer-link">
            <strong>Find the Sequence Diagram of the Architecture</strong><br>
        </a></p>
        
        <div class="query-response-section">
            <div class="query-box">
                <h3>📝 Original Query:</h3>
                <p><strong>{graph_data['queryResponse']['query']}</strong></p>
            </div>
            <div class="response-box">
                <h3>✅ Final Response:</h3>
                <p><strong>{graph_data['queryResponse']['response']}</strong></p>
            </div>
        </div>
        
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
    
    <div class="pseudocode-section">
    <h3>🧠 Core Agentic AI Algorithm Pseudocode</h3>
    <button class="toggle-pseudocode" onclick="togglePseudocode()">📖 Show/Hide Algorithm Details</button>
    
    <div id="pseudocode-content" style="display: none;">
        <div class="pseudocode-highlight">
            <strong>Algorithm Overview:</strong> This pseudocode represents the core autonomous AI agent that processes queries through intelligent reasoning, tool selection, execution, and response synthesis.
        </div>
        
        <div class="pseudocode-container">
            <span class="comment">// ===== INITIALIZATION PHASE =====</span>
            <span class="keyword">ALGORITHM</span> <span class="function-name">AgenticAI_QueryProcessor</span>
            
            <span class="keyword">INITIALIZE</span> <span class="function-name">AgentSystem</span>:
                <span class="keyword">SET</span> <span class="variable">available_tools</span> = [get_order_items, get_item_return_days, database_query, cache_manager, ...]
                <span class="keyword">SET</span> <span class="variable">reasoning_engine</span> = <span class="function-name">NeuralReasoningModule</span>()
                <span class="keyword">SET</span> <span class="variable">cache_system</span> = <span class="function-name">IntelligentCache</span>()
                <span class="keyword">SET</span> <span class="variable">error_handler</span> = <span class="function-name">AdaptiveErrorRecovery</span>()
                <span class="keyword">SET</span> <span class="variable">execution_graph</span> = <span class="function-name">CallGraph</span>()
            
            <span class="comment">// ===== MAIN PROCESSING LOOP =====</span>
            <span class="keyword">FUNCTION</span> <span class="function-name">ProcessQuery</span>(user_query):
                
                <span class="comment">// Phase 1: Query Reception and Initial Analysis</span>
                <span class="keyword">CREATE</span> <span class="variable">node_start</span> = <span class="function-name">QueryStartNode</span>(user_query, timestamp=<span class="function-name">NOW</span>())
                <span class="keyword">ADD</span> <span class="variable">node_start</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                
                <span class="comment">// Phase 2: AI Thinking and Context Analysis</span>
                <span class="variable">thinking_node</span> = <span class="keyword">CREATE</span> <span class="function-name">ThinkingNode</span>()
                <span class="variable">reasoning_context</span> = <span class="function-name">ANALYZE_QUERY_CONTEXT</span>(user_query)
                
                <span class="keyword">WHILE</span> reasoning_incomplete:
                    <span class="variable">thought_process</span> = reasoning_engine.<span class="function-name">THINK</span>(
                        query=user_query,
                        context=reasoning_context,
                        available_tools=available_tools
                    )
                    
                    <span class="keyword">ADD</span> <span class="function-name">ThinkingNode</span>(thought_process) <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                    
                    <span class="keyword">IF</span> thought_process.confidence > <span class="variable">CONFIDENCE_THRESHOLD</span>:
                        <span class="keyword">BREAK</span> reasoning_loop
                
                <span class="comment">// Phase 3: Tool Selection and Action Planning</span>
                <span class="variable">selected_actions</span> = <span class="function-name">INTELLIGENT_TOOL_SELECTION</span>(
                    query=user_query,
                    context=reasoning_context,
                    tools=available_tools
                )
                
                <span class="comment">// Phase 4: Sequential Action Execution with Dynamic Routing</span>
                <span class="variable">execution_results</span> = []
                
                <span class="keyword">FOR EACH</span> action <span class="keyword">IN</span> selected_actions:
                    
                    <span class="comment">// Check cache first (optimization)</span>
                    <span class="variable">cache_key</span> = <span class="function-name">GENERATE_CACHE_KEY</span>(action.name, action.parameters)
                    
                    <span class="keyword">IF</span> cache_system.<span class="function-name">HAS_CACHED_RESULT</span>(cache_key):
                        <span class="variable">cache_node</span> = <span class="keyword">CREATE</span> <span class="function-name">CacheHitNode</span>(action.name, cache_key)
                        <span class="variable">result</span> = cache_system.<span class="function-name">GET_CACHED_RESULT</span>(cache_key)
                        <span class="keyword">ADD</span> <span class="variable">cache_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                        
                        <span class="comment">// Logical inference on cached result</span>
                        <span class="variable">inference_node</span> = <span class="keyword">CREATE</span> <span class="function-name">LogicalInferenceNode</span>(
                            reasoning=<span class="string">"Cache hit - data retrieved instantly"</span>,
                            evidence=result,
                            conclusion=<span class="string">"Saved execution time"</span>
                        )
                        <span class="keyword">ADD</span> <span class="variable">inference_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                        <span class="keyword">CONNECT</span> <span class="variable">cache_node</span> <span class="keyword">TO</span> <span class="variable">inference_node</span>
                        
                    <span class="keyword">ELSE</span>:
                        <span class="comment">// Execute action</span>
                        <span class="variable">execution_node</span> = <span class="keyword">CREATE</span> <span class="function-name">ActionExecutionNode</span>(action.name)
                        <span class="function-name">START_TIMER</span>()
                        
                        <span class="keyword">TRY</span>:
                            <span class="variable">result</span> = <span class="function-name">EXECUTE_ACTION</span>(action.name, action.parameters)
                            <span class="variable">execution_time</span> = <span class="function-name">STOP_TIMER</span>()
                            
                            execution_node.<span class="function-name">SET_RESULT</span>(result, execution_time)
                            execution_node.<span class="function-name">SET_STATUS</span>(<span class="string">"completed"</span>)
                            
                            <span class="comment">// Cache successful results</span>
                            cache_system.<span class="function-name">STORE_RESULT</span>(cache_key, result)
                            
                        <span class="keyword">CATCH</span> error:
                            <span class="comment">// Error handling and recovery</span>
                            <span class="variable">error_node</span> = <span class="keyword">CREATE</span> <span class="function-name">ErrorNode</span>(error.type, error.message)
                            <span class="keyword">ADD</span> <span class="variable">error_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                            <span class="keyword">CONNECT</span> <span class="variable">execution_node</span> <span class="keyword">TO</span> <span class="variable">error_node</span>
                            
                            <span class="comment">// Attempt recovery or alternative routing</span>
                            <span class="variable">recovery_result</span> = error_handler.<span class="function-name">ATTEMPT_RECOVERY</span>(action, error)
                            <span class="keyword">IF</span> recovery_result.success:
                                <span class="variable">result</span> = recovery_result.data
                            <span class="keyword">ELSE</span>:
                                <span class="comment">// Continue with partial results or fallback</span>
                                <span class="variable">result</span> = <span class="function-name">FALLBACK_STRATEGY</span>(action, error)
                        
                        <span class="keyword">ADD</span> <span class="variable">execution_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                        
                        <span class="comment">// Logical inference on execution result</span>
                        <span class="keyword">IF</span> result.success:
                            <span class="variable">inference_node</span> = <span class="keyword">CREATE</span> <span class="function-name">LogicalInferenceNode</span>(
                                reasoning=f<span class="string">"Tool {{action.name}} returned {{result.record_count}} items"</span>,
                                evidence=result.data,
                                conclusion=<span class="string">"Successfully retrieved records"</span>
                            )
                            <span class="keyword">ADD</span> <span class="variable">inference_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                            <span class="keyword">CONNECT</span> <span class="variable">execution_node</span> <span class="keyword">TO</span> <span class="variable">inference_node</span>
                    
                    execution_results.<span class="function-name">APPEND</span>(result)
                    
                    <span class="comment">// Dynamic routing: decide next action based on current results</span>
                    <span class="variable">next_actions</span> = <span class="function-name">ADAPTIVE_ROUTING</span>(
                        current_results=execution_results,
                        remaining_actions=selected_actions,
                        query_context=reasoning_context
                    )
                    
                    <span class="keyword">IF</span> next_actions.requires_modification:
                        <span class="variable">selected_actions</span> = <span class="function-name">UPDATE_ACTION_PLAN</span>(selected_actions, next_actions)
                
                <span class="comment">// Phase 5: Result Synthesis and Response Generation</span>
                <span class="variable">synthesis_thinking</span> = <span class="keyword">CREATE</span> <span class="function-name">ThinkingNode</span>(
                    thought=<span class="string">"Query processing completed successfully"</span>,
                    context=f<span class="string">"Generated response based on {{LEN(execution_results)}} results"</span>
                )
                <span class="keyword">ADD</span> <span class="variable">synthesis_thinking</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                
                <span class="variable">final_response</span> = <span class="function-name">SYNTHESIZE_RESPONSE</span>(
                    query=user_query,
                    execution_results=execution_results,
                    reasoning_context=reasoning_context
                )
                
                <span class="comment">// Phase 6: Final Response Delivery</span>
                <span class="variable">response_node</span> = <span class="keyword">CREATE</span> <span class="function-name">FinalResponseNode</span>(final_response)
                <span class="keyword">ADD</span> <span class="variable">response_node</span> <span class="keyword">TO</span> <span class="variable">execution_graph</span>
                <span class="keyword">CONNECT</span> <span class="variable">synthesis_thinking</span> <span class="keyword">TO</span> <span class="variable">response_node</span>
                
                <span class="keyword">RETURN</span> final_response, execution_graph
            
            <span class="comment">// ===== SUPPORTING ALGORITHMS =====</span>
            
            <span class="keyword">FUNCTION</span> <span class="function-name">INTELLIGENT_TOOL_SELECTION</span>(query, context, tools):
                <span class="comment">// Vector similarity matching + semantic analysis</span>
                <span class="variable">query_embedding</span> = <span class="function-name">EMBED_QUERY</span>(query)
                <span class="variable">tool_scores</span> = []
                
                <span class="keyword">FOR EACH</span> tool <span class="keyword">IN</span> tools:
                    <span class="variable">tool_embedding</span> = <span class="function-name">EMBED_TOOL_DESCRIPTION</span>(tool)
                    <span class="variable">similarity_score</span> = <span class="function-name">COSINE_SIMILARITY</span>(query_embedding, tool_embedding)
                    <span class="variable">context_relevance</span> = <span class="function-name">CALCULATE_CONTEXT_RELEVANCE</span>(tool, context)
                    
                    <span class="variable">final_score</span> = similarity_score * <span class="number">0.7</span> + context_relevance * <span class="number">0.3</span>
                    tool_scores.<span class="function-name">APPEND</span>((tool, final_score))
                
                <span class="comment">// Select top-k tools based on confidence threshold</span>
                <span class="variable">selected_tools</span> = <span class="function-name">FILTER</span>(tool_scores, score > <span class="variable">SELECTION_THRESHOLD</span>)
                <span class="keyword">RETURN</span> <span class="function-name">SORT</span>(selected_tools, BY=score, DESCENDING=True)
            
            <span class="keyword">FUNCTION</span> <span class="function-name">ADAPTIVE_ROUTING</span>(current_results, remaining_actions, query_context):
                <span class="comment">// Dynamic decision making based on intermediate results</span>
                <span class="variable">confidence_level</span> = <span class="function-name">CALCULATE_CONFIDENCE</span>(current_results)
                <span class="variable">completeness_score</span> = <span class="function-name">ASSESS_COMPLETENESS</span>(current_results, query_context)
                
                <span class="keyword">IF</span> confidence_level > <span class="number">0.9</span> <span class="keyword">AND</span> completeness_score > <span class="number">0.8</span>:
                    <span class="keyword">RETURN</span> <span class="function-name">EarlyTermination</span>(reason=<span class="string">"Sufficient information gathered"</span>)
                
                <span class="keyword">IF</span> confidence_level < <span class="number">0.3</span>:
                    <span class="variable">additional_tools</span> = <span class="function-name">SUGGEST_ALTERNATIVE_TOOLS</span>(query_context, current_results)
                    <span class="keyword">RETURN</span> <span class="function-name">ModifiedPlan</span>(additional_tools)
                
                <span class="keyword">RETURN</span> <span class="function-name">ContinueExecution</span>()
            
            <span class="keyword">FUNCTION</span> <span class="function-name">SYNTHESIZE_RESPONSE</span>(query, execution_results, reasoning_context):
                <span class="comment">// Advanced response generation using LLM capabilities</span>
                <span class="variable">successful_results</span> = <span class="function-name">FILTER</span>(execution_results, result.success == True)
                
                <span class="keyword">IF</span> <span class="function-name">EMPTY</span>(successful_results):
                    <span class="keyword">RETURN</span> <span class="function-name">GENERATE_FALLBACK_RESPONSE</span>(query, reasoning_context)
                
                <span class="comment">// Combine multiple data sources intelligently</span>
                <span class="variable">combined_data</span> = <span class="function-name">MERGE_RESULTS</span>(successful_results)
                
                <span class="comment">// Generate natural language response</span>
                <span class="variable">response</span> = <span class="function-name">LLM_GENERATE_RESPONSE</span>(
                    template=<span class="string">"Based on the query '{{query}}', here's what I found: {{data}}"</span>,
                    query=query,
                    data=combined_data,
                    tone=<span class="string">"professional"</span>,
                    format=<span class="string">"customer_service"</span>
                )
                
                <span class="keyword">RETURN</span> response
            
            <span class="comment">// ===== MAIN EXECUTION ENTRY POINT =====</span>
            <span class="keyword">MAIN</span>:
                <span class="variable">user_query</span> = <span class="string">"What is the return policy for order number 1001?"</span>
                
                <span class="variable">agent_system</span> = <span class="keyword">INITIALIZE</span> <span class="function-name">AgentSystem</span>()
                <span class="variable">response</span>, <span class="variable">call_graph</span> = agent_system.<span class="function-name">ProcessQuery</span>(user_query)
                
                <span class="function-name">VISUALIZE_EXECUTION_GRAPH</span>(call_graph)
                <span class="function-name">DISPLAY_RESPONSE</span>(response)
                
                <span class="variable">performance_metrics</span> = <span class="function-name">CALCULATE_PERFORMANCE_METRICS</span>(call_graph)
                <span class="function-name">LOG_METRICS</span>(performance_metrics)
            
            <span class="keyword">END ALGORITHM</span>
                    </div>
                    
                    <div class="pseudocode-highlight">
                        <strong>Key Features:</strong>
                        <ul>
                            <li><strong>Autonomous Reasoning:</strong> Multi-phase thinking and context analysis</li>
                            <li><strong>Intelligent Tool Selection:</strong> Vector similarity + semantic analysis</li>
                            <li><strong>Dynamic Routing:</strong> Adaptive decision making based on intermediate results</li>
                            <li><strong>Cache Optimization:</strong> Intelligent caching with performance tracking</li>
                            <li><strong>Error Recovery:</strong> Multiple recovery strategies for fault tolerance</li>
                            <li><strong>Execution Transparency:</strong> Complete call graph generation for analysis</li>
                        </ul>
                    </div>
                </div>
            </div>
    
    <div class="callgraph-container">
        <div class="network-column">
            <div class="column-header">🎯 Interactive Call Graph Visualization</div>
            <div id="network"></div>
        </div>
        <div class="execution-image-column">
            <div class="column-header">📊 Detailed Execution Flow</div>
            <img src="image1.png" alt="Detailed Execution Flow" class="execution-image">
        </div>
    </div>
    
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
        
        // Shadow node tracking variables
        let activeShadowNodes = [];
        let activeShadowEdges = [];
        
        // Event listeners
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                
                // Check if it's a shadow node (ignore clicks on shadow nodes)
                if (nodeId.startsWith('shadow_')) {{
                    return;
                }}
                
                const node = graphData.nodes.find(n => n.id === nodeId);
                showNodeDetails(node);
            }} else {{
                // Clicked on empty space, clear shadow nodes
                clearShadowNodes();
                document.getElementById('node-info').innerHTML = 'Click on a node to see comprehensive detailed information';
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
        
        function togglePseudocode() {{
            const content = document.getElementById('pseudocode-content');
            const button = document.querySelector('.toggle-pseudocode');
    
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                button.textContent = '📖 Hide Algorithm Details';
            }} else {{
                content.style.display = 'none';
                button.textContent = '📖 Show Algorithm Details';
            }}
        }}
        
        function highlightActions() {{
            // Clear shadow nodes first to avoid conflicts
            clearShadowNodes();
            
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
            
            // Clear shadow nodes first
            clearShadowNodes();
            
            // Reset to original data
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
                clearShadowNodes();
                return;
            }}
            
            // Show shadow nodes for this node
            showShadowNodes(node);
            
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
                    
                    <div class="detail-section">
                        <h5>👻 Shadow Details</h5>
                        <p><em>Shadow nodes now visible below showing detailed internal processes.</em></p>
                    </div>
                </div>
            `;
            
            infoDiv.innerHTML = details;
        }}
        
        function showShadowNodes(node) {{
            // Clear existing shadow nodes
            clearShadowNodes();
            
            // Get shadow data for this node
            const shadowData = graphData.shadowNodes[node.id];
            if (!shadowData || shadowData.length === 0) {{
                return;
            }}
            
            // Get the position of the main node
            const nodePosition = network.getPositions([node.id]);
            const mainNodePos = nodePosition[node.id];
            
            // Create shadow nodes
            shadowData.forEach((shadowInfo, index) => {{
                const shadowNode = {{
                    id: shadowInfo.id,
                    label: shadowInfo.label,
                    title: shadowInfo.details,
                    x: mainNodePos.x + (index - shadowData.length/2) * 80,
                    y: mainNodePos.y + 120,
                    color: {{
                        background: 'rgba(255,255,255,0.9)',
                        border: '#666',
                        highlight: {{
                            background: 'rgba(255,255,255,1)',
                            border: '#333'
                        }}
                    }},
                    shape: 'box',
                    font: {{ size: 9, color: '#333', italic: true }},
                    size: 15,
                    borderWidth: 2,
                    borderWidthSelected: 3,
                    chosen: false,
                    physics: false,
                    opacity: 0.7
                }};
                
                activeShadowNodes.push(shadowNode);
                
                // Create edge from main node to shadow node
                const shadowEdge = {{
                    id: `edge_${{node.id}}_to_${{shadowInfo.id}}`,
                    from: node.id,
                    to: shadowInfo.id,
                    color: {{ color: '#999', opacity: 0.5 }},
                    width: 1,
                    dashes: [5, 5],
                    arrows: 'to',
                    physics: false
                }};
                
                activeShadowEdges.push(shadowEdge);
            }});
            
            // CORRECTED: Use proper vis.js API - add nodes and edges to existing dataset
            const currentNodes = graphData.nodes.slice(); // Create a copy
            const currentEdges = graphData.edges.slice(); // Create a copy
            
            // Add shadow nodes and edges
            const newNodes = currentNodes.concat(activeShadowNodes);
            const newEdges = currentEdges.concat(activeShadowEdges);
            
            // Update the network with new data
            network.setData({{ nodes: newNodes, edges: newEdges }});
        }}
        
        function clearShadowNodes() {{
            if (activeShadowNodes.length === 0 && activeShadowEdges.length === 0) {{
                return;
            }}
            
            // CORRECTED: Reset to original data instead of using getData()
            network.setData({{ 
                nodes: graphData.nodes, 
                edges: graphData.edges 
            }});
            
            // Clear the arrays
            activeShadowNodes = [];
            activeShadowEdges = [];
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
            "total_execution_time": (
                    datetime.now() - self.query_start_time).total_seconds() if self.query_start_time else 0,
            "node_counts": {},
            "tool_calls": len([n for n in self.nodes.values() if
                               n.node_type in [NodeType.TOOL_EXECUTION, NodeType.ACTION_EXECUTION]]),
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
                    self.execution_tracker.add_error(f"Cached {method_name} failed: {str(e)}",
                                                     f"cache_error_{method_name}")
                    return {"error": str(e)}
            else:
                # Track cache miss and tool execution
                node_id = self.execution_tracker.add_action_execution(
                    method_name,
                    {"args": str(args[1:])[:100], "kwargs": str(kwargs)[:100]},
                    # Convert to string to avoid dict issues
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
