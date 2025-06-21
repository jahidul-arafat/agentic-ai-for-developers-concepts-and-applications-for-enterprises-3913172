#!/usr/bin/env python3
"""
Live Call Graph Tracker for Customer Service AI Agent
Generates real-time visualization of agent's thought process and tool execution
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import threading
from collections import defaultdict

@dataclass
class CallGraphNode:
    """Represents a single node in the call graph"""
    id: str
    timestamp: datetime
    node_type: str  # 'thought', 'action', 'observation', 'subgoal', 'decision'
    content: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True

@dataclass
class CallGraphSession:
    """Represents a complete query session with call graph"""
    session_id: str
    query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes: Dict[str, CallGraphNode] = field(default_factory=dict)
    execution_path: List[str] = field(default_factory=list)
    is_decomposed: bool = False
    subgoals: List[str] = field(default_factory=list)

class LiveCallGraphTracker:
    """
    Tracks and visualizes the complete agent execution flow in real-time
    """

    def __init__(self):
        self.current_session: Optional[CallGraphSession] = None
        self.sessions: Dict[str, CallGraphSession] = {}
        self.node_counter = 0
        self.lock = threading.Lock()
        self.live_display = True

    def start_session(self, query: str, is_decomposed: bool = False) -> str:
        """Start a new call graph session"""
        session_id = f"session_{int(time.time())}_{self.node_counter}"

        with self.lock:
            self.current_session = CallGraphSession(
                session_id=session_id,
                query=query,
                start_time=datetime.now(),
                is_decomposed=is_decomposed
            )
            self.sessions[session_id] = self.current_session

        if self.live_display:
            print(f"\n🎯 CALL GRAPH SESSION STARTED: {session_id}")
            print(f"📝 Query: {query}")
            print(f"🧩 Decomposed: {'Yes' if is_decomposed else 'No'}")
            print("=" * 80)

        return session_id

    def add_thought(self, content: str, parent_id: str = None, metadata: Dict = None) -> str:
        """Add a thought node to the call graph"""
        return self._add_node(
            node_type="thought",
            content=content,
            parent_id=parent_id,
            metadata=metadata or {},
            icon="🤔"
        )

    def add_action(self, tool_name: str, action_input: Any, parent_id: str = None) -> str:
        """Add an action node to the call graph"""
        return self._add_node(
            node_type="action",
            content=f"{tool_name}({action_input})",
            parent_id=parent_id,
            metadata={
                "tool_name": tool_name,
                "input": action_input
            },
            icon="⚡"
        )

    def add_observation(self, result: Any, parent_id: str = None, execution_time: float = 0.0, success: bool = True) -> str:
        """Add an observation node to the call graph"""
        result_summary = self._summarize_result(result)
        return self._add_node(
            node_type="observation",
            content=result_summary,
            parent_id=parent_id,
            metadata={
                "full_result": str(result)[:500],
                "success": success
            },
            execution_time=execution_time,
            success=success,
            icon="👁️" if success else "❌"
        )

    def add_subgoal(self, subgoal: str, priority: int = 1, parent_id: str = None) -> str:
        """Add a subgoal node to the call graph"""
        node_id = self._add_node(
            node_type="subgoal",
            content=subgoal,
            parent_id=parent_id,
            metadata={
                "priority": priority,
                "status": "pending"
            },
            icon="🎯"
        )

        if self.current_session:
            self.current_session.subgoals.append(node_id)

        return node_id

    def add_decision(self, decision: str, options: List[str] = None, chosen: str = None, parent_id: str = None) -> str:
        """Add a decision node to the call graph"""
        return self._add_node(
            node_type="decision",
            content=decision,
            parent_id=parent_id,
            metadata={
                "options": options or [],
                "chosen": chosen
            },
            icon="🔀"
        )

    def update_subgoal_status(self, subgoal_id: str, status: str, result: str = None):
        """Update the status of a subgoal"""
        with self.lock:
            if self.current_session and subgoal_id in self.current_session.nodes:
                node = self.current_session.nodes[subgoal_id]
                node.metadata["status"] = status
                if result:
                    node.metadata["result"] = result

                if self.live_display:
                    status_icon = "✅" if status == "completed" else "⚠️" if status == "failed" else "🔄"
                    print(f"    {status_icon} Subgoal {status.upper()}: {node.content[:60]}...")

    def _add_node(self, node_type: str, content: str, parent_id: str = None,
                  metadata: Dict = None, execution_time: float = 0.0,
                  success: bool = True, icon: str = "📍") -> str:
        """Internal method to add a node to the call graph"""

        with self.lock:
            if not self.current_session:
                return ""

            self.node_counter += 1
            node_id = f"node_{self.node_counter}"

            node = CallGraphNode(
                id=node_id,
                timestamp=datetime.now(),
                node_type=node_type,
                content=content,
                parent_id=parent_id,
                metadata=metadata or {},
                execution_time=execution_time,
                success=success
            )

            # Add to session
            self.current_session.nodes[node_id] = node
            self.current_session.execution_path.append(node_id)

            # Update parent-child relationships
            if parent_id and parent_id in self.current_session.nodes:
                self.current_session.nodes[parent_id].children.append(node_id)

            # Live display
            if self.live_display:
                self._display_node_live(node, icon)

            return node_id

    def _display_node_live(self, node: CallGraphNode, icon: str):
        """Display node in real-time"""
        indent = "  " * self._get_node_depth(node.id)
        timestamp = node.timestamp.strftime("%H:%M:%S.%f")[:-3]

        # Format content based on node type
        if node.node_type == "thought":
            print(f"{indent}{icon} [{timestamp}] THOUGHT: {node.content}")
        elif node.node_type == "action":
            tool_name = node.metadata.get("tool_name", "unknown")
            print(f"{indent}{icon} [{timestamp}] ACTION: {tool_name}")
            print(f"{indent}    Input: {node.metadata.get('input', 'N/A')}")
        elif node.node_type == "observation":
            status = "SUCCESS" if node.success else "FAILED"
            time_info = f" ({node.execution_time:.3f}s)" if node.execution_time > 0 else ""
            print(f"{indent}{icon} [{timestamp}] OBSERVATION [{status}]{time_info}: {node.content}")
        elif node.node_type == "subgoal":
            priority = node.metadata.get("priority", 1)
            print(f"{indent}{icon} [{timestamp}] SUBGOAL #{priority}: {node.content}")
        elif node.node_type == "decision":
            print(f"{indent}{icon} [{timestamp}] DECISION: {node.content}")
            if node.metadata.get("chosen"):
                print(f"{indent}    Chosen: {node.metadata['chosen']}")

    def _get_node_depth(self, node_id: str) -> int:
        """Calculate the depth of a node in the call graph"""
        if not self.current_session or node_id not in self.current_session.nodes:
            return 0

        node = self.current_session.nodes[node_id]
        depth = 0

        while node.parent_id:
            depth += 1
            if node.parent_id in self.current_session.nodes:
                node = self.current_session.nodes[node.parent_id]
            else:
                break

        return depth

    def _summarize_result(self, result: Any) -> str:
        """Create a summary of the result for display"""
        if isinstance(result, dict):
            if 'error' in result:
                return f"Error: {result['error'][:100]}..."
            elif 'orders_found' in result:
                return f"Found {result['orders_found']} orders"
            elif len(result) > 0:
                first_key = list(result.keys())[0]
                return f"Dict[{len(result)} keys]: {first_key}=..."
            else:
                return "Empty dict"
        elif isinstance(result, list):
            if len(result) > 0:
                return f"List[{len(result)}]: [{str(result[0])[:50]}...]"
            else:
                return "Empty list"
        elif isinstance(result, str):
            return result[:100] + ("..." if len(result) > 100 else "")
        else:
            return str(result)[:100]

    def end_session(self, final_result: str = None) -> CallGraphSession:
        """End the current session and return the call graph"""
        with self.lock:
            if not self.current_session:
                return None

            self.current_session.end_time = datetime.now()
            session = self.current_session

            if self.live_display:
                duration = (session.end_time - session.start_time).total_seconds()
                print(f"\n🏁 SESSION COMPLETED: {session.session_id}")
                print(f"⏱️  Duration: {duration:.2f}s")
                print(f"📊 Total nodes: {len(session.nodes)}")
                print(f"🎯 Subgoals: {len(session.subgoals)}")
                if final_result:
                    print(f"✅ Final result: {final_result[:100]}...")
                print("=" * 80)

            self.current_session = None
            return session

    def generate_call_graph_report(self, session_id: str = None) -> Dict[str, Any]:
        """Generate a comprehensive call graph report"""
        session = self.current_session if not session_id else self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        # Analyze the call graph
        node_types = defaultdict(int)
        successful_actions = 0
        failed_actions = 0
        total_execution_time = 0.0

        for node in session.nodes.values():
            node_types[node.node_type] += 1
            if node.node_type == "observation":
                if node.success:
                    successful_actions += 1
                else:
                    failed_actions += 1
                total_execution_time += node.execution_time

        # Build execution tree
        tree = self._build_execution_tree(session)

        # Calculate metrics
        duration = (session.end_time - session.start_time).total_seconds() if session.end_time else 0
        success_rate = (successful_actions / (successful_actions + failed_actions) * 100) if (successful_actions + failed_actions) > 0 else 0

        return {
            "session_info": {
                "session_id": session.session_id,
                "query": session.query,
                "duration": duration,
                "is_decomposed": session.is_decomposed,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None
            },
            "metrics": {
                "total_nodes": len(session.nodes),
                "node_types": dict(node_types),
                "success_rate": success_rate,
                "total_execution_time": total_execution_time,
                "avg_execution_time": total_execution_time / max(successful_actions + failed_actions, 1)
            },
            "execution_tree": tree,
            "subgoals": [
                {
                    "id": sg_id,
                    "content": session.nodes[sg_id].content,
                    "status": session.nodes[sg_id].metadata.get("status", "unknown")
                }
                for sg_id in session.subgoals if sg_id in session.nodes
            ],
            "execution_path": [
                {
                    "node_id": node_id,
                    "type": session.nodes[node_id].node_type,
                    "content": session.nodes[node_id].content[:100],
                    "timestamp": session.nodes[node_id].timestamp.isoformat()
                }
                for node_id in session.execution_path if node_id in session.nodes
            ]
        }

    def _build_execution_tree(self, session: CallGraphSession) -> Dict:
        """Build a hierarchical execution tree"""
        def build_node_tree(node_id: str) -> Dict:
            if node_id not in session.nodes:
                return {}

            node = session.nodes[node_id]
            return {
                "id": node_id,
                "type": node.node_type,
                "content": node.content,
                "timestamp": node.timestamp.isoformat(),
                "success": node.success,
                "execution_time": node.execution_time,
                "metadata": node.metadata,
                "children": [build_node_tree(child_id) for child_id in node.children]
            }

        # Find root nodes (nodes without parents)
        root_nodes = [node_id for node_id, node in session.nodes.items() if not node.parent_id]

        return {
            "roots": [build_node_tree(root_id) for root_id in root_nodes]
        }

    def export_call_graph(self, session_id: str = None, format: str = "json") -> str:
        """Export call graph to file"""
        report = self.generate_call_graph_report(session_id)
        if "error" in report:
            return report["error"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = session_id or "current"
        filename = f"call_graph_{session_name}_{timestamp}.{format}"

        try:
            if format == "json":
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            elif format == "md":
                with open(filename, 'w') as f:
                    f.write(self._generate_markdown_report(report))
            else:
                return f"Unsupported format: {format}"

            return f"Call graph exported to {filename}"
        except Exception as e:
            return f"Export failed: {e}"

    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate a markdown report of the call graph"""
        md = f"""# Call Graph Report

## Session Information
- **Session ID:** {report['session_info']['session_id']}
- **Query:** {report['session_info']['query']}
- **Duration:** {report['session_info']['duration']:.2f}s
- **Decomposed:** {report['session_info']['is_decomposed']}

## Metrics
- **Total Nodes:** {report['metrics']['total_nodes']}
- **Success Rate:** {report['metrics']['success_rate']:.1f}%
- **Total Execution Time:** {report['metrics']['total_execution_time']:.3f}s
- **Average Execution Time:** {report['metrics']['avg_execution_time']:.3f}s

## Node Types Distribution
"""
        for node_type, count in report['metrics']['node_types'].items():
            md += f"- **{node_type.title()}:** {count}\n"

        md += "\n## Execution Path\n"
        for i, step in enumerate(report['execution_path'], 1):
            md += f"{i}. **{step['type'].title()}** ({step['timestamp']}): {step['content']}\n"

        if report['subgoals']:
            md += "\n## Subgoals\n"
            for i, subgoal in enumerate(report['subgoals'], 1):
                status_icon = "✅" if subgoal['status'] == 'completed' else "❌" if subgoal['status'] == 'failed' else "⏳"
                md += f"{i}. {status_icon} {subgoal['content']}\n"

        return md

# Global instance
call_graph_tracker = LiveCallGraphTracker()

# Integration functions for the existing CustomerServiceAgent
def track_agent_execution(original_method):
    """Decorator to track agent execution with call graph"""
    def wrapper(self, query: str, *args, **kwargs):
        # Check if this is a decomposed query
        is_decomposed = hasattr(self, 'query_decomposer') and self.query_decomposer

        # Start call graph session
        session_id = call_graph_tracker.start_session(query, is_decomposed)

        try:
            # Add initial thought
            thought_id = call_graph_tracker.add_thought(f"Starting to process query: {query}")

            # Execute the original method
            result = original_method(self, query, *args, **kwargs)

            # Add final observation
            call_graph_tracker.add_observation(
                f"Query completed successfully: {str(result)[:100]}...",
                parent_id=thought_id,
                success=True
            )

            return result

        except Exception as e:
            # Add error observation
            call_graph_tracker.add_observation(
                f"Query failed: {str(e)}",
                parent_id=thought_id if 'thought_id' in locals() else None,
                success=False
            )
            raise
        finally:
            # End session
            call_graph_tracker.end_session()

    return wrapper

def track_tool_execution(original_method):
    """Decorator to track individual tool executions"""
    def wrapper(*args, **kwargs):
        # Extract tool name and input
        tool_name = original_method.__name__
        tool_input = args[1:] if len(args) > 1 else kwargs

        # Add action node
        action_id = call_graph_tracker.add_action(tool_name, tool_input)

        start_time = time.time()
        try:
            # Execute tool
            result = original_method(*args, **kwargs)
            execution_time = time.time() - start_time

            # Add successful observation
            call_graph_tracker.add_observation(
                result,
                parent_id=action_id,
                execution_time=execution_time,
                success=True
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            # Add failed observation
            call_graph_tracker.add_observation(
                f"Tool execution failed: {str(e)}",
                parent_id=action_id,
                execution_time=execution_time,
                success=False
            )

            return {"error": str(e)}

    return wrapper

def track_subgoal_execution(decomposer_method):
    """Decorator to track subgoal decomposition and execution"""
    def wrapper(self, query: str):
        # Add decision node for decomposition
        decision_id = call_graph_tracker.add_decision(
            "Analyzing query complexity for potential decomposition",
            options=["Standard execution", "Intelligent decomposition"],
            chosen="Intelligent decomposition"
        )

        # Execute decomposition
        subgoals = decomposer_method(self, query)

        # Add subgoal nodes
        subgoal_ids = []
        for i, subgoal_info in enumerate(subgoals, 1):
            subgoal_id = call_graph_tracker.add_subgoal(
                subgoal_info['subgoal'],
                priority=i,
                parent_id=decision_id
            )
            subgoal_ids.append(subgoal_id)

        return subgoals

    return wrapper


# Agent Wrapper Classes and Functions
class CallGraphAgentWrapper:
    """Wrapper to inject call graph tracking into LlamaIndex agent"""

    def __init__(self, original_agent):
        self.original_agent = original_agent
        self._patch_agent_methods()

    def _patch_agent_methods(self):
        """Patch agent methods to include call graph tracking"""

        # Store original methods
        original_query = self.original_agent.query

        def tracked_query(query_str):
            """Wrapped query method with call graph tracking"""
            session_id = call_graph_tracker.start_session(query_str)

            try:
                # Add initial thought
                thought_id = call_graph_tracker.add_thought(
                    f"Agent starting to process query: {query_str}"
                )

                # Execute original query
                result = original_query(query_str)

                # Add completion observation
                call_graph_tracker.add_observation(
                    "Agent query completed successfully",
                    parent_id=thought_id,
                    success=True
                )

                return result

            except Exception as e:
                call_graph_tracker.add_observation(
                    f"Agent query failed: {str(e)}",
                    success=False
                )
                raise
            finally:
                call_graph_tracker.end_session()

        # Replace the original method
        self.original_agent.query = tracked_query

    def __getattr__(self, name):
        """Delegate all other attributes to the original agent"""
        return getattr(self.original_agent, name)


def run_query_with_live_visualization(agent, query: str):
    """Run query with live call graph visualization"""

    # Enable live display
    call_graph_tracker.live_display = True

    print(f"\n🎬 STARTING LIVE CALL GRAPH VISUALIZATION")
    print(f"📝 Query: {query}")
    print("🔄 Watch the agent's thought process unfold...")
    print("=" * 80)

    try:
        # Execute the query (this will automatically track the call graph)
        if hasattr(agent, 'query_decomposer') and agent.query_decomposer:
            # Check complexity first
            assessment = agent.query_decomposer.assess_query_complexity(query)
            if assessment['requires_decomposition']:
                result = agent.run_intelligent_query(query)
            else:
                result = agent.run_query(query)
        else:
            result = agent.run_query(query)

        # Show final call graph summary
        print(f"\n📊 CALL GRAPH SUMMARY")
        print("=" * 40)
        agent.show_live_call_graph()

        return result

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        return None


# Enhanced Step Tracking for ReAct agent
class StepTrackingReActAgent:
    """Enhanced ReAct agent with step-by-step call graph tracking"""

    def __init__(self, original_agent):
        self.original_agent = original_agent
        self.step_counter = 0

    def query(self, query_str):
        """Execute query with detailed step tracking"""
        session_id = call_graph_tracker.start_session(query_str)

        try:
            # Track the reasoning loop
            current_thought = None
            current_action = None

            # This would require hooking into the ReAct loop
            # For now, we'll track at the tool level
            return self._execute_with_tracking(query_str)

        finally:
            call_graph_tracker.end_session()

    def _execute_with_tracking(self, query_str):
        """Execute with step-by-step tracking"""

        # Add initial planning thought
        planning_thought = call_graph_tracker.add_thought(
            "Planning approach to answer the query"
        )

        # Execute the original query
        result = self.original_agent.query(query_str)

        # Add final synthesis
        call_graph_tracker.add_observation(
            f"Query execution completed with result",
            parent_id=planning_thought,
            success=True
        )

        return result


# Utility Functions
def demonstrate_call_graph():
    """Demonstration function showing call graph capabilities"""
    print("""
🎯 CALL GRAPH TRACKER DEMONSTRATION

The Live Call Graph Tracker provides real-time visualization of:

1. 🤔 THOUGHTS - Agent's reasoning process
2. ⚡ ACTIONS - Tool calls and function executions  
3. 👁️ OBSERVATIONS - Results and outcomes
4. 🎯 SUBGOALS - Query decomposition steps
5. 🔀 DECISIONS - Choice points and branching logic

Example output during a complex query:

🎯 CALL GRAPH SESSION STARTED: session_1703123456_1
📝 Query: Three customers (orders 1007, 1017, 1023) all received MacBook Pros...
🧩 Decomposed: Yes
================================================================================

  🤔 [12:34:56.123] THOUGHT: Starting to process query
    ⚡ [12:34:56.234] ACTION: get_multiple_orders_parallel
        Input: {'order_ids': '1007,1017,1023'}
      👁️ [12:34:56.345] OBSERVATION [SUCCESS] (0.112s): Found 3 orders
    🎯 [12:34:56.456] SUBGOAL #1: Retrieve order details for customers
      🔀 [12:34:56.567] DECISION: Query complexity assessment
          Chosen: Intelligent decomposition
        🤔 [12:34:56.678] THOUGHT: Executing subgoal 1
          ⚡ [12:34:56.789] ACTION: get_order_items
              Input: {'order_id': 1007}
            👁️ [12:34:56.890] OBSERVATION [SUCCESS] (0.045s): List[2]: ['MacBook Pro 14"', 'Bluetooth Mouse Apple']

This provides complete visibility into the agent's decision-making process!
""")


# Alternative simplified wrapper for basic tracking
class SimpleCallGraphWrapper:
    """Simplified wrapper for basic call graph functionality"""

    def __init__(self, original_agent):
        self.original_agent = original_agent

    def query(self, query_str):
        """Simple query tracking"""
        # Start session
        session_id = call_graph_tracker.start_session(query_str, is_decomposed=False)

        try:
            # Add initial thought
            thought_id = call_graph_tracker.add_thought(
                f"Processing query: {query_str}"
            )

            # Execute query
            result = self.original_agent.query(query_str)

            # Add success observation
            call_graph_tracker.add_observation(
                f"Query completed: {str(result)[:100]}...",
                parent_id=thought_id,
                success=True
            )

            return result

        except Exception as e:
            # Add failure observation
            call_graph_tracker.add_observation(
                f"Query failed: {str(e)}",
                success=False
            )
            raise
        finally:
            # End session
            call_graph_tracker.end_session()

    def __getattr__(self, name):
        """Delegate other attributes to original agent"""
        return getattr(self.original_agent, name)


if __name__ == "__main__":
    # Example of how to use the call graph tracker
    demonstrate_call_graph()