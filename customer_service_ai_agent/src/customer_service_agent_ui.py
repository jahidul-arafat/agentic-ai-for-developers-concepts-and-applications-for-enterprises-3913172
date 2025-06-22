#!/usr/bin/env python3
"""
Customer Service AI Agent Web Interface
A comprehensive web UI for monitoring, testing, and interacting with the Customer Service AI Agent
Runs on port 5010 with full visualization and control capabilities
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
import tempfile
import uuid
from dataclasses import asdict

# Flask and web dependencies
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Import all the customer service agent components
from customer_service_ai_agent.backup.customer_service_agent_mps_enhanced_v02 import (
    CustomerServiceAgent,
    config,
    tool_tracker,
    cache_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'customer-service-ui-secret-key'

# Enable CORS for all routes
CORS(app)

# Global application state
application_state = {
    'agent': None,
    'agent_initialized': False,
    'initialization_status': {},
    'database_stats': {},
    'performance_metrics': {},
    'test_sessions': [],
    'configuration': {},
    'health_status': {},
    'query_history': [],
    'active_queries': {},
    'system_logs': [],
    'monitoring_data': {
        'tool_usage': [],
        'cache_performance': [],
        'database_queries': [],
        'response_times': []
    }
}

# Thread-safe locks
import threading
state_lock = threading.Lock()

class WebUILogger:
    """Custom logger that captures logs for web display"""

    def __init__(self):
        self.logs = []
        self.max_logs = 1000

    def add_log(self, level, message, component=None):
        """Add a log entry"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'component': component or 'System'
        }

        self.logs.append(log_entry)

        # Keep only the most recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

    def get_recent_logs(self, count=50):
        """Get recent log entries"""
        return self.logs[-count:] if self.logs else []

    def clear_logs(self):
        """Clear all logs"""
        self.logs.clear()

# Global web UI logger
web_logger = WebUILogger()

class AgentManager:
    """Manages the Customer Service Agent lifecycle"""

    def __init__(self):
        self.agent = None
        self.initialization_thread = None
        self.is_initializing = False
        self.initialization_status = {
            'current_step': '',
            'progress': 0,
            'total_steps': 5,
            'completed_steps': [],
            'failed_steps': [],
            'logs': []
        }

    def initialize_agent_async(self):
        """Initialize the agent in a background thread"""
        if self.is_initializing:
            return False

        self.is_initializing = True
        self.initialization_thread = threading.Thread(target=self._initialize_agent)
        self.initialization_thread.daemon = True
        self.initialization_thread.start()
        return True

    def _initialize_agent(self):
        """Internal method to initialize the agent"""
        try:
            web_logger.add_log('INFO', 'Starting Customer Service AI Agent initialization...', 'Agent')

            # Step 1: Create agent instance
            self._update_initialization_status('Creating agent instance...', 0)
            self.agent = CustomerServiceAgent()
            self._update_initialization_status('Agent instance created', 1)

            # Step 2: Setup database
            self._update_initialization_status('Setting up database connection...', 1)
            if self.agent.setup_database():
                web_logger.add_log('SUCCESS', 'Database connected successfully', 'Database')
                self._update_initialization_status('Database connected', 2)
            else:
                raise Exception("Database setup failed")

            # Step 3: Setup LLM
            self._update_initialization_status('Configuring LLM and embeddings...', 2)
            if self.agent.setup_llm():
                web_logger.add_log('SUCCESS', 'LLM configured successfully', 'LLM')
                self._update_initialization_status('LLM configured', 3)
            else:
                raise Exception("LLM setup failed")

            # Step 4: Setup support documents
            self._update_initialization_status('Loading support documents...', 3)
            if self.agent.setup_support_documents():
                web_logger.add_log('SUCCESS', 'Support documents loaded', 'Documents')
                self._update_initialization_status('Support documents loaded', 4)
            else:
                raise Exception("Support documents setup failed")

            # Step 5: Create tools and agent
            self._update_initialization_status('Creating agent tools...', 4)
            if self.agent.create_tools() and self.agent.create_agent():
                web_logger.add_log('SUCCESS', 'Agent tools and AI agent created', 'Agent')
                self._update_initialization_status('Agent fully initialized', 5)
            else:
                raise Exception("Tools or agent creation failed")

            # Setup query decomposer
            self.agent.setup_query_decomposer()

            # Update global state
            with state_lock:
                application_state['agent'] = self.agent
                application_state['agent_initialized'] = True
                application_state['initialization_status'] = self.initialization_status.copy()

            web_logger.add_log('SUCCESS', 'Customer Service AI Agent fully initialized and ready!', 'Agent')

        except Exception as e:
            error_msg = f"Agent initialization failed: {str(e)}"
            web_logger.add_log('ERROR', error_msg, 'Agent')
            self._update_initialization_status(f"Failed: {error_msg}", -1)

            with state_lock:
                application_state['agent_initialized'] = False
                application_state['initialization_status'] = self.initialization_status.copy()

        finally:
            self.is_initializing = False

    def _update_initialization_status(self, message, progress):
        """Update initialization status"""
        self.initialization_status['current_step'] = message
        if progress >= 0:
            self.initialization_status['progress'] = progress
            if progress <= self.initialization_status['total_steps']:
                if message not in self.initialization_status['completed_steps']:
                    self.initialization_status['completed_steps'].append(message)
        else:
            self.initialization_status['failed_steps'].append(message)

        self.initialization_status['logs'].append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'progress': progress
        })

        # Keep only recent logs
        if len(self.initialization_status['logs']) > 50:
            self.initialization_status['logs'] = self.initialization_status['logs'][-50:]

# Global agent manager
agent_manager = AgentManager()

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'agent_initialized': application_state.get('agent_initialized', False),
            'initialization_in_progress': agent_manager.is_initializing,
            'components': {}
        }

        if application_state.get('agent'):
            agent = application_state['agent']

            # Database health
            try:
                health_status['components']['database'] = agent.db_manager.health_check()
            except:
                health_status['components']['database'] = {'status': 'unhealthy'}

            # LLM health
            try:
                test_response = agent.agent.query("Hello")
                health_status['components']['llm'] = {'status': 'healthy', 'test_successful': True}
            except:
                health_status['components']['llm'] = {'status': 'unhealthy', 'test_successful': False}

            # Tools health
            health_status['components']['tools'] = {
                'status': 'healthy' if agent.tools else 'unhealthy',
                'tool_count': len(agent.tools) if agent.tools else 0
            }

        return jsonify(health_status)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_agent():
    """Initialize the customer service agent"""
    try:
        if application_state.get('agent_initialized'):
            return jsonify({
                'success': False,
                'message': 'Agent already initialized',
                'status': application_state.get('initialization_status', {})
            })

        if agent_manager.is_initializing:
            return jsonify({
                'success': False,
                'message': 'Initialization already in progress',
                'status': agent_manager.initialization_status
            })

        # Start initialization
        if agent_manager.initialize_agent_async():
            return jsonify({
                'success': True,
                'message': 'Agent initialization started',
                'status': agent_manager.initialization_status
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start initialization'
            })

    except Exception as e:
        web_logger.add_log('ERROR', f"Initialize endpoint error: {str(e)}", 'API')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/initialization-status', methods=['GET'])
def get_initialization_status():
    """Get current initialization status"""
    return jsonify({
        'is_initializing': agent_manager.is_initializing,
        'initialized': application_state.get('agent_initialized', False),
        'status': agent_manager.initialization_status
    })

@app.route('/api/query', methods=['POST'])
def execute_query():
    """Execute a query using the agent"""
    try:
        if not application_state.get('agent_initialized'):
            return jsonify({
                'success': False,
                'error': 'Agent not initialized. Please initialize the agent first.'
            }), 400

        data = request.get_json()
        query = data.get('query', '').strip()
        execution_type = data.get('execution_type', 'standard')  # standard, enhanced, intelligent

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400

        agent = application_state['agent']
        query_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Add to active queries
        application_state['active_queries'][query_id] = {
            'query': query,
            'start_time': start_time,
            'execution_type': execution_type,
            'status': 'running'
        }

        web_logger.add_log('INFO', f"Executing query: {query[:100]}...", 'Query')

        try:
            # Execute based on type
            if execution_type == 'enhanced':
                response = agent.run_query_with_enhanced_tracking(query)
            elif execution_type == 'intelligent':
                response = agent.run_intelligent_query(query)
            else:
                response = agent.run_query(query)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Get tool usage stats
            tool_report = tool_tracker.get_comprehensive_report()
            cache_stats = cache_manager.get_detailed_stats()

            result = {
                'success': True,
                'query_id': query_id,
                'query': query,
                'response': str(response),
                'execution_time': execution_time,
                'execution_type': execution_type,
                'timestamp': start_time.isoformat(),
                'tool_usage': {
                    'tools_called': tool_report['session_summary']['total_tool_calls'],
                    'success_rate': tool_report['session_summary']['success_rate'],
                    'recent_calls': tool_report['recent_activity'][-5:] if tool_report['recent_activity'] else []
                },
                'cache_performance': {
                    'hit_rate': cache_stats['global']['hit_rate'],
                    'total_calls': cache_stats['global']['total_calls']
                }
            }

            # Update query history
            application_state['query_history'].append(result)
            if len(application_state['query_history']) > 100:
                application_state['query_history'] = application_state['query_history'][-100:]

            # Remove from active queries
            if query_id in application_state['active_queries']:
                del application_state['active_queries'][query_id]

            web_logger.add_log('SUCCESS', f"Query completed in {execution_time:.2f}s", 'Query')

            return jsonify(result)

        except Exception as query_error:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            error_result = {
                'success': False,
                'query_id': query_id,
                'query': query,
                'error': str(query_error),
                'execution_time': execution_time,
                'timestamp': start_time.isoformat()
            }

            # Remove from active queries
            if query_id in application_state['active_queries']:
                del application_state['active_queries'][query_id]

            web_logger.add_log('ERROR', f"Query failed: {str(query_error)}", 'Query')

            return jsonify(error_result), 500

    except Exception as e:
        web_logger.add_log('ERROR', f"Query endpoint error: {str(e)}", 'API')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/scenarios', methods=['GET'])
def get_predefined_scenarios():
    """Get list of predefined test scenarios"""
    scenarios = [
        # Basic scenarios
        {
            "id": "basic_1",
            "name": "Return Policy Query",
            "query": "What is the return policy for order number 1001?",
            "description": "Tests order lookup and return policy retrieval",
            "difficulty": "Basic",
            "category": "Return Policy"
        },
        {
            "id": "basic_2",
            "name": "Multi-part Question",
            "query": "When is the delivery date and items shipped for order 1003 and how can I contact customer support?",
            "description": "Tests multiple tool usage in single query",
            "difficulty": "Basic",
            "category": "Multi-tool"
        },
        {
            "id": "basic_3",
            "name": "Invalid Order",
            "query": "What is the return policy for order number 9999?",
            "description": "Tests handling of non-existent orders",
            "difficulty": "Basic",
            "category": "Error Handling"
        },

        # Intermediate scenarios
        {
            "id": "intermediate_1",
            "name": "Customer History Analysis",
            "query": "Show me all orders for customer john.smith@email.com and tell me which items can still be returned based on delivery dates",
            "description": "Tests customer search, multiple orders, date calculations, and policy application",
            "difficulty": "Intermediate",
            "category": "Customer Analysis"
        },
        {
            "id": "intermediate_2",
            "name": "Product Category Return Policy",
            "query": "I bought a laptop, mouse, and HDMI cable in different orders. What are the return policies for each and which one expires first?",
            "description": "Tests product search across orders, policy comparison, and time analysis",
            "difficulty": "Intermediate",
            "category": "Policy Comparison"
        },

        # Advanced scenarios
        {
            "id": "advanced_1",
            "name": "Complex Return Calculation",
            "query": "For order 1022, calculate the exact return deadline for each item considering the delivery date was June 6th, and tell me what happens if I return only the mouse but keep the laptop",
            "description": "Tests date arithmetic, partial returns, policy calculations, and business logic",
            "difficulty": "Advanced",
            "category": "Complex Calculation"
        },
        {
            "id": "advanced_2",
            "name": "Customer Lifecycle Analysis",
            "query": "Analyze the complete order history for customers in Auburn, AL. Which products are most popular, what's the average order value, and identify any delivery issues",
            "description": "Tests geographic filtering, statistical analysis, trend identification, and problem detection",
            "difficulty": "Advanced",
            "category": "Analytics"
        },

        # Expert scenarios
        {
            "id": "expert_1",
            "name": "Multi-Customer Dispute Resolution",
            "query": "Three customers (orders 1007, 1017, 1023) all received MacBook Pros but are reporting different issues: one has screen problems, one has battery issues, one has keyboard problems. Analyze their orders, determine warranty coverage, and recommend resolution strategy for each",
            "description": "Tests multi-order analysis, issue categorization, warranty determination, and personalized solutions",
            "difficulty": "Expert",
            "category": "Dispute Resolution"
        },
        {
            "id": "expert_2",
            "name": "Predictive Customer Service",
            "query": "Based on order patterns, delivery dates, and customer behavior, identify which current customers are most likely to have issues or complaints in the next week, and proactively suggest what we should do to prevent problems",
            "description": "Tests predictive analysis, risk modeling, proactive service, and prevention strategies",
            "difficulty": "Expert",
            "category": "Predictive Analysis"
        }
    ]

    return jsonify({
        'scenarios': scenarios,
        'total_count': len(scenarios),
        'difficulties': list(set(s['difficulty'] for s in scenarios)),
        'categories': list(set(s['category'] for s in scenarios))
    })

@app.route('/api/scenarios/run', methods=['POST'])
def run_scenario():
    """Run a specific predefined scenario"""
    try:
        if not application_state.get('agent_initialized'):
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 400

        data = request.get_json()
        scenario_id = data.get('scenario_id')
        execution_type = data.get('execution_type', 'standard')

        # Get scenarios
        scenarios_response = get_predefined_scenarios()
        scenarios = scenarios_response.get_json()['scenarios']

        # Find the scenario
        scenario = next((s for s in scenarios if s['id'] == scenario_id), None)
        if not scenario:
            return jsonify({
                'success': False,
                'error': 'Scenario not found'
            }), 404

        # Execute the scenario query
        query_data = {
            'query': scenario['query'],
            'execution_type': execution_type
        }

        # Call the query endpoint
        with app.test_client() as client:
            response = client.post('/api/query',
                                   json=query_data,
                                   headers={'Content-Type': 'application/json'})

            result = response.get_json()

            # Add scenario metadata
            if result.get('success'):
                result['scenario'] = scenario

            return jsonify(result), response.status_code

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-suite', methods=['POST'])
def run_test_suite():
    """Run a comprehensive test suite"""
    try:
        if not application_state.get('agent_initialized'):
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 400

        data = request.get_json()
        test_type = data.get('test_type', 'basic')  # basic, intermediate, advanced, expert, all
        execution_type = data.get('execution_type', 'standard')

        # Get scenarios
        scenarios_response = get_predefined_scenarios()
        scenarios = scenarios_response.get_json()['scenarios']

        # Filter scenarios based on test type
        if test_type == 'all':
            selected_scenarios = scenarios
        else:
            selected_scenarios = [s for s in scenarios if s['difficulty'].lower() == test_type.lower()]

        if not selected_scenarios:
            return jsonify({
                'success': False,
                'error': f'No scenarios found for test type: {test_type}'
            }), 400

        # Create test session
        session_id = str(uuid.uuid4())
        session_start = datetime.now()

        test_session = {
            'session_id': session_id,
            'test_type': test_type,
            'execution_type': execution_type,
            'start_time': session_start,
            'total_scenarios': len(selected_scenarios),
            'completed': 0,
            'results': [],
            'status': 'running'
        }

        application_state['test_sessions'].append(test_session)

        web_logger.add_log('INFO', f"Starting test suite: {test_type} ({len(selected_scenarios)} scenarios)", 'TestSuite')

        # Execute scenarios
        for i, scenario in enumerate(selected_scenarios):
            try:
                # Execute query
                query_data = {
                    'query': scenario['query'],
                    'execution_type': execution_type
                }

                with app.test_client() as client:
                    response = client.post('/api/query',
                                           json=query_data,
                                           headers={'Content-Type': 'application/json'})

                    result = response.get_json()
                    result['scenario'] = scenario
                    result['test_number'] = i + 1

                    test_session['results'].append(result)
                    test_session['completed'] = i + 1

                    web_logger.add_log('INFO', f"Completed scenario {i+1}/{len(selected_scenarios)}: {scenario['name']}", 'TestSuite')

            except Exception as scenario_error:
                error_result = {
                    'success': False,
                    'scenario': scenario,
                    'test_number': i + 1,
                    'error': str(scenario_error)
                }
                test_session['results'].append(error_result)
                test_session['completed'] = i + 1

                web_logger.add_log('ERROR', f"Scenario {i+1} failed: {str(scenario_error)}", 'TestSuite')

        # Finalize session
        test_session['end_time'] = datetime.now()
        test_session['duration'] = (test_session['end_time'] - session_start).total_seconds()
        test_session['status'] = 'completed'

        # Calculate summary statistics
        successful_tests = sum(1 for r in test_session['results'] if r.get('success'))
        test_session['summary'] = {
            'total_tests': len(test_session['results']),
            'successful_tests': successful_tests,
            'failed_tests': len(test_session['results']) - successful_tests,
            'success_rate': (successful_tests / len(test_session['results']) * 100) if test_session['results'] else 0,
            'average_execution_time': sum(r.get('execution_time', 0) for r in test_session['results'] if r.get('success')) / successful_tests if successful_tests > 0 else 0
        }

        web_logger.add_log('SUCCESS', f"Test suite completed: {successful_tests}/{len(selected_scenarios)} passed", 'TestSuite')

        return jsonify({
            'success': True,
            'test_session': test_session
        })

    except Exception as e:
        web_logger.add_log('ERROR', f"Test suite error: {str(e)}", 'TestSuite')
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-sessions', methods=['GET'])
def get_test_sessions():
    """Get all test sessions"""
    return jsonify({
        'test_sessions': application_state.get('test_sessions', []),
        'total_sessions': len(application_state.get('test_sessions', []))
    })

@app.route('/api/test-sessions/<session_id>', methods=['GET'])
def get_test_session(session_id):
    """Get a specific test session"""
    session = next((s for s in application_state.get('test_sessions', []) if s['session_id'] == session_id), None)

    if not session:
        return jsonify({
            'success': False,
            'error': 'Test session not found'
        }), 404

    return jsonify({
        'success': True,
        'test_session': session
    })

@app.route('/api/monitoring/tools', methods=['GET'])
def get_tool_usage_monitoring():
    """Get tool usage monitoring data"""
    try:
        tool_report = tool_tracker.get_comprehensive_report()

        return jsonify({
            'success': True,
            'tool_usage': tool_report,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitoring/cache', methods=['GET'])
def get_cache_monitoring():
    """Get cache performance monitoring data"""
    try:
        cache_stats = cache_manager.get_detailed_stats()

        return jsonify({
            'success': True,
            'cache_performance': cache_stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitoring/database', methods=['GET'])
def get_database_monitoring():
    """Get database performance monitoring data"""
    try:
        if not application_state.get('agent'):
            return jsonify({
                'success': False,
                'error': 'Agent not initialized'
            }), 400

        agent = application_state['agent']

        # Get database stats
        db_stats = agent.db_manager.query_stats.copy()
        connection_info = agent.db_manager.get_connection_info()

        # Get recent database activity
        recent_activity = []
        for query_result in application_state.get('query_history', [])[-10:]:
            if query_result.get('success'):
                recent_activity.append({
                    'timestamp': query_result['timestamp'],
                    'query_preview': query_result['query'][:50] + '...' if len(query_result['query']) > 50 else query_result['query'],
                    'execution_time': query_result.get('execution_time', 0)
                })

        return jsonify({
            'success': True,
            'database_stats': db_stats,
            'connection_info': connection_info,
            'recent_activity': recent_activity,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitoring/system', methods=['GET'])
def get_system_monitoring():
    """Get comprehensive system monitoring data"""
    try:
        import psutil

        # System metrics
        system_metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }

        # Application metrics
        process = psutil.Process()
        app_metrics = {
            'app_memory_mb': process.memory_info().rss / 1024 / 1024,
            'app_cpu_percent': process.cpu_percent(),
            'app_threads': process.num_threads(),
            'app_uptime_seconds': time.time() - process.create_time()
        }

        # Query metrics
        query_metrics = {
            'total_queries': len(application_state.get('query_history', [])),
            'active_queries': len(application_state.get('active_queries', {})),
            'recent_queries': len([q for q in application_state.get('query_history', [])
                                   if datetime.fromisoformat(q['timestamp']) > datetime.now() - timedelta(hours=1)])
        }

        return jsonify({
            'success': True,
            'system_metrics': system_metrics,
            'app_metrics': app_metrics,
            'query_metrics': query_metrics,
            'timestamp': datetime.now().isoformat()
        })

    except ImportError:
        return jsonify({
            'success': False,
            'error': 'psutil not available for system monitoring'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/configuration', methods=['GET'])
def get_configuration():
    """Get current agent configuration"""
    try:
        if not application_state.get('agent'):
            # Return default configuration
            return jsonify({
                'success': True,
                'configuration': asdict(config),
                'agent_initialized': False
            })

        agent = application_state['agent']

        # Get configuration details
        configuration = {
            'database': {
                'host': config.database_host,
                'database': config.database_name,
                'pool_size': config.database_pool_size,
                'connection_info': agent.db_manager.get_connection_info()
            },
            'llm': {
                'model': config.llm_model,
                'url': config.llm_url,
                'temperature': config.llm_temperature,
                'max_tokens': config.llm_max_tokens,
                'timeout': config.llm_timeout
            },
            'embeddings': {
                'model': config.embedding_model,
                'device': config.embedding_device,
                'max_length': config.embedding_max_length
            },
            'performance': {
                'max_workers': config.max_workers,
                'cache_ttl': config.cache_ttl,
                'chunk_size': config.chunk_size,
                'memory_monitoring': config.enable_memory_monitoring,
                'query_caching': config.enable_query_caching
            },
            'agent': {
                'verbose': config.agent_verbose,
                'max_iterations': config.max_iterations,
                'parallel_tools': config.agent_allow_parallel_tool_calls
            }
        }

        return jsonify({
            'success': True,
            'configuration': configuration,
            'agent_initialized': application_state.get('agent_initialized', False),
            'tools_count': len(agent.tools) if agent.tools else 0
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/configuration', methods=['POST'])
def update_configuration():
    """Update agent configuration (requires restart)"""
    try:
        data = request.get_json()

        # Update configuration
        # Note: This would require agent restart for most settings
        web_logger.add_log('INFO', 'Configuration update requested', 'Config')

        return jsonify({
            'success': True,
            'message': 'Configuration updated. Agent restart required for changes to take effect.',
            'restart_required': True
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/logs', methods=['GET'])
def get_system_logs():
    """Get system logs"""
    count = request.args.get('count', 50, type=int)
    level = request.args.get('level', '')
    component = request.args.get('component', '')

    logs = web_logger.get_recent_logs(count)

    # Filter by level and component if specified
    if level:
        logs = [log for log in logs if log['level'].upper() == level.upper()]

    if component:
        logs = [log for log in logs if component.lower() in log['component'].lower()]

    return jsonify({
        'success': True,
        'logs': logs,
        'total_logs': len(web_logger.logs),
        'filtered_count': len(logs)
    })

@app.route('/api/logs', methods=['DELETE'])
def clear_system_logs():
    """Clear system logs"""
    web_logger.clear_logs()
    web_logger.add_log('INFO', 'System logs cleared', 'System')

    return jsonify({
        'success': True,
        'message': 'System logs cleared'
    })

@app.route('/api/query-history', methods=['GET'])
def get_query_history():
    """Get query execution history"""
    limit = request.args.get('limit', 50, type=int)
    success_only = request.args.get('success_only', False, type=bool)

    history = application_state.get('query_history', [])

    if success_only:
        history = [q for q in history if q.get('success')]

    # Sort by timestamp (most recent first) and limit
    history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]

    return jsonify({
        'success': True,
        'query_history': history,
        'total_queries': len(application_state.get('query_history', [])),
        'filtered_count': len(history)
    })

@app.route('/api/query-history', methods=['DELETE'])
def clear_query_history():
    """Clear query history"""
    application_state['query_history'] = []
    web_logger.add_log('INFO', 'Query history cleared', 'System')

    return jsonify({
        'success': True,
        'message': 'Query history cleared'
    })

@app.route('/api/export', methods=['GET'])
def export_data():
    """Export application data"""
    try:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'agent_initialized': application_state.get('agent_initialized', False),
            'configuration': asdict(config),
            'query_history': application_state.get('query_history', []),
            'test_sessions': application_state.get('test_sessions', []),
            'system_logs': web_logger.get_recent_logs(1000),
            'tool_usage': tool_tracker.get_comprehensive_report() if application_state.get('agent_initialized') else {},
            'cache_performance': cache_manager.get_detailed_stats() if application_state.get('agent_initialized') else {},
            'statistics': {
                'total_queries': len(application_state.get('query_history', [])),
                'total_test_sessions': len(application_state.get('test_sessions', [])),
                'total_logs': len(web_logger.logs)
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(export_data, f, indent=2, default=str)
            temp_path = f.name

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'customer_service_agent_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/restart', methods=['POST'])
def restart_agent():
    """Restart the agent (reinitialize)"""
    try:
        # Clear current state
        with state_lock:
            application_state['agent'] = None
            application_state['agent_initialized'] = False
            application_state['initialization_status'] = {}

        agent_manager.agent = None
        agent_manager.is_initializing = False

        # Clear caches and trackers
        if 'agent' in application_state and application_state['agent']:
            cache_manager.clear_all_caches()
            tool_tracker.clear_history()

        web_logger.add_log('INFO', 'Agent restart initiated', 'System')

        return jsonify({
            'success': True,
            'message': 'Agent restart initiated. Use /api/initialize to start initialization.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Frontend HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Service AI Agent - Control Panel</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .status-indicator {
            animation: pulse 2s infinite;
        }
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
        }
        .metric-card {
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;

        // Main App Component
        function App() {
            const [activeTab, setActiveTab] = useState('dashboard');
            const [agentStatus, setAgentStatus] = useState({});
            const [isInitialized, setIsInitialized] = useState(false);
            const [isInitializing, setIsInitializing] = useState(false);
            const [initStatus, setInitStatus] = useState({});
            const [refreshInterval, setRefreshInterval] = useState(null);

            // Auto-refresh agent status
            useEffect(() => {
                loadAgentStatus();
                const interval = setInterval(loadAgentStatus, 5000);
                setRefreshInterval(interval);
                return () => clearInterval(interval);
            }, []);

            const loadAgentStatus = async () => {
                try {
                    const response = await fetch('/api/health');
                    const data = await response.json();
                    setAgentStatus(data);
                    setIsInitialized(data.agent_initialized || false);
                    setIsInitializing(data.initialization_in_progress || false);
                } catch (error) {
                    console.error('Error loading agent status:', error);
                }
            };

            const loadInitializationStatus = async () => {
                try {
                    const response = await fetch('/api/initialization-status');
                    const data = await response.json();
                    setInitStatus(data.status || {});
                    setIsInitializing(data.is_initializing || false);
                    setIsInitialized(data.initialized || false);
                } catch (error) {
                    console.error('Error loading initialization status:', error);
                }
            };

            // Poll initialization status when initializing
            useEffect(() => {
                let interval;
                if (isInitializing) {
                    interval = setInterval(loadInitializationStatus, 1000);
                }
                return () => {
                    if (interval) clearInterval(interval);
                };
            }, [isInitializing]);

            const initializeAgent = async () => {
                try {
                    const response = await fetch('/api/initialize', { method: 'POST' });
                    const data = await response.json();
                    if (data.success) {
                        setIsInitializing(true);
                    } else {
                        alert(data.message || 'Failed to start initialization');
                    }
                } catch (error) {
                    console.error('Error initializing agent:', error);
                    alert('Failed to initialize agent');
                }
            };

            const tabs = [
                { id: 'dashboard', label: 'Dashboard', icon: '📊' },
                { id: 'query', label: 'Query Interface', icon: '💬' },
                { id: 'scenarios', label: 'Test Scenarios', icon: '🧪' },
                { id: 'monitoring', label: 'Monitoring', icon: '📈' },
                { id: 'configuration', label: 'Configuration', icon: '⚙️' },
                { id: 'logs', label: 'System Logs', icon: '📋' }
            ];

            return (
                <div className="min-h-screen bg-gray-50">
                    {/* Header */}
                    <header className="gradient-bg text-white shadow-lg">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex justify-between h-16 items-center">
                                <div>
                                    <h1 className="text-xl font-bold">🤖 Customer Service AI Agent</h1>
                                    <p className="text-sm text-blue-100">Enhanced Multi-Purpose System Control Panel</p>
                                </div>
                                <div className="flex items-center space-x-4">
                                    <StatusIndicator 
                                        isInitialized={isInitialized} 
                                        isInitializing={isInitializing}
                                        status={agentStatus}
                                    />
                                    {!isInitialized && !isInitializing && (
                                        <button
                                            onClick={initializeAgent}
                                            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded font-medium"
                                        >
                                            🚀 Initialize Agent
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </header>

                    {/* Initialization Progress */}
                    {isInitializing && (
                        <InitializationProgress status={initStatus} />
                    )}

                    {/* Navigation */}
                    <nav className="bg-white shadow-sm border-b">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex space-x-8">
                                {tabs.map(tab => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                            activeTab === tab.id
                                                ? 'border-blue-500 text-blue-600'
                                                : 'border-transparent text-gray-500 hover:text-gray-700'
                                        }`}
                                    >
                                        <span className="mr-2">{tab.icon}</span>
                                        {tab.label}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </nav>

                    {/* Main Content */}
                    <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
                        {activeTab === 'dashboard' && <DashboardTab agentStatus={agentStatus} isInitialized={isInitialized} />}
                        {activeTab === 'query' && <QueryTab isInitialized={isInitialized} />}
                        {activeTab === 'scenarios' && <ScenariosTab isInitialized={isInitialized} />}
                        {activeTab === 'monitoring' && <MonitoringTab isInitialized={isInitialized} />}
                        {activeTab === 'configuration' && <ConfigurationTab />}
                        {activeTab === 'logs' && <LogsTab />}
                    </main>
                </div>
            );
        }

        // Status Indicator Component
        function StatusIndicator({ isInitialized, isInitializing, status }) {
            const getStatusColor = () => {
                if (isInitializing) return 'bg-yellow-500 status-indicator';
                if (isInitialized) return 'bg-green-500';
                return 'bg-red-500';
            };

            const getStatusText = () => {
                if (isInitializing) return 'Initializing...';
                if (isInitialized) return 'Ready';
                return 'Not Initialized';
            };

            return (
                <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
                    <span className="text-sm font-medium">{getStatusText()}</span>
                    {status.timestamp && (
                        <span className="text-xs text-blue-200">
                            Last update: {new Date(status.timestamp).toLocaleTimeString()}
                        </span>
                    )}
                </div>
            );
        }

        // Initialization Progress Component
        function InitializationProgress({ status }) {
            const progress = status.progress || 0;
            const totalSteps = status.total_steps || 5;
            const progressPercent = (progress / totalSteps) * 100;

            return (
                <div className="bg-blue-50 border-b border-blue-200">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-lg font-medium text-blue-900">🚀 Initializing Customer Service AI Agent</h3>
                            <span className="text-sm text-blue-700">{progress}/{totalSteps} steps completed</span>
                        </div>
                        
                        <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                            <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${progressPercent}%` }}
                            ></div>
                        </div>
                        
                        <div className="text-sm text-blue-700">
                            <span className="font-medium">Current step:</span> {status.current_step || 'Starting...'}
                        </div>
                        
                        {status.completed_steps && status.completed_steps.length > 0 && (
                            <details className="mt-2">
                                <summary className="text-sm text-blue-600 cursor-pointer">View completed steps</summary>
                                <ul className="mt-2 text-xs text-blue-600 space-y-1">
                                    {status.completed_steps.map((step, index) => (
                                        <li key={index}>✅ {step}</li>
                                    ))}
                                </ul>
                            </details>
                        )}
                    </div>
                </div>
            );
        }

        // Dashboard Tab Component
        function DashboardTab({ agentStatus, isInitialized }) {
            const [metrics, setMetrics] = useState({});
            const [toolUsage, setToolUsage] = useState({});
            const [cacheStats, setCacheStats] = useState({});

            useEffect(() => {
                if (isInitialized) {
                    loadDashboardData();
                    const interval = setInterval(loadDashboardData, 10000);
                    return () => clearInterval(interval);
                }
            }, [isInitialized]);

            const loadDashboardData = async () => {
                try {
                    // Load system metrics
                    const metricsResponse = await fetch('/api/monitoring/system');
                    if (metricsResponse.ok) {
                        const metricsData = await metricsResponse.json();
                        setMetrics(metricsData);
                    }

                    // Load tool usage
                    const toolResponse = await fetch('/api/monitoring/tools');
                    if (toolResponse.ok) {
                        const toolData = await toolResponse.json();
                        setToolUsage(toolData.tool_usage || {});
                    }

                    // Load cache stats
                    const cacheResponse = await fetch('/api/monitoring/cache');
                    if (cacheResponse.ok) {
                        const cacheData = await cacheResponse.json();
                        setCacheStats(cacheData.cache_performance || {});
                    }
                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            };

            if (!isInitialized) {
                return (
                    <div className="text-center py-12">
                        <div className="text-6xl mb-4">🤖</div>
                        <h2 className="text-2xl font-bold text-gray-900 mb-2">Customer Service AI Agent</h2>
                        <p className="text-gray-600 mb-6">
                            The agent is not initialized. Click the "Initialize Agent" button to start the system.
                        </p>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-2xl mx-auto">
                            <h3 className="font-medium text-blue-900 mb-2">What will be initialized:</h3>
                            <ul className="text-sm text-blue-700 space-y-1 text-left">
                                <li>✓ Database connection and schema validation</li>
                                <li>✓ LLM and embedding model configuration</li>
                                <li>✓ Support document indexing and processing</li>
                                <li>✓ AI agent tools and query engine setup</li>
                                <li>✓ Enhanced tracking and monitoring systems</li>
                            </ul>
                        </div>
                    </div>
                );
            }

            return (
                <div className="space-y-6">
                    {/* System Overview */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-4">🎯 System Overview</h2>
                        
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                            <MetricCard
                                title="System Health"
                                value={agentStatus.status === 'healthy' ? '🟢 Healthy' : '🔴 Issues'}
                                subtitle="Overall system status"
                                color="green"
                            />
                            <MetricCard
                                title="Total Queries"
                                value={metrics.query_metrics?.total_queries || 0}
                                subtitle="Queries processed"
                                color="blue"
                            />
                            <MetricCard
                                title="Active Queries"
                                value={metrics.query_metrics?.active_queries || 0}
                                subtitle="Currently processing"
                                color="yellow"
                            />
                            <MetricCard
                                title="Success Rate"
                                value={toolUsage.session_summary?.success_rate?.toFixed(1) + '%' || 'N/A'}
                                subtitle="Query success rate"
                                color="purple"
                            />
                        </div>
                    </div>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* System Resources */}
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">💻 System Resources</h3>
                            
                            {metrics.system_metrics && (
                                <div className="space-y-4">
                                    <ResourceBar
                                        label="CPU Usage"
                                        value={metrics.system_metrics.cpu_percent}
                                        max={100}
                                        color="blue"
                                        unit="%"
                                    />
                                    <ResourceBar
                                        label="Memory Usage"
                                        value={metrics.system_metrics.memory_percent}
                                        max={100}
                                        color="green"
                                        unit="%"
                                    />
                                    <ResourceBar
                                        label="Disk Usage"
                                        value={metrics.system_metrics.disk_usage_percent}
                                        max={100}
                                        color="yellow"
                                        unit="%"
                                    />
                                </div>
                            )}
                        </div>

                        {/* Cache Performance */}
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">💾 Cache Performance</h3>
                            
                            {cacheStats.global && (
                                <div className="space-y-4">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm font-medium text-gray-700">Hit Rate</span>
                                        <span className="text-lg font-bold text-green-600">
                                            {cacheStats.global.hit_rate?.toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm font-medium text-gray-700">Total Calls</span>
                                        <span className="text-lg font-bold text-blue-600">
                                            {cacheStats.global.total_calls}
                                        </span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm font-medium text-gray-700">Cache Hits</span>
                                        <span className="text-lg font-bold text-purple-600">
                                            {cacheStats.global.cache_hits}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Recent Activity */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">⚡ Recent Activity</h3>
                        <RecentActivity />
                    </div>
                </div>
            );
        }

        // Metric Card Component
        function MetricCard({ title, value, subtitle, color }) {
            const colorClasses = {
                green: 'bg-green-50 text-green-700 border-green-200',
                blue: 'bg-blue-50 text-blue-700 border-blue-200',
                yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
                purple: 'bg-purple-50 text-purple-700 border-purple-200'
            };

            return (
                <div className={`metric-card p-4 rounded-lg border ${colorClasses[color]}`}>
                    <div className="text-2xl font-bold">{value}</div>
                    <div className="text-sm font-medium">{title}</div>
                    <div className="text-xs opacity-75">{subtitle}</div>
                </div>
            );
        }

        // Resource Bar Component
        function ResourceBar({ label, value, max, color, unit }) {
            const percentage = (value / max) * 100;
            const colorClasses = {
                blue: 'bg-blue-500',
                green: 'bg-green-500',
                yellow: 'bg-yellow-500',
                red: 'bg-red-500'
            };

            return (
                <div>
                    <div className="flex justify-between text-sm">
                        <span className="font-medium text-gray-700">{label}</span>
                        <span className="text-gray-600">{value?.toFixed(1)}{unit}</span>
                    </div>
                    <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                        <div 
                            className={`h-2 rounded-full ${colorClasses[color]}`}
                            style={{ width: `${Math.min(percentage, 100)}%` }}
                        ></div>
                    </div>
                </div>
            );
        }

        // Recent Activity Component
        function RecentActivity() {
            const [queryHistory, setQueryHistory] = useState([]);
            const [loading, setLoading] = useState(true);

            useEffect(() => {
                loadQueryHistory();
            }, []);

            const loadQueryHistory = async () => {
                try {
                    const response = await fetch('/api/query-history?limit=10');
                    const data = await response.json();
                    if (data.success) {
                        setQueryHistory(data.query_history);
                    }
                } catch (error) {
                    console.error('Error loading query history:', error);
                } finally {
                    setLoading(false);
                }
            };

            if (loading) {
                return <div className="text-center py-4">Loading recent activity...</div>;
            }

            if (queryHistory.length === 0) {
                return (
                    <div className="text-center py-8 text-gray-500">
                        <div className="text-4xl mb-2">📭</div>
                        <p>No recent queries. Start by running a query or test scenario.</p>
                    </div>
                );
            }

            return (
                <div className="space-y-3">
                    {queryHistory.map((query, index) => (
                        <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded">
                            <div className={`w-2 h-2 mt-2 rounded-full ${query.success ? 'bg-green-500' : 'bg-red-500'}`}></div>
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-900 truncate">
                                    {query.query}
                                </p>
                                <div className="flex items-center space-x-4 text-xs text-gray-500 mt-1">
                                    <span>{new Date(query.timestamp).toLocaleString()}</span>
                                    <span>{query.execution_time?.toFixed(2)}s</span>
                                    <span>{query.execution_type || 'standard'}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            );
        }

        // Query Tab Component
        function QueryTab({ isInitialized }) {
            const [query, setQuery] = useState('');
            const [executionType, setExecutionType] = useState('standard');
            const [loading, setLoading] = useState(false);
            const [result, setResult] = useState(null);
            const [activeQueries, setActiveQueries] = useState({});

            const predefinedQueries = [
            "What is the return policy for order number 1001?",
            "Show me all orders for customer john.smith@email.com",
            "What items are in order 1007?",
            "When was order 1003 delivered?",
            "What are the warranty details for laptops?",
            "How can I contact customer support?",
            "What's the return policy for mice and keyboards?",
            "Analyze customer orders for Auburn, AL customers",
            "Calculate return deadlines for order 1022",
            "Generate predictive risk analysis for customer service"
        ];

        useEffect(() => {
            loadActiveQueries();
            const interval = setInterval(loadActiveQueries, 2000);
            return () => clearInterval(interval);
        }, []);

        const loadActiveQueries = async () => {
            try {
                const response = await fetch('/api/query-history?limit=5');
                const data = await response.json();
                if (data.success) {
                    // Mock active queries for demonstration
                    setActiveQueries({});
                }
            } catch (error) {
                console.error('Error loading active queries:', error);
            }
        };

        const executeQuery = async () => {
            if (!query.trim() || loading) return;

            setLoading(true);
            setResult(null);

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query.trim(),
                        execution_type: executionType
                    })
                });

                const data = await response.json();
                setResult(data);
            } catch (error) {
                setResult({
                    success: false,
                    error: error.message
                });
            } finally {
                setLoading(false);
            }
        };

        if (!isInitialized) {
            return (
                <div className="text-center py-12">
                    <div className="text-4xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-gray-900 mb-2">Agent Not Initialized</h2>
                    <p className="text-gray-600">Please initialize the agent first to use the query interface.</p>
                </div>
            );
        }

        return (
            <div className="space-y-6">
                {/* Query Input */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">💬 Query Interface</h2>
                    
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Enter your query
                            </label>
                            <textarea
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                rows={3}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="Ask a question about orders, customers, returns, or policies..."
                                disabled={loading}
                            />
                        </div>

                        <div className="flex items-center space-x-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Execution Type
                                </label>
                                <select
                                    value={executionType}
                                    onChange={(e) => setExecutionType(e.target.value)}
                                    className="border border-gray-300 rounded-md px-3 py-2 text-sm"
                                    disabled={loading}
                                >
                                    <option value="standard">Standard</option>
                                    <option value="enhanced">Enhanced Tracking</option>
                                    <option value="intelligent">Intelligent Decomposition</option>
                                </select>
                            </div>

                            <button
                                onClick={executeQuery}
                                disabled={loading || !query.trim()}
                                className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {loading ? (
                                    <>
                                        <div className="loading-spinner inline-block mr-2"></div>
                                        Processing...
                                    </>
                                ) : (
                                    '🚀 Execute Query'
                                )}
                            </button>
                        </div>

                        {/* Predefined Queries */}
                        <div>
                            <p className="text-sm text-gray-700 mb-2">Or try a predefined query:</p>
                            <div className="flex flex-wrap gap-2">
                                {predefinedQueries.map((predefinedQuery, index) => (
                                    <button
                                        key={index}
                                        onClick={() => setQuery(predefinedQuery)}
                                        disabled={loading}
                                        className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
                                    >
                                        {predefinedQuery.length > 60 ? predefinedQuery.substring(0, 60) + '...' : predefinedQuery}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Query Result */}
                {result && (
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">
                            {result.success ? '✅ Query Result' : '❌ Query Failed'}
                        </h3>
                        
                        {result.success ? (
                            <div className="space-y-4">
                                {/* Response */}
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                    <h4 className="font-medium text-blue-900 mb-2">Response:</h4>
                                    <p className="text-blue-800 whitespace-pre-wrap">{result.response}</p>
                                </div>

                                {/* Metadata */}
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                    <div className="bg-gray-50 p-3 rounded">
                                        <span className="font-medium text-gray-700">Execution Time:</span>
                                        <div className="text-lg font-bold text-green-600">
                                            {result.execution_time?.toFixed(2)}s
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 p-3 rounded">
                                        <span className="font-medium text-gray-700">Execution Type:</span>
                                        <div className="text-lg font-bold text-blue-600">
                                            {result.execution_type}
                                        </div>
                                    </div>
                                    <div className="bg-gray-50 p-3 rounded">
                                        <span className="font-medium text-gray-700">Query ID:</span>
                                        <div className="text-sm font-mono text-gray-600">
                                            {result.query_id}
                                        </div>
                                    </div>
                                </div>

                                {/* Tool Usage */}
                                {result.tool_usage && (
                                    <details className="border border-gray-200 rounded">
                                        <summary className="p-3 cursor-pointer font-medium">
                                            🔧 Tool Usage Details
                                        </summary>
                                        <div className="p-3 border-t border-gray-200 bg-gray-50">
                                            <div className="grid grid-cols-2 gap-4 mb-3">
                                                <div>
                                                    <span className="text-sm font-medium">Tools Called:</span>
                                                    <span className="ml-2 font-bold">{result.tool_usage.tools_called}</span>
                                                </div>
                                                <div>
                                                    <span className="text-sm font-medium">Success Rate:</span>
                                                    <span className="ml-2 font-bold">{result.tool_usage.success_rate?.toFixed(1)}%</span>
                                                </div>
                                            </div>
                                            {result.tool_usage.recent_calls && result.tool_usage.recent_calls.length > 0 && (
                                                <div>
                                                    <p className="text-sm font-medium mb-2">Recent Tool Calls:</p>
                                                    <div className="space-y-1">
                                                        {result.tool_usage.recent_calls.map((call, index) => (
                                                            <div key={index} className="text-xs bg-white p-2 rounded border">
                                                                <span className={`inline-block w-2 h-2 rounded-full mr-2 ${call.success ? 'bg-green-500' : 'bg-red-500'}`}></span>
                                                                <span className="font-medium">{call.tool_name}</span>
                                                                <span className="text-gray-500 ml-2">({call.execution_time?.toFixed(3)}s)</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </details>
                                )}
                            </div>
                        ) : (
                            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                <h4 className="font-medium text-red-900 mb-2">Error:</h4>
                                <p className="text-red-800">{result.error}</p>
                                {result.execution_time && (
                                    <p className="text-sm text-red-600 mt-2">
                                        Failed after {result.execution_time?.toFixed(2)}s
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    }

    // Scenarios Tab Component
    function ScenariosTab({ isInitialized }) {
        const [scenarios, setScenarios] = useState([]);
        const [testSessions, setTestSessions] = useState([]);
        const [runningTest, setRunningTest] = useState(null);
        const [selectedDifficulty, setSelectedDifficulty] = useState('all');
        const [executionType, setExecutionType] = useState('standard');

        useEffect(() => {
            loadScenarios();
            loadTestSessions();
        }, []);

        const loadScenarios = async () => {
            try {
                const response = await fetch('/api/scenarios');
                const data = await response.json();
                setScenarios(data.scenarios || []);
            } catch (error) {
                console.error('Error loading scenarios:', error);
            }
        };

        const loadTestSessions = async () => {
            try {
                const response = await fetch('/api/test-sessions');
                const data = await response.json();
                setTestSessions(data.test_sessions || []);
            } catch (error) {
                console.error('Error loading test sessions:', error);
            }
        };

        const runScenario = async (scenarioId) => {
            try {
                const response = await fetch('/api/scenarios/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        scenario_id: scenarioId,
                        execution_type: executionType
                    })
                });
                const data = await response.json();
                
                if (data.success) {
                    alert(`Scenario completed in ${data.execution_time?.toFixed(2)}s`);
                } else {
                    alert(`Scenario failed: ${data.error}`);
                }
            } catch (error) {
                alert(`Error running scenario: ${error.message}`);
            }
        };

        const runTestSuite = async (testType) => {
            setRunningTest(testType);
            try {
                const response = await fetch('/api/test-suite', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        test_type: testType,
                        execution_type: executionType
                    })
                });
                const data = await response.json();
                
                if (data.success) {
                    await loadTestSessions();
                    alert(`Test suite completed: ${data.test_session.summary.successful_tests}/${data.test_session.summary.total_tests} passed`);
                } else {
                    alert(`Test suite failed: ${data.error}`);
                }
            } catch (error) {
                alert(`Error running test suite: ${error.message}`);
            } finally {
                setRunningTest(null);
            }
        };

        if (!isInitialized) {
            return (
                <div className="text-center py-12">
                    <div className="text-4xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-gray-900 mb-2">Agent Not Initialized</h2>
                    <p className="text-gray-600">Please initialize the agent first to run test scenarios.</p>
                </div>
            );
        }

        const filteredScenarios = selectedDifficulty === 'all' 
            ? scenarios 
            : scenarios.filter(s => s.difficulty.toLowerCase() === selectedDifficulty.toLowerCase());

        const difficulties = ['all', ...new Set(scenarios.map(s => s.difficulty))];

        return (
            <div className="space-y-6">
                {/* Test Suite Controls */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">🧪 Test Scenarios</h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Quick Test Suites */}
                        <div>
                            <h3 className="font-medium text-gray-900 mb-3">Quick Test Suites</h3>
                            <div className="space-y-2">
                                {['basic', 'intermediate', 'advanced', 'expert', 'all'].map(testType => (
                                    <button
                                        key={testType}
                                        onClick={() => runTestSuite(testType)}
                                        disabled={runningTest !== null}
                                        className={`w-full text-left p-3 rounded border hover:bg-gray-50 disabled:opacity-50 ${
                                            runningTest === testType ? 'bg-blue-50 border-blue-300' : 'border-gray-200'
                                        }`}
                                    >
                                        <div className="flex justify-between items-center">
                                            <span className="font-medium capitalize">{testType} Tests</span>
                                            {runningTest === testType ? (
                                                <div className="loading-spinner"></div>
                                            ) : (
                                                <span className="text-sm text-gray-500">
                                                    {scenarios.filter(s => testType === 'all' || s.difficulty.toLowerCase() === testType).length} scenarios
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-gray-600 mt-1">
                                            {testType === 'basic' && 'Simple functionality tests'}
                                            {testType === 'intermediate' && 'Multi-step analysis tests'}
                                            {testType === 'advanced' && 'Complex calculation tests'}
                                            {testType === 'expert' && 'Multi-customer resolution tests'}
                                            {testType === 'all' && 'Complete test suite'}
                                        </p>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Execution Settings */}
                        <div>
                            <h3 className="font-medium text-gray-900 mb-3">Execution Settings</h3>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">
                                        Execution Type
                                    </label>
                                    <select
                                        value={executionType}
                                        onChange={(e) => setExecutionType(e.target.value)}
                                        className="w-full border border-gray-300 rounded-md px-3 py-2"
                                    >
                                        <option value="standard">Standard Execution</option>
                                        <option value="enhanced">Enhanced Tracking</option>
                                        <option value="intelligent">Intelligent Decomposition</option>
                                    </select>
                                </div>

                                <div className="bg-blue-50 p-3 rounded">
                                    <h4 className="text-sm font-medium text-blue-900 mb-1">Execution Types:</h4>
                                    <ul className="text-xs text-blue-800 space-y-1">
                                        <li><strong>Standard:</strong> Basic query execution</li>
                                        <li><strong>Enhanced:</strong> Detailed tool usage tracking</li>
                                        <li><strong>Intelligent:</strong> Query decomposition for complex queries</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Individual Scenarios */}
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-medium text-gray-900">Individual Scenarios</h3>
                        <select
                            value={selectedDifficulty}
                            onChange={(e) => setSelectedDifficulty(e.target.value)}
                            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
                        >
                            {difficulties.map(difficulty => (
                                <option key={difficulty} value={difficulty}>
                                    {difficulty === 'all' ? 'All Difficulties' : `${difficulty} (${scenarios.filter(s => s.difficulty.toLowerCase() === difficulty.toLowerCase()).length})`}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {filteredScenarios.map(scenario => (
                            <div key={scenario.id} className="border border-gray-200 rounded-lg p-4">
                                <div className="flex justify-between items-start mb-2">
                                    <h4 className="font-medium text-gray-900">{scenario.name}</h4>
                                    <span className={`px-2 py-1 text-xs rounded-full ${
                                        scenario.difficulty === 'Basic' ? 'bg-green-100 text-green-800' :
                                        scenario.difficulty === 'Intermediate' ? 'bg-blue-100 text-blue-800' :
                                        scenario.difficulty === 'Advanced' ? 'bg-yellow-100 text-yellow-800' :
                                        'bg-red-100 text-red-800'
                                    }`}>
                                        {scenario.difficulty}
                                    </span>
                                </div>
                                <p className="text-sm text-gray-600 mb-3">{scenario.description}</p>
                                <div className="bg-gray-50 p-2 rounded text-xs font-mono mb-3">
                                    {scenario.query}
                                </div>
                                <button
                                    onClick={() => runScenario(scenario.id)}
                                    className="w-full bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700"
                                >
                                    🏃‍♂️ Run Scenario
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Test Sessions History */}
                {testSessions.length > 0 && (
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">📊 Test Session History</h3>
                        
                        <div className="space-y-4">
                            {testSessions.slice(-5).reverse().map(session => (
                                <div key={session.session_id} className="border border-gray-200 rounded-lg p-4">
                                    <div className="flex justify-between items-start mb-2">
                                        <div>
                                            <h4 className="font-medium text-gray-900">
                                                {session.test_type.charAt(0).toUpperCase() + session.test_type.slice(1)} Test Suite
                                            </h4>
                                            <p className="text-sm text-gray-600">
                                                {new Date(session.start_time).toLocaleString()}
                                            </p>
                                        </div>
                                        <span className={`px-2 py-1 text-xs rounded-full ${
                                            session.status === 'completed' ? 'bg-green-100 text-green-800' :
                                            session.status === 'running' ? 'bg-blue-100 text-blue-800' :
                                            'bg-red-100 text-red-800'
                                        }`}>
                                            {session.status}
                                        </span>
                                    </div>
                                    
                                    {session.summary && (
                                        <div className="grid grid-cols-4 gap-4 text-sm">
                                            <div>
                                                <span className="text-gray-600">Total:</span>
                                                <div className="font-bold">{session.summary.total_tests}</div>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Passed:</span>
                                                <div className="font-bold text-green-600">{session.summary.successful_tests}</div>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Success Rate:</span>
                                                <div className="font-bold text-blue-600">{session.summary.success_rate?.toFixed(1)}%</div>
                                            </div>
                                            <div>
                                                <span className="text-gray-600">Avg Time:</span>
                                                <div className="font-bold text-purple-600">{session.summary.average_execution_time?.toFixed(2)}s</div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    }

    // Monitoring Tab Component
    function MonitoringTab({ isInitialized }) {
        const [toolUsage, setToolUsage] = useState({});
        const [cachePerformance, setCachePerformance] = useState({});
        const [databaseMetrics, setDatabaseMetrics] = useState({});
        const [systemMetrics, setSystemMetrics] = useState({});
        const [loading, setLoading] = useState(true);

        useEffect(() => {
            if (isInitialized) {
                loadMonitoringData();
                const interval = setInterval(loadMonitoringData, 10000);
                return () => clearInterval(interval);
            }
        }, [isInitialized]);

        const loadMonitoringData = async () => {
            try {
                // Load all monitoring data
                const [toolResponse, cacheResponse, dbResponse, systemResponse] = await Promise.all([
                    fetch('/api/monitoring/tools'),
                    fetch('/api/monitoring/cache'),
                    fetch('/api/monitoring/database'),
                    fetch('/api/monitoring/system')
                ]);

                if (toolResponse.ok) {
                    const toolData = await toolResponse.json();
                    setToolUsage(toolData.tool_usage || {});
                }

                if (cacheResponse.ok) {
                    const cacheData = await cacheResponse.json();
                    setCachePerformance(cacheData.cache_performance || {});
                }

                if (dbResponse.ok) {
                    const dbData = await dbResponse.json();
                    setDatabaseMetrics(dbData);
                }

                if (systemResponse.ok) {
                    const systemData = await systemResponse.json();
                    setSystemMetrics(systemData);
                }
            } catch (error) {
                console.error('Error loading monitoring data:', error);
            } finally {
                setLoading(false);
            }
        };

        if (!isInitialized) {
            return (
                <div className="text-center py-12">
                    <div className="text-4xl mb-4">⚠️</div>
                    <h2 className="text-xl font-bold text-gray-900 mb-2">Agent Not Initialized</h2>
                    <p className="text-gray-600">Please initialize the agent first to view monitoring data.</p>
                </div>
            );
        }

        if (loading) {
            return (
                <div className="text-center py-12">
                    <div className="loading-spinner mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading monitoring data...</p>
                </div>
            );
        }

        return (
            <div className="space-y-6">
                {/* System Performance */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">💻 System Performance</h2>
                    
                    {systemMetrics.success && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {/* System Resources */}
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">System Resources</h3>
                                <div className="space-y-3">
                                    <ResourceBar
                                        label="CPU Usage"
                                        value={systemMetrics.system_metrics?.cpu_percent || 0}
                                        max={100}
                                        color="blue"
                                        unit="%"
                                    />
                                    <ResourceBar
                                        label="Memory Usage"
                                        value={systemMetrics.system_metrics?.memory_percent || 0}
                                        max={100}
                                        color="green"
                                        unit="%"
                                    />
                                    <ResourceBar
                                        label="Disk Usage"
                                        value={systemMetrics.system_metrics?.disk_usage_percent || 0}
                                        max={100}
                                        color="yellow"
                                        unit="%"
                                    />
                                </div>
                            </div>

                            {/* Application Metrics */}
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">Application Metrics</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span>App Memory:</span>
                                        <span className="font-bold">{systemMetrics.app_metrics?.app_memory_mb?.toFixed(1)} MB</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>App CPU:</span>
                                        <span className="font-bold">{systemMetrics.app_metrics?.app_cpu_percent?.toFixed(1)}%</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Threads:</span>
                                        <span className="font-bold">{systemMetrics.app_metrics?.app_threads}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Uptime:</span>
                                        <span className="font-bold">
                                            {Math.floor((systemMetrics.app_metrics?.app_uptime_seconds || 0) / 3600)}h {Math.floor(((systemMetrics.app_metrics?.app_uptime_seconds || 0) % 3600) / 60)}m
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Query Metrics */}
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">Query Metrics</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span>Total Queries:</span>
                                        <span className="font-bold">{systemMetrics.query_metrics?.total_queries}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Active Queries:</span>
                                        <span className="font-bold">{systemMetrics.query_metrics?.active_queries}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Recent (1h):</span>
                                        <span className="font-bold">{systemMetrics.query_metrics?.recent_queries}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Tool Usage Analytics */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">🔧 Tool Usage Analytics</h2>
                    
                    {toolUsage.session_summary && (
                        <div>
                            {/* Session Overview */}
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                                <MetricCard
                                    title="Total Tool Calls"
                                    value={toolUsage.session_summary.total_tool_calls}
                                    subtitle="Since session start"
                                    color="blue"
                                />
                                <MetricCard
                                    title="Success Rate"
                                    value={`${toolUsage.session_summary.success_rate?.toFixed(1)}%`}
                                    subtitle="Tool call success"
                                    color="green"
                                />
                                <MetricCard
                                    title="Tools Used"
                                    value={toolUsage.session_summary.unique_tools_used}
                                    subtitle="Unique tools"
                                    color="purple"
                                />
                                <MetricCard
                                    title="Call Rate"
                                    value={`${toolUsage.session_summary.calls_per_minute?.toFixed(1)}/min`}
                                    subtitle="Calls per minute"
                                    color="yellow"
                                />
                            </div>

                            {/* Top Tools */}
                            {toolUsage.tool_ranking && (
                                <div className="mb-6">
                                    <h3 className="font-medium text-gray-900 mb-3">🏆 Most Used Tools</h3>
                                    <div className="space-y-2">
                                        {toolUsage.tool_ranking.slice(0, 5).map(([toolName, callCount], index) => (
                                            <div key={toolName} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                                                <div className="flex items-center space-x-3">
                                                    <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                                                        {index + 1}
                                                    </span>
                                                    <span className="font-medium">{toolName}</span>
                                                </div>
                                                <div className="text-right">
                                                    <div className="font-bold text-blue-600">{callCount} calls</div>
                                                    {toolUsage.performance_analysis && toolUsage.performance_analysis[toolName] && (
                                                        <div className="text-xs text-gray-500">
                                                            {toolUsage.performance_analysis[toolName].efficiency_score?.toFixed(1)}% success
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Recent Activity */}
                            {toolUsage.recent_activity && (
                                <div>
                                    <h3 className="font-medium text-gray-900 mb-3">⚡ Recent Tool Activity</h3>
                                    <div className="space-y-2">
                                        {toolUsage.recent_activity.slice(0, 10).map((activity, index) => (
                                            <div key={index} className="flex items-start space-x-3 p-3 border border-gray-200 rounded">
                                                <div className={`w-2 h-2 rounded-full mt-2 ${activity.success ? 'bg-green-500' : 'bg-red-500'}`}></div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center justify-between">
                                                        <span className="font-medium text-sm">{activity.tool_name}</span>
                                                        <span className="text-xs text-gray-500">
                                                            {new Date(activity.timestamp).toLocaleTimeString()}
                                                        </span>
                                                    </div>
                                                    <p className="text-xs text-gray-600 truncate">{activity.result_summary}</p>
                                                    <div className="flex items-center space-x-4 text-xs text-gray-500 mt-1">
                                                        <span>{activity.execution_time?.toFixed(3)}s</span>
                                                        <span>{activity.caller_context}</span>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Cache Performance */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">💾 Cache Performance</h2>
                    
                    {cachePerformance.global && (
                        <div>
                            {/* Global Stats */}
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                                <MetricCard
                                    title="Hit Rate"
                                    value={`${cachePerformance.global.hit_rate?.toFixed(1)}%`}
                                    subtitle="Cache effectiveness"
                                    color="green"
                                />
                                <MetricCard
                                    title="Total Calls"
                                    value={cachePerformance.global.total_calls}
                                    subtitle="All cache queries"
                                    color="blue"
                                />
                                <MetricCard
                                    title="Cache Hits"
                                    value={cachePerformance.global.cache_hits}
                                    subtitle="Served from cache"
                                    color="purple"
                                />
                                <MetricCard
                                    title="Time Saved"
                                    value={`${cachePerformance.global.time_saved?.toFixed(1)}s`}
                                    subtitle="Estimated savings"
                                    color="yellow"
                                />
                            </div>

                            {/* Method-specific Performance */}
                            {cachePerformance.by_method && Object.keys(cachePerformance.by_method).length > 0 && (
                                <div>
                                    <h3 className="font-medium text-gray-900 mb-3">📊 Cache Performance by Method</h3>
                                    <div className="space-y-3">
                                        {Object.entries(cachePerformance.by_method).map(([method, stats]) => (
                                            <div key={method} className="border border-gray-200 rounded p-4">
                                                <div className="flex justify-between items-center mb-2">
                                                    <span className="font-medium">{method}</span>
                                                    <span className={`px-2 py-1 text-xs rounded-full ${
                                                        stats.hit_rate > 80 ? 'bg-green-100 text-green-800' :
                                                        stats.hit_rate > 60 ? 'bg-yellow-100 text-yellow-800' :
                                                        'bg-red-100 text-red-800'
                                                    }`}>
                                                        {stats.hit_rate?.toFixed(1)}% hit rate
                                                    </span>
                                                </div>
                                                <div className="grid grid-cols-4 gap-4 text-sm">
                                                    <div>
                                                        <span className="text-gray-600">Calls:</span>
                                                        <div className="font-bold">{stats.total_calls}</div>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-600">Hits:</span>
                                                        <div className="font-bold text-green-600">{stats.cache_hits}</div>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-600">Avg Time:</span>
                                                        <div className="font-bold">{stats.avg_execution_time?.toFixed(3)}s</div>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-600">Cache Size:</span>
                                                        <div className="font-bold">{stats.cache_size}</div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Database Performance */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-4">🗄️ Database Performance</h2>
                    
                    {databaseMetrics.success && databaseMetrics.database_stats && (
                        <div>
                            {/* Database Stats */}
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                                <MetricCard
                                    title="Total Queries"
                                    value={databaseMetrics.database_stats.total_queries}
                                    subtitle="All database queries"
                                    color="blue"
                                />
                                <MetricCard
                                    title="Success Rate"
                                    value={`${((databaseMetrics.database_stats.successful_queries / databaseMetrics.database_stats.total_queries) * 100).toFixed(1)}%`}
                                    subtitle="Query success rate"
                                    color="green"
                                />
                                <MetricCard
                                    title="Avg Query Time"
                                    value={`${databaseMetrics.database_stats.avg_query_time?.toFixed(3)}s`}
                                    subtitle="Average execution"
                                    color="purple"
                                />
                                <MetricCard
                                    title="Failed Queries"
                                    value={databaseMetrics.database_stats.failed_queries}
                                    subtitle="Error count"
                                    color="yellow"
                                />
                            </div>

                            {/* Connection Info */}
                            {databaseMetrics.connection_info && (
                                <div className="bg-gray-50 p-4 rounded mb-4">
                                    <h3 className="font-medium text-gray-900 mb-2">Connection Information</h3>
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <span className="text-gray-600">Connection Type:</span>
                                            <span className="ml-2 font-medium">{databaseMetrics.connection_info.type}</span>
                                        </div>
                                        <div>
                                            <span className="text-gray-600">Status:</span>
                                            <span className="ml-2 font-medium">{databaseMetrics.connection_info.status}</span>
                                        </div>
                                        {databaseMetrics.connection_info.pool_size && (
                                            <div>
                                                <span className="text-gray-600">Pool Size:</span>
                                                <span className="ml-2 font-medium">{databaseMetrics.connection_info.pool_size}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Recent Activity */}
                            {databaseMetrics.recent_activity && (
                                <div>
                                    <h3 className="font-medium text-gray-900 mb-3">📋 Recent Database Activity</h3>
                                    <div className="space-y-2">
                                        {databaseMetrics.recent_activity.map((activity, index) => (
                                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                                                <div>
                                                    <span className="font-medium text-sm">{activity.query_preview}</span>
                                                    <div className="text-xs text-gray-500">
                                                        {new Date(activity.timestamp).toLocaleString()}
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <div className="text-sm font-bold">{activity.execution_time?.toFixed(3)}s</div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        );
    }

    // Configuration Tab Component
    function ConfigurationTab() {
        const [configuration, setConfiguration] = useState({});
        const [loading, setLoading] = useState(true);

        useEffect(() => {
            loadConfiguration();
        }, []);

        const loadConfiguration = async () => {
            try {
                const response = await fetch('/api/configuration');
                const data = await response.json();
                if (data.success) {
                    setConfiguration(data.configuration);
                }
            } catch (error) {
                console.error('Error loading configuration:', error);
            } finally {
                setLoading(false);
            }
        };

        const restartAgent = async () => {
            if (!confirm('Are you sure you want to restart the agent? This will reset the current session.')) {
                return;
            }

            try {
                const response = await fetch('/api/restart', { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    alert('Agent restart initiated. Please reinitialize the agent.');
                    window.location.reload();
                } else {
                    alert(`Failed to restart agent: ${data.error}`);
                }
            } catch (error) {
                alert(`Error restarting agent: ${error.message}`);
            }
        };

        if (loading) {
            return (
                <div className="text-center py-12">
                    <div className="loading-spinner mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading configuration...</p>
                </div>
            );
        }

        return (
            <div className="space-y-6">
                {/* Current Configuration */}
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-lg font-medium text-gray-900">⚙️ Current Configuration</h2>
                        <button
                            onClick={restartAgent}
                            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
                        >
                            🔄 Restart Agent
                        </button>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Database Configuration */}
                        {configuration.database && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">🗄️ Database</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Host:</span>
                                        <span className="font-medium">{configuration.database.host}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Database:</span>
                                        <span className="font-medium">{configuration.database.database}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Pool Size:</span>
                                        <span className="font-medium">{configuration.database.pool_size}</span>
                                    </div>
                                    {configuration.database.connection_info && (
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Status:</span>
                                            <span className={`font-medium ${
                                                configuration.database.connection_info.status === 'active' ? 'text-green-600' : 'text-red-600'
                                            }`}>
                                                {configuration.database.connection_info.status}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* LLM Configuration */}
                        {configuration.llm && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">🤖 Language Model</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Model:</span>
                                        <span className="font-medium">{configuration.llm.model}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">URL:</span>
                                        <span className="font-medium">{configuration.llm.url}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Temperature:</span>
                                        <span className="font-medium">{configuration.llm.temperature}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Max Tokens:</span>
                                        <span className="font-medium">{configuration.llm.max_tokens}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Timeout:</span>
                                        <span className="font-medium">{configuration.llm.timeout}s</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Embeddings Configuration */}
                        {configuration.embeddings && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">🧠 Embeddings</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Model:</span>
                                        <span className="font-medium">{configuration.embeddings.model}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Device:</span>
                                        <span className="font-medium">{configuration.embeddings.device}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Max Length:</span>
                                        <span className="font-medium">{configuration.embeddings.max_length}</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Performance Configuration */}
                        {configuration.performance && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">⚡ Performance</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Max Workers:</span>
                                        <span className="font-medium">{configuration.performance.max_workers}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Cache TTL:</span>
                                        <span className="font-medium">{configuration.performance.cache_ttl}s</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Chunk Size:</span>
                                        <span className="font-medium">{configuration.performance.chunk_size}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Memory Monitoring:</span>
                                        <span className={`font-medium ${configuration.performance.memory_monitoring ? 'text-green-600' : 'text-red-600'}`}>
                                            {configuration.performance.memory_monitoring ? 'Enabled' : 'Disabled'}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Query Caching:</span>
                                        <span className={`font-medium ${configuration.performance.query_caching ? 'text-green-600' : 'text-red-600'}`}>
                                            {configuration.performance.query_caching ? 'Enabled' : 'Disabled'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Agent Configuration */}
                        {configuration.agent && (
                            <div>
                                <h3 className="font-medium text-gray-900 mb-3">🤖 Agent Settings</h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Verbose:</span>
                                        <span className={`font-medium ${configuration.agent.verbose ? 'text-green-600' : 'text-red-600'}`}>
                                            {configuration.agent.verbose ? 'Enabled' : 'Disabled'}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Max Iterations:</span>
                                        <span className="font-medium">{configuration.agent.max_iterations}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Parallel Tools:</span>
                                        <span className={`font-medium ${configuration.agent.parallel_tools ? 'text-green-600' : 'text-red-600'}`}>
                                            {configuration.agent.parallel_tools ? 'Enabled' : 'Disabled'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Tools Information */}
                        <div>
                            <h3 className="font-medium text-gray-900 mb-3">🔧 Tools</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-600">Tools Count:</span>
                                    <span className="font-medium">{configuration.tools_count || 0}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-600">Agent Initialized:</span>
                                    <span className={`font-medium ${configuration.agent_initialized ? 'text-green-600' : 'text-red-600'}`}>
                                        {configuration.agent_initialized ? 'Yes' : 'No'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Configuration Notes */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h3 className="font-medium text-blue-900 mb-3">📝 Configuration Notes</h3>
                    <ul className="text-sm text-blue-800 space-y-2">
                        <li>• Configuration changes require an agent restart to take effect</li>
                        <li>• Database connections use connection pooling for better performance</li>
                        <li>• LLM settings are loaded from the .env configuration file</li>
                        <li>• Memory monitoring and query caching improve performance</li>
                        <li>• Parallel tool execution can speed up complex queries</li>
                    </ul>
                </div>

                {/* Environment Variables */}
                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">🌍 Environment Variables</h3>
                    <div className="bg-gray-50 p-4 rounded">
                        <p className="text-sm text-gray-600 mb-3">
                            The agent uses environment variables from the .env file for configuration. 
                            Key variables include:
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                            <div>
                                <div className="font-bold text-gray-800 mb-1">Database:</div>
                                <div>DB_HOST, DB_USER, DB_PASSWORD</div>
                                <div>DB_NAME, DB_POOL_SIZE</div>
                            </div>
                            <div>
                                <div className="font-bold text-gray-800 mb-1">LLM:</div>
                                <div>LLM_URL, LLM_MODEL</div>
                                <div>LLM_TEMPERATURE, LLM_MAX_TOKENS</div>
                            </div>
                            <div>
                                <div className="font-bold text-gray-800 mb-1">Performance:</div>
                                <div>MAX_WORKERS, CACHE_TTL</div>
                                <div>CHUNK_SIZE, ENABLE_MONITORING</div>
                            </div>
                            <div>
                                <div className="font-bold text-gray-800 mb-1">Embeddings:</div>
                                <div>EMBEDDING_MODEL, EMBEDDING_DEVICE</div>
                                <div>EMBEDDING_MAX_LENGTH</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Logs Tab Component
    function LogsTab() {
        const [logs, setLogs] = useState([]);
        const [filters, setFilters] = useState({
            level: '',
            component: '',
            count: 100
        });
        const [loading, setLoading] = useState(true);
        const logsEndRef = useRef(null);

        useEffect(() => {
            loadLogs();
            const interval = setInterval(loadLogs, 5000);
            return () => clearInterval(interval);
        }, [filters]);

        useEffect(() => {
            scrollToBottom();
        }, [logs]);

        const loadLogs = async () => {
            try {
                const params = new URLSearchParams();
                if (filters.level) params.append('level', filters.level);
                if (filters.component) params.append('component', filters.component);
                params.append('count', filters.count.toString());

                const response = await fetch(`/api/logs?${params}`);
                const data = await response.json();
                if (data.success) {
                    setLogs(data.logs);
                }
            } catch (error) {
                console.error('Error loading logs:', error);
            } finally {
                setLoading(false);
            }
        };

        const clearLogs = async () => {
            if (!confirm('Are you sure you want to clear all system logs?')) {
                return;
            }

            try {
                const response = await fetch('/api/logs', { method: 'DELETE' });
                const data = await response.json();
                if (data.success) {
                    setLogs([]);
                }
            } catch (error) {
                console.error('Error clearing logs:', error);
            }
        };

        const scrollToBottom = () => {
            logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        };

        const getLevelColor = (level) => {
            switch (level.toUpperCase()) {
                case 'ERROR': return 'text-red-600 bg-red-50';
                case 'SUCCESS': return 'text-green-600 bg-green-50';
                case 'INFO': return 'text-blue-600 bg-blue-50';
                case 'WARNING': return 'text-yellow-600 bg-yellow-50';
                default: return 'text-gray-600 bg-gray-50';
            }
        };

        return (
            <div className="space-y-6">
                {/* Log Controls */}
                <div className="bg-white shadow rounded-lg p-6">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-lg font-medium text-gray-900">📋 System Logs</h2>
                        <div className="flex space-x-2">
                            <button
                                onClick={scrollToBottom}
                                className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
                            >
                                ⬇️ Scroll to Bottom
                            </button>
                            <button
                                onClick={clearLogs}
                                className="bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700"
                            >
                                🗑️ Clear Logs
                            </button>
                        </div>
                    </div>

                    {/* Filters */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Log Level
                            </label>
                            <select
                                value={filters.level}
                                onChange={(e) => setFilters(prev => ({ ...prev, level: e.target.value }))}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                            >
                                <option value="">All Levels</option>
                                <option value="ERROR">Error</option>
                                <option value="SUCCESS">Success</option>
                                <option value="INFO">Info</option>
                                <option value="WARNING">Warning</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Component
                            </label>
                            <select
                                value={filters.component}
                                onChange={(e) => setFilters(prev => ({ ...prev, component: e.target.value }))}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                            >
                                <option value="">All Components</option>
                                <option value="System">System</option>
                                <option value="Agent">Agent</option>
                                <option value="Database">Database</option>
                                <option value="LLM">LLM</option>
                                <option value="Query">Query</option>
                                <option value="TestSuite">Test Suite</option>
                                <option value="API">API</option>
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Max Entries
                            </label>
                            <select
                                value={filters.count}
                                onChange={(e) => setFilters(prev => ({ ...prev, count: parseInt(e.target.value) }))}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                            >
                                <option value="50">50</option>
                                <option value="100">100</option>
                                <option value="250">250</option>
                                <option value="500">500</option>
                            </select>
                        </div>

                        <div className="flex items-end">
                            <button
                                onClick={loadLogs}
                                className="w-full bg-gray-600 text-white px-3 py-2 rounded text-sm hover:bg-gray-700"
                            >
                                🔄 Refresh
                            </button>
                        </div>
                    </div>
                </div>

                {/* Logs Display */}
                <div className="bg-white shadow rounded-lg">
                    <div className="p-4 border-b border-gray-200">
                        <div className="flex justify-between items-center">
                            <h3 className="font-medium text-gray-900">
                                Log Entries ({logs.length})
                            </h3>
                            <div className="text-sm text-gray-500">
                                Auto-refresh every 5 seconds
                            </div>
                        </div>
                    </div>

                    <div className="max-h-96 overflow-y-auto">
                        {loading ? (
                            <div className="text-center py-8">
                                <div className="loading-spinner mx-auto mb-4"></div>
                                <p className="text-gray-600">Loading logs...</p>
                            </div>
                        ) : logs.length === 0 ? (
                            <div className="text-center py-8 text-gray-500">
                                <div className="text-4xl mb-2">📭</div>
                                <p>No logs found with current filters</p>
                            </div>
                        ) : (
                            <div className="space-y-1 p-4">
                                {logs.map((log, index) => (
                                    <div
                                        key={index}
                                        className={`log-entry p-3 rounded border-l-4 ${
                                            log.level === 'ERROR' ? 'border-red-500 bg-red-50' :
                                            log.level === 'SUCCESS' ? 'border-green-500 bg-green-50' :
                                            log.level === 'INFO' ? 'border-blue-500 bg-blue-50' :
                                            log.level === 'WARNING' ? 'border-yellow-500 bg-yellow-50' :
                                            'border-gray-500 bg-gray-50'
                                        }`}
                                    >
                                        <div className="flex items-start justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center space-x-3 mb-1">
                                                    <span className={`px-2 py-1 text-xs rounded font-medium ${getLevelColor(log.level)}`}>
                                                        {log.level}
                                                    </span>
                                                    <span className="text-xs text-gray-500 font-medium">
                                                        {log.component}
                                                    </span>
                                                    <span className="text-xs text-gray-400">
                                                        {new Date(log.timestamp).toLocaleString()}
                                                    </span>
                                                </div>
                                                <div className="text-sm text-gray-900">
                                                    {log.message}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                <div ref={logsEndRef} />
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    // Render the main app
    ReactDOM.render(<App />, document.getElementById('root'));
    
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main frontend application"""
    return Response(HTML_TEMPLATE, mimetype='text/html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Cleanup function
def cleanup():
    """Clean up resources on shutdown"""
    try:
        if application_state.get('agent'):
            application_state['agent'].db_manager.disconnect()
        web_logger.add_log('INFO', 'Application shutdown completed', 'System')
    except Exception as e:
        print(f"Cleanup error: {e}")

# Additional utility functions
def get_system_info():
    """Get system information for debugging"""
    import platform
    import sys

    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'flask_version': '2.3.0',  # Update with actual version
        'timestamp': datetime.now().isoformat(),
        'working_directory': os.getcwd()
    }

# Development server setup
if __name__ == '__main__':
    print("🚀 Starting Customer Service AI Agent Web Interface...")
    print("=" * 80)
    print("🤖 CUSTOMER SERVICE AI AGENT - WEB CONTROL PANEL")
    print("=" * 80)
    print("🌐 Web Interface: http://localhost:5010")
    print("📊 Dashboard: http://localhost:5010")
    print("🔍 Health Check: http://localhost:5010/api/health")
    print("📚 API Documentation: Available via OpenAPI endpoints")
    print()
    print("✨ Features Available:")
    print("   ✅ Real-time Agent Monitoring")
    print("   ✅ Interactive Query Interface")
    print("   ✅ Comprehensive Test Scenarios")
    print("   ✅ Performance Analytics")
    print("   ✅ Tool Usage Tracking")
    print("   ✅ Cache Performance Monitoring")
    print("   ✅ Database Performance Metrics")
    print("   ✅ System Resource Monitoring")
    print("   ✅ Configuration Management")
    print("   ✅ Real-time System Logs")
    print("   ✅ Data Export Capabilities")
    print("   ✅ Agent Lifecycle Management")
    print()
    print("🔧 Agent Capabilities:")
    print("   • Enhanced Multi-Purpose System (MPS)")
    print("   • Intelligent Query Decomposition")
    print("   • Advanced Tool Usage Tracking")
    print("   • Predictive Customer Service Analysis")
    print("   • Comprehensive Return Policy Management")
    print("   • Multi-customer Dispute Resolution")
    print("   • Geographic Performance Analysis")
    print("   • Real-time Performance Monitoring")
    print()
    print("💡 Usage Instructions:")
    print("   1. Open http://localhost:5010 in your web browser")
    print("   2. Click 'Initialize Agent' to start the system")
    print("   3. Wait for initialization to complete")
    print("   4. Use the Dashboard to monitor system status")
    print("   5. Test queries in the Query Interface")
    print("   6. Run comprehensive tests in Test Scenarios")
    print("   7. Monitor performance in the Monitoring tab")
    print("=" * 80)
    print()

    try:
        # Register cleanup
        import atexit
        atexit.register(cleanup)

        # Initialize web logger
        web_logger.add_log('INFO', 'Customer Service AI Agent Web Interface starting...', 'System')
        web_logger.add_log('INFO', f'System info: {get_system_info()}', 'System')

        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5010,
            debug=True,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent duplicate processes
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        cleanup()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        web_logger.add_log('ERROR', f'Server error: {str(e)}', 'System')
        cleanup()
    finally:
        print("\n👋 Customer Service AI Agent Web Interface shutdown complete")

# Additional helper functions for the web interface

def validate_query_input(query):
    """Validate query input"""
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string"

    if len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long"

    if len(query) > 10000:
        return False, "Query is too long (max 10,000 characters)"

    return True, None

def format_execution_time(seconds):
    """Format execution time for display"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"

def generate_query_summary(result):
    """Generate a summary of query execution"""
    if not result:
        return "No result available"

    summary = []

    if result.get('success'):
        summary.append(f"✅ Query executed successfully")
        if result.get('execution_time'):
            summary.append(f"in {format_execution_time(result['execution_time'])}")
    else:
        summary.append(f"❌ Query failed")
        if result.get('error'):
            summary.append(f"- {result['error']}")

    if result.get('tool_usage'):
        tools_called = result['tool_usage'].get('tools_called', 0)
        if tools_called > 0:
            summary.append(f"🔧 {tools_called} tools used")

    return " ".join(summary)

def create_performance_report():
    """Create a comprehensive performance report"""
    if not application_state.get('agent_initialized'):
        return {"error": "Agent not initialized"}

    try:
        # Gather all performance data
        tool_report = tool_tracker.get_comprehensive_report()
        cache_stats = cache_manager.get_detailed_stats()

        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'session_duration': tool_report['session_summary']['session_duration_minutes'],
            'total_queries': len(application_state.get('query_history', [])),
            'tool_performance': {
                'total_calls': tool_report['session_summary']['total_tool_calls'],
                'success_rate': tool_report['session_summary']['success_rate'],
                'most_used_tools': tool_report['tool_ranking'][:5],
                'performance_analysis': tool_report['performance_analysis']
            },
            'cache_performance': {
                'global_hit_rate': cache_stats['global']['hit_rate'],
                'total_cache_calls': cache_stats['global']['total_calls'],
                'time_saved': cache_stats['global']['time_saved'],
                'method_performance': cache_stats['by_method']
            },
            'recommendations': []
        }

        # Generate recommendations
        if cache_stats['global']['hit_rate'] < 50:
            report['recommendations'].append("Consider increasing cache TTL to improve hit rate")

        if tool_report['session_summary']['success_rate'] < 95:
            report['recommendations'].append("Review failed tool calls to improve reliability")

        if tool_report['session_summary']['calls_per_minute'] > 10:
            report['recommendations'].append("High activity detected - monitor resource usage")

        return report

    except Exception as e:
        return {"error": f"Failed to generate report: {str(e)}"}

# WebSocket support for real-time updates (optional)
try:
    from flask_socketio import SocketIO, emit

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        emit('status', {
            'message': 'Connected to Customer Service AI Agent',
            'timestamp': datetime.now().isoformat()
        })
        web_logger.add_log('INFO', 'Client connected via WebSocket', 'WebSocket')

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        web_logger.add_log('INFO', 'Client disconnected from WebSocket', 'WebSocket')

    @socketio.on('request_status')
    def handle_status_request():
        """Handle real-time status request"""
        try:
            health_response = health_check()
            emit('status_update', health_response.get_json())
        except Exception as e:
            emit('status_error', {'error': str(e)})

    @socketio.on('request_logs')
    def handle_logs_request(data):
        """Handle real-time logs request"""
        try:
            count = data.get('count', 20)
            logs = web_logger.get_recent_logs(count)
            emit('logs_update', {'logs': logs})
        except Exception as e:
            emit('logs_error', {'error': str(e)})

    def broadcast_log_update(log_entry):
        """Broadcast log update to all connected clients"""
        try:
            socketio.emit('log_update', log_entry)
        except:
            pass  # Ignore WebSocket errors

    def broadcast_query_update(query_data):
        """Broadcast query update to all connected clients"""
        try:
            socketio.emit('query_update', query_data)
        except:
            pass  # Ignore WebSocket errors

    # Override web logger to broadcast updates
    original_add_log = web_logger.add_log
    def enhanced_add_log(level, message, component=None):
        log_entry = original_add_log(level, message, component)
        broadcast_log_update({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'component': component or 'System'
        })
        return log_entry

    web_logger.add_log = enhanced_add_log

    print("✅ WebSocket support enabled for real-time updates")

except ImportError:
    socketio = None
    print("⚠️  WebSocket support not available (install flask-socketio for real-time features)")

    def broadcast_log_update(log_entry):
        pass

    def broadcast_query_update(query_data):
        pass

# API Documentation
@app.route('/api/docs')
def api_documentation():
    """Serve API documentation"""
    docs_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Service AI Agent - API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #2563eb; }
            .path { font-family: monospace; background: #e5e7eb; padding: 2px 6px; border-radius: 3px; }
            .description { margin: 10px 0; }
            .example { background: #1f2937; color: #f9fafb; padding: 10px; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <h1>🤖 Customer Service AI Agent - API Documentation</h1>
        
        <h2>Base URL</h2>
        <p><code>http://localhost:5010</code></p>
        
        <h2>Core Endpoints</h2>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/health</span></div>
            <div class="description">Get comprehensive system health status</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/initialize</span></div>
            <div class="description">Initialize the Customer Service AI Agent</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/initialization-status</span></div>
            <div class="description">Get current initialization progress</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/query</span></div>
            <div class="description">Execute a query using the AI agent</div>
            <div class="example">
POST /api/query
{
  "query": "What is the return policy for order 1001?",
  "execution_type": "enhanced"
}
            </div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/scenarios</span></div>
            <div class="description">Get list of predefined test scenarios</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/scenarios/run</span></div>
            <div class="description">Run a specific test scenario</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/test-suite</span></div>
            <div class="description">Run comprehensive test suite</div>
        </div>
        
        <h2>Monitoring Endpoints</h2>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/monitoring/tools</span></div>
            <div class="description">Get tool usage monitoring data</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/monitoring/cache</span></div>
            <div class="description">Get cache performance monitoring data</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/monitoring/database</span></div>
            <div class="description">Get database performance monitoring data</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/monitoring/system</span></div>
            <div class="description">Get comprehensive system monitoring data</div>
        </div>
        
        <h2>Configuration & Management</h2>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/configuration</span></div>
            <div class="description">Get current agent configuration</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/logs</span></div>
            <div class="description">Get system logs with optional filtering</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">DELETE</span> <span class="path">/api/logs</span></div>
            <div class="description">Clear system logs</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">GET</span> <span class="path">/api/export</span></div>
            <div class="description">Export all application data as JSON</div>
        </div>
        
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="path">/api/restart</span></div>
            <div class="description">Restart the agent (reinitialize)</div>
        </div>
        
        <h2>Response Format</h2>
        <p>All API endpoints return JSON responses with the following structure:</p>
        <div class="example">
{
  "success": true/false,
  "data": {...},      // Response data (varies by endpoint)
  "error": "...",     // Error message (if success=false)
  "timestamp": "..."  // ISO timestamp
}
        </div>
        
        <h2>WebSocket Events (if enabled)</h2>
        <ul>
            <li><code>connect</code> - Client connection established</li>
            <li><code>status_update</code> - Real-time status updates</li>
            <li><code>log_update</code> - Real-time log entries</li>
            <li><code>query_update</code> - Real-time query results</li>
        </ul>
        
        <hr>
        <p><strong>For complete interactive testing, use the main web interface at <a href="http://localhost:5010">http://localhost:5010</a></strong></p>
    </body>
    </html>
    """
    return Response(docs_html, mimetype='text/html')

print("📚 Customer Service AI Agent Web Interface loaded successfully!")
print("🎯 Ready to serve requests on http://localhost:5010")
print("📖 API Documentation available at http://localhost:5010/api/docs")