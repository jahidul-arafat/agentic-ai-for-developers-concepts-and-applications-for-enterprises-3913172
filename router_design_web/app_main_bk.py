#!/usr/bin/env python3
"""
Complete Web Interface for Agent Router Application
Runs on port 5010 with full API and frontend capabilities
"""

import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import tempfile
import shutil

import requests
# Flask and web dependencies
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import uuid

# LlamaIndex dependencies
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import nest_asyncio

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Enable CORS for all routes
CORS(app)

# Global variables for the application state
application_state = {
    'llm': None,
    'embed_model': None,
    'indexes': {},
    'query_engines': {},
    'router_agent': None,
    'uploaded_files': {},
    'test_results': [],
    'configuration': {
        'llm_model': None,
        'embedding_model': None,
        'huggingface_token': None,
        'lm_studio_url': 'http://127.0.0.1:1234/v1'
    }
}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Available models configuration
AVAILABLE_LLM_MODELS = {
    "deepseek-coder-33b-instruct": "DeepSeek Coder 33B",
    "open_gpt4_8x7b_v0.2": "OpenGPT4 8x7B",
    "llama-3-groq-8b-tool-use": "Llama 3 Groq 8B Tool Use"
}

AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "HuggingFace MiniLM-L6-v2",
    "text-embedding-ada-002": "OpenAI Ada-002 (via local)"
}


# Router implementation classes
class SimpleSmartRouter:
    """Simple keyword-based router for reliability"""

    def __init__(self, engines_dict):
        self.engines = engines_dict
        self.query_count = 0
        self.routing_decisions = []

    def query(self, query_str):
        self.query_count += 1
        query_lower = query_str.lower()

        # Score each engine
        scores = {}
        decision_log = {
            'query': query_str,
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'decision': None,
            'reasoning': []
        }

        for engine_name in self.engines.keys():
            scores[engine_name] = 0

            # Check for explicit mentions
            if engine_name.lower().replace('_specifications', '') in query_lower:
                scores[engine_name] += 10
                decision_log['reasoning'].append(f"{engine_name} explicitly mentioned (+10)")

            # Check contextual keywords
            if 'aeroflow' in engine_name.lower():
                keywords = ['aero', 'flow', 'aerodynamic']
                for keyword in keywords:
                    if keyword in query_lower:
                        scores[engine_name] += 2
                        decision_log['reasoning'].append(f"AeroFlow keyword '{keyword}' found (+2)")

            elif 'ecosprint' in engine_name.lower():
                keywords = ['eco', 'sprint', 'environment', 'green']
                for keyword in keywords:
                    if keyword in query_lower:
                        scores[engine_name] += 2
                        decision_log['reasoning'].append(f"EcoSprint keyword '{keyword}' found (+2)")

        decision_log['scores'] = scores

        # Make decision
        best_engine = max(scores.keys(), key=lambda k: scores[k])
        max_score = scores[best_engine]

        # Handle ties or no clear winner
        if max_score == 0:
            # Check for comparison words
            comparison_words = ['better', 'compare', 'vs', 'versus', 'which', 'best']
            if any(word in query_lower for word in comparison_words):
                best_engine = list(self.engines.keys())[0]  # Default to first
                decision_log['reasoning'].append("Comparison query detected - using first engine")
            else:
                # Alternate based on query count
                engine_list = list(self.engines.keys())
                best_engine = engine_list[self.query_count % len(engine_list)]
                decision_log['reasoning'].append(f"Alternating selection - query #{self.query_count}")

        decision_log['decision'] = best_engine
        self.routing_decisions.append(decision_log)

        # Execute query
        try:
            response = self.engines[best_engine].query(query_str)
            return response
        except Exception as e:
            logger.error(f"Error executing query on {best_engine}: {e}")
            raise


class HybridRouter:
    """Router that tries LLM first, falls back to keywords"""

    def __init__(self, llm_router, keyword_router):
        self.llm_router = llm_router
        self.keyword_router = keyword_router
        self.llm_failures = 0
        self.routing_log = []

    def query(self, query_str):
        log_entry = {
            'query': query_str,
            'timestamp': datetime.now().isoformat(),
            'method_used': None,
            'success': False,
            'error': None
        }

        if self.llm_router and self.llm_failures < 3:
            try:
                logger.info("Trying LLM router...")
                response = self.llm_router.query(query_str)
                log_entry['method_used'] = 'LLM'
                log_entry['success'] = True
                self.routing_log.append(log_entry)
                return response
            except Exception as e:
                self.llm_failures += 1
                log_entry['error'] = str(e)
                logger.warning(f"LLM router failed ({self.llm_failures}/3): {e}")
                logger.info("Falling back to keyword router...")

        # Use keyword router
        try:
            response = self.keyword_router.query(query_str)
            log_entry['method_used'] = 'Keyword'
            log_entry['success'] = True
            self.routing_log.append(log_entry)
            return response
        except Exception as e:
            log_entry['error'] = str(e)
            log_entry['method_used'] = 'Keyword'
            self.routing_log.append(log_entry)
            raise


# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'llm_configured': application_state['llm'] is not None,
            'embedding_configured': application_state['embed_model'] is not None,
            'router_configured': application_state['router_agent'] is not None,
            'files_uploaded': len(application_state['uploaded_files'])
        }
    })


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get available LLM and embedding models"""
    return jsonify({
        'llm_models': AVAILABLE_LLM_MODELS,
        'embedding_models': AVAILABLE_EMBEDDING_MODELS
    })


@app.route('/api/configure', methods=['POST'])
def configure_models():
    """Configure LLM and embedding models"""
    try:
        data = request.get_json()

        llm_model = data.get('llm_model')
        embedding_model = data.get('embedding_model')
        huggingface_token = data.get('huggingface_token')
        lm_studio_url = data.get('lm_studio_url', 'http://127.0.0.1:1234/v1')

        # Update configuration
        application_state['configuration'].update({
            'llm_model': llm_model,
            'embedding_model': embedding_model,
            'huggingface_token': huggingface_token,
            'lm_studio_url': lm_studio_url
        })

        # Configure LLM
        if llm_model:
            application_state['llm'] = OpenAILike(
                model=llm_model,
                api_base=lm_studio_url,
                api_key="lm-studio",
                is_local=True,
                temperature=0.1,
                max_tokens=2048,
            )
            Settings.llm = application_state['llm']
            logger.info(f"Configured LLM: {llm_model}")

        # Configure embedding model
        if embedding_model:
            if embedding_model == "text-embedding-ada-002":
                application_state['embed_model'] = OpenAIEmbedding(
                    model="text-embedding-ada-002",
                    api_base=lm_studio_url,
                    api_key="lm-studio",
                )
            else:
                # Set HuggingFace token if provided
                if huggingface_token:
                    os.environ['HUGGINGFACE_HUB_TOKEN'] = huggingface_token

                application_state['embed_model'] = HuggingFaceEmbedding(
                    model_name=f"sentence-transformers/{embedding_model}"
                )

            Settings.embed_model = application_state['embed_model']
            logger.info(f"Configured embedding model: {embedding_model}")

        return jsonify({
            'success': True,
            'message': 'Models configured successfully',
            'configuration': application_state['configuration']
        })

    except Exception as e:
        logger.error(f"Error configuring models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process multiple files"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        uploaded_info = []

        for file in files:
            if file and file.filename:
                # Secure the filename
                filename = secure_filename(file.filename)
                file_id = str(uuid.uuid4())

                # Save file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
                file.save(file_path)

                # Store file info
                file_info = {
                    'id': file_id,
                    'original_name': file.filename,
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'uploaded_at': datetime.now().isoformat(),
                    'processed': False
                }

                application_state['uploaded_files'][file_id] = file_info
                uploaded_info.append(file_info)

                logger.info(f"Uploaded file: {filename} ({file_info['size']} bytes)")

        return jsonify({
            'success': True,
            'files': uploaded_info,
            'message': f'Successfully uploaded {len(uploaded_info)} files'
        })

    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large'}), 413
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    return jsonify({
        'files': list(application_state['uploaded_files'].values())
    })


@app.route('/api/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete an uploaded file"""
    try:
        if file_id not in application_state['uploaded_files']:
            return jsonify({'error': 'File not found'}), 404

        file_info = application_state['uploaded_files'][file_id]

        # Delete file from filesystem
        if os.path.exists(file_info['path']):
            os.remove(file_info['path'])

        # Remove from state
        del application_state['uploaded_files'][file_id]

        # Clean up associated indexes
        if file_id in application_state['indexes']:
            del application_state['indexes'][file_id]
        if file_id in application_state['query_engines']:
            del application_state['query_engines'][file_id]

        return jsonify({
            'success': True,
            'message': f'File {file_info["original_name"]} deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/create-indexes', methods=['POST'])
def create_indexes():
    """Create vector indexes for uploaded files"""
    try:
        if not application_state['embed_model']:
            return jsonify({'error': 'Embedding model not configured'}), 400

        data = request.get_json() or {}
        chunk_size = data.get('chunk_size', 1024)
        file_ids = data.get('file_ids', list(application_state['uploaded_files'].keys()))

        if not file_ids:
            return jsonify({'error': 'No files to process'}), 400

        splitter = SentenceSplitter(chunk_size=chunk_size)
        results = []

        for file_id in file_ids:
            if file_id not in application_state['uploaded_files']:
                continue

            file_info = application_state['uploaded_files'][file_id]

            try:
                # Load documents
                documents = SimpleDirectoryReader(
                    input_files=[file_info['path']]
                ).load_data()

                # Create nodes
                nodes = splitter.get_nodes_from_documents(documents)

                # Create vector index
                index = VectorStoreIndex(nodes)

                # Create query engine
                query_engine = index.as_query_engine()

                # Store in application state
                application_state['indexes'][file_id] = index
                application_state['query_engines'][file_id] = query_engine

                # Update file info
                file_info['processed'] = True
                file_info['nodes_count'] = len(nodes)
                file_info['chunk_size'] = chunk_size

                results.append({
                    'file_id': file_id,
                    'filename': file_info['original_name'],
                    'nodes_count': len(nodes),
                    'success': True
                })

                logger.info(f"Created index for {file_info['original_name']}: {len(nodes)} nodes")

            except Exception as e:
                logger.error(f"Error processing file {file_info['original_name']}: {e}")
                results.append({
                    'file_id': file_id,
                    'filename': file_info['original_name'],
                    'success': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results,
            'message': f'Processed {len([r for r in results if r["success"]])} files successfully'
        })

    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/create-router', methods=['POST'])
def create_router():
    """Create the router agent from processed indexes"""
    try:
        if not application_state['llm']:
            return jsonify({'error': 'LLM not configured'}), 400

        if not application_state['query_engines']:
            return jsonify({'error': 'No query engines available. Please create indexes first.'}), 400

        data = request.get_json() or {}
        router_type = data.get('router_type', 'hybrid')  # 'llm', 'keyword', 'hybrid'

        # Create tools from query engines
        tools = []
        for file_id, query_engine in application_state['query_engines'].items():
            file_info = application_state['uploaded_files'][file_id]
            tool_name = f"{Path(file_info['original_name']).stem}_specifications"

            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=tool_name,
                description=(
                    f"Use this tool for questions about {file_info['original_name']} specifications, "
                    f"including: design details, features, technology components, maintenance procedures, "
                    f"warranty information, performance metrics, and technical specifications."
                ),
            )
            tools.append(tool)

        # Create router based on type
        if router_type == 'llm':
            try:
                router_agent = RouterQueryEngine(
                    selector=LLMSingleSelector.from_defaults(llm=application_state['llm']),
                    query_engine_tools=tools,
                    verbose=True
                )
            except Exception as e:
                logger.warning(f"LLM router creation failed: {e}")
                return jsonify({'error': f'LLM router failed: {str(e)}'}), 500

        elif router_type == 'keyword':
            # Create simple keyword router
            engines_dict = {tool.metadata.name: tool.query_engine for tool in tools}
            router_agent = SimpleSmartRouter(engines_dict)

        else:  # hybrid
            # Try LLM router first
            llm_router = None
            try:
                llm_router = RouterQueryEngine(
                    selector=LLMSingleSelector.from_defaults(llm=application_state['llm']),
                    query_engine_tools=tools,
                    verbose=True
                )
            except Exception as e:
                logger.warning(f"LLM router creation failed, using keyword only: {e}")

            # Create keyword router
            engines_dict = {tool.metadata.name: tool.query_engine for tool in tools}
            keyword_router = SimpleSmartRouter(engines_dict)

            if llm_router:
                router_agent = HybridRouter(llm_router, keyword_router)
            else:
                router_agent = keyword_router

        application_state['router_agent'] = router_agent

        return jsonify({
            'success': True,
            'router_type': router_type,
            'tools_count': len(tools),
            'tool_names': [tool.metadata.name for tool in tools],
            'message': f'Router created successfully with {len(tools)} tools'
        })

    except Exception as e:
        logger.error(f"Error creating router: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_router():
    """Query the router agent"""
    try:
        if not application_state['router_agent']:
            return jsonify({'error': 'Router not configured'}), 400

        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Execute query
        start_time = datetime.now()
        response = application_state['router_agent'].query(query)
        end_time = datetime.now()

        response_time = (end_time - start_time).total_seconds()

        # Get routing information if available
        routing_info = {}
        if hasattr(application_state['router_agent'], 'routing_log'):
            routing_info = application_state['router_agent'].routing_log[-1] if application_state[
                'router_agent'].routing_log else {}
        elif hasattr(application_state['router_agent'], 'routing_decisions'):
            routing_info = application_state['router_agent'].routing_decisions[-1] if application_state[
                'router_agent'].routing_decisions else {}

        result = {
            'success': True,
            'query': query,
            'response': str(response),
            'response_time': response_time,
            'timestamp': start_time.isoformat(),
            'routing_info': routing_info
        }

        # Store in test results
        application_state['test_results'].append(result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/test', methods=['POST'])
def run_tests():
    """Enhanced comprehensive router tests with better error handling"""
    try:
        if not application_state['router_agent']:
            return jsonify({'error': 'Router not configured'}), 400

        data = request.get_json() or {}
        test_type = data.get('test_type', 'comprehensive')

        # Comprehensive test cases from the notebook
        test_cases = []

        if test_type == 'basic':
            test_cases = [
                {"query": "What are the key features?", "category": "Generic"},
                {"query": "Tell me about specifications", "category": "Generic"},
                {"query": "What is the warranty coverage?", "category": "Generic"}
            ]
        else:  # comprehensive - all test cases from notebook
            test_cases = [
                # Explicit Vehicle Mentions
                {"query": "What colors are available for AeroFlow?", "category": "Explicit - AeroFlow"},
                {"query": "Tell me about EcoSprint's battery specifications", "category": "Explicit - EcoSprint"},
                {"query": "How do I maintain my AeroFlow vehicle?", "category": "Explicit - AeroFlow"},
                {"query": "What is EcoSprint's top speed?", "category": "Explicit - EcoSprint"},

                # Ambiguous/Comparison Queries
                {"query": "Which vehicle has better performance?", "category": "Ambiguous - Comparison"},
                {"query": "What are the available color options?", "category": "Ambiguous - Generic"},
                {"query": "Compare the two electric vehicles", "category": "Ambiguous - Comparison"},
                {"query": "Which one is more environmentally friendly?", "category": "Ambiguous - Environmental"},

                # Contextual Keywords
                {"query": "Tell me about the eco-friendly features", "category": "Contextual - Eco"},
                {"query": "What about aerodynamic design?", "category": "Contextual - Aero"},
                {"query": "How green is this vehicle?", "category": "Contextual - Green"},
                {"query": "What about the flow dynamics?", "category": "Contextual - Flow"},

                # Technical Specifications
                {"query": "What is the battery capacity?", "category": "Technical - Battery"},
                {"query": "How long does charging take?", "category": "Technical - Charging"},
                {"query": "What safety features are included?", "category": "Technical - Safety"},
                {"query": "What is the warranty coverage?", "category": "Technical - Warranty"},

                # Additional edge cases
                {"query": "Compare AeroFlow and EcoSprint features", "category": "Explicit Comparison"},
                {"query": "Which has better range?", "category": "Comparison - Range"},
                {"query": "What are the maintenance requirements?", "category": "Generic - Maintenance"},
                {"query": "Tell me about pricing", "category": "Generic - Pricing"},

                # Contextual variations
                {"query": "What about the family-friendly features?", "category": "Contextual - Family"},
                {"query": "How efficient is the motor?", "category": "Technical - Efficiency"},
                {"query": "What charging options are available?", "category": "Technical - Charging Options"},
                {"query": "Tell me about the interior design", "category": "Technical - Design"}
            ]

        # Run tests with detailed intelligence tracking and better error handling
        results = []
        successful_tests = 0

        for i, test_case in enumerate(test_cases):
            try:
                print(f"🧪 Running test {i + 1}/{len(test_cases)}: {test_case['query']}")

                start_time = datetime.now()

                # Execute query with timeout protection
                try:
                    response = application_state['router_agent'].query(test_case['query'])

                    if response is None:
                        raise Exception("Router returned None response")

                    response_text = str(response)

                except Exception as query_error:
                    logger.warning(f"Query failed for test {i + 1}: {query_error}")
                    response_text = f"Query failed: {str(query_error)}"

                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()

                # Get routing intelligence safely
                routing_intelligence = {}
                try:
                    if hasattr(application_state['router_agent'], 'routing_log') and application_state[
                        'router_agent'].routing_log:
                        latest_log = application_state['router_agent'].routing_log[-1]
                        routing_intelligence = {
                            'method_used': latest_log.get('method_used', 'Unknown'),
                            'reasoning': latest_log.get('llm_reasoning') or latest_log.get('keyword_reasoning')
                        }
                    elif hasattr(application_state['router_agent'], 'routing_decisions') and application_state[
                        'router_agent'].routing_decisions:
                        latest_decision = application_state['router_agent'].routing_decisions[-1]
                        routing_intelligence = {
                            'method_used': 'Keyword',
                            'decision': latest_decision.get('decision', 'Unknown'),
                            'scores': latest_decision.get('score_breakdown', {}),
                            'reasoning_steps': latest_decision.get('reasoning_steps', []),
                            'final_reasoning': latest_decision.get('final_reasoning', 'No reasoning available')
                        }
                except Exception as routing_error:
                    logger.warning(f"Could not extract routing intelligence for test {i + 1}: {routing_error}")
                    routing_intelligence = {'method_used': 'Unknown', 'error': str(routing_error)}

                test_result = {
                    'test_id': i + 1,
                    'query': test_case['query'],
                    'category': test_case['category'],
                    'response': response_text,
                    'response_time': response_time,
                    'success': True,
                    'response_length': len(response_text),
                    'timestamp': start_time.isoformat(),
                    'routing_intelligence': routing_intelligence
                }

                results.append(test_result)
                successful_tests += 1

            except Exception as e:
                logger.error(f"Test {i + 1} failed with error: {e}")
                results.append({
                    'test_id': i + 1,
                    'query': test_case['query'],
                    'category': test_case['category'],
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # Calculate summary statistics
        summary = {
            'total_tests': len(results),
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / len(results) * 100) if results else 0,
            'average_response_time': sum(r.get('response_time', 0) for r in results if
                                         r.get('success')) / successful_tests if successful_tests > 0 else 0,
            'average_response_length': sum(r.get('response_length', 0) for r in results if
                                           r.get('success')) / successful_tests if successful_tests > 0 else 0,
            'categories_tested': len(set(r['category'] for r in results)),
            'routing_methods_used': list(set(r.get('routing_intelligence', {}).get('method_used') for r in results if
                                             r.get('success') and r.get('routing_intelligence', {}).get('method_used')))
        }

        test_session = {
            'test_type': test_type,
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'results': results
        }

        application_state['test_results'].append(test_session)

        print(f"✅ Test session completed: {successful_tests}/{len(results)} tests successful")

        return jsonify({
            'success': True,
            'test_session': test_session
        })

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return jsonify({'error': str(e)}), 500


# 3. Add new endpoint for routing analysis
@app.route('/api/routing-analysis', methods=['GET'])
def get_routing_analysis():
    """Get detailed routing analysis and statistics"""
    try:
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'router_type': type(application_state['router_agent']).__name__ if application_state[
                'router_agent'] else None,
            'total_queries': 0,
            'routing_statistics': {},
            'recent_decisions': []
        }

        if application_state['router_agent']:
            # Get routing logs based on router type
            if hasattr(application_state['router_agent'], 'routing_log'):
                # Hybrid router
                routing_log = application_state['router_agent'].routing_log
                analysis['total_queries'] = len(routing_log)
                analysis['llm_failures'] = application_state['router_agent'].llm_failures

                # Statistics
                methods_used = [log.get('method_used') for log in routing_log]
                analysis['routing_statistics'] = {
                    'llm_usage': methods_used.count('LLM'),
                    'keyword_usage': methods_used.count('Keyword'),
                    'success_rate': sum(1 for log in routing_log if log.get('success')) / len(
                        routing_log) * 100 if routing_log else 0
                }

                analysis['recent_decisions'] = routing_log[-10:]  # Last 10 decisions

            elif hasattr(application_state['router_agent'], 'routing_decisions'):
                # Keyword router
                decisions = application_state['router_agent'].routing_decisions
                analysis['total_queries'] = len(decisions)

                # Statistics
                engines_used = [d.get('decision') for d in decisions if d.get('decision')]
                from collections import Counter
                engine_counts = Counter(engines_used)

                analysis['routing_statistics'] = {
                    'engine_usage': dict(engine_counts),
                    'average_score': sum(d.get('winning_score', 0) for d in decisions) / len(
                        decisions) if decisions else 0,
                    'zero_score_queries': sum(1 for d in decisions if d.get('winning_score', 0) == 0)
                }

                analysis['recent_decisions'] = decisions[-10:]  # Last 10 decisions

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error getting routing analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/inspect', methods=['GET'])
def inspect_indexes():
    """Inspect vector indexes and provide analysis"""
    try:
        inspection_data = {
            'timestamp': datetime.now().isoformat(),
            'indexes': {},
            'summary': {
                'total_indexes': len(application_state['indexes']),
                'total_files': len(application_state['uploaded_files']),
                'processed_files': len(
                    [f for f in application_state['uploaded_files'].values() if f.get('processed', False)])
            }
        }

        # Inspect each index
        for file_id, index in application_state['indexes'].items():
            file_info = application_state['uploaded_files'][file_id]

            index_analysis = {
                'file_info': {
                    'original_name': file_info['original_name'],
                    'size': file_info['size'],
                    'uploaded_at': file_info['uploaded_at']
                },
                'nodes_count': file_info.get('nodes_count', 0),
                'chunk_size': file_info.get('chunk_size', 'Unknown'),
                'vector_store_type': type(index.vector_store).__name__
            }

            # Try to get embedding information
            try:
                vector_store = index.vector_store
                if hasattr(vector_store, '_data') and hasattr(vector_store._data, 'embedding_dict'):
                    embeddings_dict = vector_store._data.embedding_dict
                    if embeddings_dict:
                        sample_embedding = next(iter(embeddings_dict.values()))
                        index_analysis['embedding_dimension'] = len(sample_embedding)
                        index_analysis['stored_vectors'] = len(embeddings_dict)
            except Exception as e:
                logger.warning(f"Could not analyze embeddings for {file_info['original_name']}: {e}")

            inspection_data['indexes'][file_id] = index_analysis

        return jsonify(inspection_data)

    except Exception as e:
        logger.error(f"Error inspecting indexes: {e}")
        return jsonify({'error': str(e)}), 500


# Add this new endpoint to your app.py file

@app.route('/api/indexes', methods=['GET'])
def list_indexes():
    """List all created vector indexes with basic information"""
    try:
        indexes_list = []

        for file_id, index in application_state['indexes'].items():
            file_info = application_state['uploaded_files'].get(file_id)
            if file_info:
                index_info = {
                    'file_id': file_id,
                    'file_name': file_info['original_name'],
                    'file_size': file_info['size'],
                    'nodes_count': file_info.get('nodes_count', 0),
                    'chunk_size': file_info.get('chunk_size', 'Unknown'),
                    'created_at': file_info['uploaded_at'],
                    'processed': file_info.get('processed', False),
                    'vector_store_type': type(index.vector_store).__name__ if index else 'Unknown'
                }

                # Try to get query engine status
                index_info['has_query_engine'] = file_id in application_state['query_engines']

                indexes_list.append(index_info)

        # Sort by creation time (most recent first)
        indexes_list.sort(key=lambda x: x['created_at'], reverse=True)

        return jsonify({
            'success': True,
            'total_indexes': len(indexes_list),
            'indexes': indexes_list,
            'summary': {
                'total_files_indexed': len(indexes_list),
                'total_nodes': sum(idx.get('nodes_count', 0) for idx in indexes_list),
                'total_size_bytes': sum(idx.get('file_size', 0) for idx in indexes_list)
            }
        })

    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Also add this endpoint to get a specific index details
@app.route('/api/indexes/<file_id>', methods=['GET'])
def get_index_details(file_id):
    """Get detailed information about a specific index"""
    try:
        if file_id not in application_state['indexes']:
            return jsonify({
                'success': False,
                'error': 'Index not found'
            }), 404

        index = application_state['indexes'][file_id]
        file_info = application_state['uploaded_files'][file_id]

        index_details = {
            'file_id': file_id,
            'file_info': {
                'original_name': file_info['original_name'],
                'filename': file_info['filename'],
                'size': file_info['size'],
                'uploaded_at': file_info['uploaded_at'],
                'processed': file_info.get('processed', False)
            },
            'index_info': {
                'nodes_count': file_info.get('nodes_count', 0),
                'chunk_size': file_info.get('chunk_size', 'Unknown'),
                'vector_store_type': type(index.vector_store).__name__,
                'has_query_engine': file_id in application_state['query_engines']
            }
        }

        # Try to get embedding information
        try:
            vector_store = index.vector_store
            if hasattr(vector_store, '_data') and hasattr(vector_store._data, 'embedding_dict'):
                embeddings_dict = vector_store._data.embedding_dict
                if embeddings_dict:
                    sample_embedding = next(iter(embeddings_dict.values()))
                    index_details['index_info']['embedding_dimension'] = len(sample_embedding)
                    index_details['index_info']['stored_vectors'] = len(embeddings_dict)
        except Exception as e:
            logger.warning(f"Could not analyze embeddings for {file_info['original_name']}: {e}")

        return jsonify({
            'success': True,
            'index': index_details
        })

    except Exception as e:
        logger.error(f"Error getting index details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-data', methods=['GET'])
def export_data():
    """Export application data for download"""
    try:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'configuration': application_state['configuration'],
            'files': {file_id: {
                'original_name': info['original_name'],
                'size': info['size'],
                'uploaded_at': info['uploaded_at'],
                'processed': info.get('processed', False),
                'nodes_count': info.get('nodes_count', 0)
            } for file_id, info in application_state['uploaded_files'].items()},
            'test_results': application_state['test_results'][-50:],  # Last 50 results
            'summary': {
                'total_files': len(application_state['uploaded_files']),
                'total_indexes': len(application_state['indexes']),
                'total_queries': len(application_state['test_results']),
                'router_configured': application_state['router_agent'] is not None
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(export_data, f, indent=2)
            temp_path = f.name

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'agent_router_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get comprehensive application status"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'configuration': application_state['configuration'],
            'system_status': {
                'llm_configured': application_state['llm'] is not None,
                'embedding_configured': application_state['embed_model'] is not None,
                'router_configured': application_state['router_agent'] is not None,
                'files_uploaded': len(application_state['uploaded_files']),
                'indexes_created': len(application_state['indexes']),
                'queries_executed': len(application_state['test_results'])
            },
            'router_info': {},
            'recent_activity': application_state['test_results'][-10:] if application_state['test_results'] else []
        }

        # Add router-specific information
        if application_state['router_agent']:
            router_type = type(application_state['router_agent']).__name__
            status['router_info']['type'] = router_type

            if hasattr(application_state['router_agent'], 'llm_failures'):
                status['router_info']['llm_failures'] = application_state['router_agent'].llm_failures

            if hasattr(application_state['router_agent'], 'query_count'):
                status['router_info']['query_count'] = application_state['router_agent'].query_count

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500


# Frontend HTML Template
# Replace the HTML_TEMPLATE variable with this corrected version:

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Router Interface</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .upload-area {
            border: 2px dashed #cbd5e0;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #4299e1;
        }
        .upload-area.dragover {
            border-color: #3182ce;
            background-color: #ebf8ff;
        }
        .log-entry {
            font-family: monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
        }
        .spinner {
            border: 2px solid #f3f3f3;
            border-top: 2px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;

        // Main App Component
        function App() {
            const [activeTab, setActiveTab] = useState('configuration');
            const [status, setStatus] = useState({});
            const [loading, setLoading] = useState(false);

            useEffect(() => {
                loadStatus();
                const interval = setInterval(loadStatus, 10000); // Update every 10 seconds
                return () => clearInterval(interval);
            }, []);

            const loadStatus = async () => {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    setStatus(data);
                } catch (error) {
                    console.error('Error loading status:', error);
                }
            };

            const tabs = [
                { id: 'configuration', label: 'Configuration', icon: '⚙️' },
                { id: 'files', label: 'File Management', icon: '📁' },
                { id: 'indexes', label: 'Vector Indexes', icon: '🔍' },
                { id: 'router', label: 'Router Setup', icon: '🚦' },
                { id: 'testing', label: 'Testing', icon: '🧪' },
                { id: 'monitoring', label: 'Monitoring', icon: '📊' }
            ];

            return (
                <div className="min-h-screen bg-gray-50">
                    {/* Header */}
                    <header className="bg-white shadow-sm border-b">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex justify-between h-auto py-4">
                                <div className="flex flex-col justify-center">
                                    <h1 className="text-xl font-semibold text-gray-900">
                                        🤖 Intelligent Agent Router Interface (Agentic AI)
                                    </h1>
                                    <p>
                                        Developed by <a href="https://www.linkedin.com/in/jahidul-arafat-presidential-fellow-phd-candidate-791a7490/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">Jahidul Arafat, ex-Oracle, PhD Candidate (NSF k8s Project Intern), Auburn University (R1 Carnegie Research University), USA</a>
                                    </p>
                                    <p>Last updated: {status.timestamp}</p>
                                </div>
                                <div className="flex items-center space-x-4">
                                    <StatusIndicator status={status} />
                                </div>
                            </div>
                        </div>
                    </header>

                    {/* Navigation Tabs */}
                    <nav className="bg-white shadow-sm">
                        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                            <div className="flex space-x-8">
                                {tabs.map(tab => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`py-4 px-1 border-b-2 font-medium text-sm ${
                                            activeTab === tab.id
                                                ? 'border-blue-500 text-blue-600'
                                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
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
                        {activeTab === 'configuration' && <ConfigurationTab onUpdate={loadStatus} />}
                        {activeTab === 'files' && <FileManagementTab onUpdate={loadStatus} />}
                        {activeTab === 'indexes' && <IndexesTab onUpdate={loadStatus} />}
                        {activeTab === 'router' && <RouterTab onUpdate={loadStatus} />}
                        {activeTab === 'testing' && <TestingTab onUpdate={loadStatus} />}
                        {activeTab === 'monitoring' && <MonitoringTab status={status} />}
                    </main>
                </div>
            );
        }

        // Status Indicator Component
        function StatusIndicator({ status }) {
            const systemStatus = status.system_status || {};
            
            const getStatusColor = () => {
                if (systemStatus.router_configured) return 'bg-green-500';
                if (systemStatus.llm_configured && systemStatus.embedding_configured) return 'bg-yellow-500';
                return 'bg-red-500';
            };

            const getStatusText = () => {
                if (systemStatus.router_configured) return 'Router Ready';
                if (systemStatus.llm_configured && systemStatus.embedding_configured) return 'Models Ready';
                return 'Not Configured';
            };

            return (
                <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
                    <span className="text-sm text-gray-600">{getStatusText()}</span>
                </div>
            );
        }

        // Configuration Tab Component
        function ConfigurationTab({ onUpdate }) {
            const [config, setConfig] = useState({
                llm_model: '',
                embedding_model: '',
                huggingface_token: '',
                lm_studio_url: 'http://127.0.0.1:1234/v1'
            });
            const [models, setModels] = useState({ llm_models: {}, embedding_models: {} });
            const [loading, setLoading] = useState(false);
            const [message, setMessage] = useState('');

            useEffect(() => {
                loadModels();
                loadCurrentConfig();
            }, []);

            const loadModels = async () => {
                try {
                    const response = await fetch('/api/models');
                    const data = await response.json();
                    setModels(data);
                } catch (error) {
                    console.error('Error loading models:', error);
                }
            };

            const loadCurrentConfig = async () => {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    if (data.configuration) {
                        setConfig(prev => ({ ...prev, ...data.configuration }));
                    }
                } catch (error) {
                    console.error('Error loading config:', error);
                }
            };

            const handleSubmit = async (e) => {
                e.preventDefault();
                setLoading(true);
                setMessage('');

                try {
                    const response = await fetch('/api/configure', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        setMessage('✅ Configuration updated successfully!');
                        onUpdate();
                    } else {
                        setMessage(`❌ Error: ${data.error}`);
                    }
                } catch (error) {
                    setMessage(`❌ Error: ${error.message}`);
                } finally {
                    setLoading(false);
                }
            };

            return (
                <div className="bg-white shadow rounded-lg p-6">
                    <h2 className="text-lg font-medium text-gray-900 mb-6">Model Configuration</h2>
                    
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                            {/* LLM Model Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    LLM Model
                                </label>
                                <select
                                    value={config.llm_model}
                                    onChange={(e) => setConfig(prev => ({ ...prev, llm_model: e.target.value }))}
                                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    required
                                >
                                    <option value="">Select LLM Model</option>
                                    {Object.entries(models.llm_models).map(([key, value]) => (
                                        <option key={key} value={key}>{value}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Embedding Model Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Embedding Model
                                </label>
                                <select
                                    value={config.embedding_model}
                                    onChange={(e) => setConfig(prev => ({ ...prev, embedding_model: e.target.value }))}
                                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    required
                                >
                                    <option value="">Select Embedding Model</option>
                                    {Object.entries(models.embedding_models).map(([key, value]) => (
                                        <option key={key} value={key}>{value}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* LM Studio URL */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                LM Studio URL
                            </label>
                            <input
                                type="url"
                                value={config.lm_studio_url}
                                onChange={(e) => setConfig(prev => ({ ...prev, lm_studio_url: e.target.value }))}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="http://127.0.0.1:1234/v1"
                            />
                        </div>

                        {/* HuggingFace Token */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                HuggingFace Token (Optional)
                            </label>
                            <input
                                type="password"
                                value={config.huggingface_token}
                                onChange={(e) => setConfig(prev => ({ ...prev, huggingface_token: e.target.value }))}
                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="hf_..."
                            />
                            <p className="mt-1 text-sm text-gray-500">
                                Required for some HuggingFace embedding models
                            </p>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                        >
                            {loading ? (
                                <>
                                    <span className="spinner mr-2"></span>
                                    Configuring...
                                </>
                            ) : (
                                'Configure Models'
                            )}
                        </button>
                    </form>

                    {message && (
                        <div className="mt-4 p-3 rounded-md bg-gray-50 border">
                            <p className="text-sm">{message}</p>
                        </div>
                    )}
                </div>
            );
        }

        // File Management Tab Component  
        function FileManagementTab({ onUpdate }) {
            const [files, setFiles] = useState([]);
            const [uploading, setUploading] = useState(false);
            const [message, setMessage] = useState('');
            const fileInputRef = useRef(null);

            useEffect(() => {
                loadFiles();
            }, []);

            const loadFiles = async () => {
                try {
                    const response = await fetch('/api/files');
                    const data = await response.json();
                    setFiles(data.files || []);
                } catch (error) {
                    console.error('Error loading files:', error);
                }
            };

            const handleFileUpload = async (fileList) => {
                setUploading(true);
                setMessage('');

                const formData = new FormData();
                Array.from(fileList).forEach(file => {
                    formData.append('files', file);
                });

                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        setMessage(`✅ ${data.message}`);
                        loadFiles();
                        onUpdate();
                    } else {
                        setMessage(`❌ Error: ${data.error}`);
                    }
                } catch (error) {
                    setMessage(`❌ Error: ${error.message}`);
                } finally {
                    setUploading(false);
                }
            };

            const handleDelete = async (fileId) => {
                if (!confirm('Are you sure you want to delete this file?')) return;

                try {
                    const response = await fetch(`/api/files/${fileId}`, {
                        method: 'DELETE'
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        setMessage('✅ File deleted successfully');
                        loadFiles();
                        onUpdate();
                    } else {
                        setMessage(`❌ Error: ${data.error}`);
                    }
                } catch (error) {
                    setMessage(`❌ Error: ${error.message}`);
                }
            };

            const handleDrop = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.currentTarget.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileUpload(files);
                }
            };

            const handleDragOver = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.currentTarget.classList.add('dragover');
            };

            const handleDragLeave = (e) => {
                e.preventDefault();
                e.stopPropagation();
                e.currentTarget.classList.remove('dragover');
            };

            return (
                <div className="space-y-6">
                    {/* Upload Area */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">Upload Documents</h2>
                        
                        <div
                            className="upload-area"
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <div className="text-center">
                                <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                                <div className="mt-4">
                                    <p className="text-sm text-gray-600">
                                        {uploading ? (
                                            <>
                                                <span className="spinner mr-2"></span>
                                                Uploading files...
                                            </>
                                        ) : (
                                            <>
                                                <span className="font-medium text-blue-600 hover:text-blue-500">
                                                    Click to upload
                                                </span> or drag and drop
                                            </>
                                        )}
                                    </p>
                                    <p className="text-xs text-gray-500 mt-1">
                                        PDF, TXT, DOCX files up to 100MB
                                    </p>
                                </div>
                            </div>
                        </div>

                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            accept=".pdf,.txt,.docx,.doc"
                            onChange={(e) => handleFileUpload(e.target.files)}
                            className="hidden"
                        />

                        {message && (
                            <div className="mt-4 p-3 rounded-md bg-gray-50 border">
                                <p className="text-sm">{message}</p>
                            </div>
                        )}
                    </div>

                    {/* Files List */}
                    <div className="bg-white shadow rounded-lg">
                        <div className="px-6 py-4 border-b border-gray-200">
                            <h3 className="text-lg font-medium text-gray-900">
                                Uploaded Files ({files.length})
                            </h3>
                        </div>
                        
                        {files.length === 0 ? (
                            <div className="p-6 text-center text-gray-500">
                                No files uploaded yet
                            </div>
                        ) : (
                            <div className="divide-y divide-gray-200">
                                {files.map(file => (
                                    <div key={file.id} className="p-6 flex items-center justify-between">
                                        <div className="flex items-center space-x-4">
                                            <div className="flex-shrink-0">
                                                <span className="text-2xl">📄</span>
                                            </div>
                                            <div>
                                                <h4 className="text-sm font-medium text-gray-900">
                                                    {file.original_name}
                                                </h4>
                                                <p className="text-sm text-gray-500">
                                                    {(file.size / 1024).toFixed(1)} KB • 
                                                    Uploaded {new Date(file.uploaded_at).toLocaleString()}
                                                    {file.processed && (
                                                        <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                                            Processed ({file.nodes_count} chunks)
                                                        </span>
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleDelete(file.id)}
                                            className="text-red-600 hover:text-red-900 text-sm"
                                        >
                                            Delete
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        // Vector Indexes Tab Component
        function IndexesTab({ onUpdate }) {
            const [files, setFiles] = useState([]);
            const [chunkSize, setChunkSize] = useState(1024);
            const [selectedFiles, setSelectedFiles] = useState([]);
            const [creating, setCreating] = useState(false);
            const [message, setMessage] = useState('');
            const [inspection, setInspection] = useState(null);

            useEffect(() => {
                loadFiles();
                loadInspection();
            }, []);

            const loadFiles = async () => {
                try {
                    const response = await fetch('/api/files');
                    const data = await response.json();
                    setFiles(data.files || []);
                    // Select all unprocessed files by default
                    const unprocessed = (data.files || []).filter(f => !f.processed).map(f => f.id);
                    setSelectedFiles(unprocessed);
                } catch (error) {
                    console.error('Error loading files:', error);
                }
            };

            const loadInspection = async () => {
                try {
                    const response = await fetch('/api/inspect');
                    const data = await response.json();
                    setInspection(data);
                } catch (error) {
                    console.error('Error loading inspection:', error);
                }
            };

            const createIndexes = async () => {
                setCreating(true);
                setMessage('');

                try {
                    const response = await fetch('/api/create-indexes', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            file_ids: selectedFiles,
                            chunk_size: chunkSize
                        })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        setMessage(`✅ ${data.message}`);
                        loadFiles();
                        loadInspection();
                        onUpdate();
                    } else {
                        setMessage(`❌ Error: ${data.error}`);
                    }
                } catch (error) {
                    setMessage(`❌ Error: ${error.message}`);
                } finally {
                    setCreating(false);
                }
            };

            const toggleFileSelection = (fileId) => {
                setSelectedFiles(prev => 
                    prev.includes(fileId) 
                        ? prev.filter(id => id !== fileId)
                        : [...prev, fileId]
                );
            };

            return (
                <div className="space-y-6">
                    {/* Index Creation */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">Create Vector Indexes</h2>
                        
                        <div className="space-y-4">
                            {/* Chunk Size Setting */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Chunk Size (characters)
                                </label>
                                <input
                                    type="number"
                                    value={chunkSize}
                                    onChange={(e) => setChunkSize(parseInt(e.target.value))}
                                    min="256"
                                    max="4096"
                                    step="256"
                                    className="w-32 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                                <p className="mt-1 text-sm text-gray-500">
                                    Recommended: 1024 for balanced performance
                                </p>
                            </div>

                            {/* File Selection */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Select Files to Process
                                </label>
                                <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-200 rounded-md p-3">
                                    {files.map(file => (
                                        <label key={file.id} className="flex items-center space-x-3">
                                            <input
                                                type="checkbox"
                                                checked={selectedFiles.includes(file.id)}
                                                onChange={() => toggleFileSelection(file.id)}
                                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                            />
                                            <span className="text-sm text-gray-900">
                                                {file.original_name}
                                                {file.processed && (
                                                    <span className="ml-2 text-green-600 text-xs">(Processed)</span>
                                                )}
                                            </span>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={createIndexes}
                                disabled={creating || selectedFiles.length === 0}
                                className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                            >
                                {creating ? (
                                    <>
                                        <span className="spinner mr-2"></span>
                                        Creating Indexes...
                                    </>
                                ) : (
                                    `Create Indexes (${selectedFiles.length} files)`
                                )}
                            </button>
                        </div>

                        {message && (
                            <div className="mt-4 p-3 rounded-md bg-gray-50 border">
                                <p className="text-sm">{message}</p>
                            </div>
                        )}
                    </div>

                    {/* Index Inspection */}
                    {inspection && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Index Analysis</h3>
                            
                            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 mb-6">
                                <div className="bg-blue-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-blue-600">
                                        {inspection.summary.total_indexes}
                                    </div>
                                    <div className="text-sm text-blue-800">Total Indexes</div>
                                </div>
                                <div className="bg-green-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-green-600">
                                        {inspection.summary.processed_files}
                                    </div>
                                    <div className="text-sm text-green-800">Processed Files</div>
                                </div>
                                <div className="bg-purple-50 p-4 rounded-lg">
                                    <div className="text-2xl font-bold text-purple-600">
                                        {Object.values(inspection.indexes).reduce((sum, idx) => sum + (idx.nodes_count || 0), 0)}
                                    </div>
                                    <div className="text-sm text-purple-800">Total Chunks</div>
                                </div>
                            </div>

                            {Object.entries(inspection.indexes).map(([fileId, index]) => (
                                <div key={fileId} className="border border-gray-200 rounded-lg p-4">
                                    <h5 className="font-medium text-gray-900 mb-2">
                                        {index.file_info.original_name}
                                    </h5>
                                    <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                                        <div>
                                            <span className="font-medium">Chunks:</span> {index.nodes_count}
                                        </div>
                                        <div>
                                            <span className="font-medium">Chunk Size:</span> {index.chunk_size}
                                        </div>
                                        <div>
                                            <span className="font-medium">Vector Store:</span> {index.vector_store_type}
                                        </div>
                                        {index.embedding_dimension && (
                                            <div>
                                                <span className="font-medium">Embedding Dim:</span> {index.embedding_dimension}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            );
        }

        // Router Tab Component
        function RouterTab({ onUpdate }) {
            const [routerType, setRouterType] = useState('hybrid');
            const [creating, setCreating] = useState(false);
            const [message, setMessage] = useState('');
            const [routerInfo, setRouterInfo] = useState(null);

            useEffect(() => {
                loadRouterInfo();
            }, []);

            const loadRouterInfo = async () => {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    setRouterInfo(data.router_info);
                } catch (error) {
                    console.error('Error loading router info:', error);
                }
            };

            const createRouter = async () => {
                setCreating(true);
                setMessage('');

                try {
                    const response = await fetch('/api/create-router', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ router_type: routerType })
                    });

                    const data = await response.json();
                    
                    if (data.success) {
                        setMessage(`✅ ${data.message}`);
                        loadRouterInfo();
                        onUpdate();
                    } else {
                        setMessage(`❌ Error: ${data.error}`);
                    }
                } catch (error) {
                    setMessage(`❌ Error: ${error.message}`);
                } finally {
                    setCreating(false);
                }
            };

            return (
                <div className="space-y-6">
                    {/* Router Creation */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">Router Configuration</h2>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Router Type
                                </label>
                                <div className="space-y-2">
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            value="hybrid"
                                            checked={routerType === 'hybrid'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                        />
                                        <span className="ml-2 text-sm text-gray-900">
                                            <strong>Hybrid Router</strong> - LLM with keyword fallback (Recommended)
                                        </span>
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            value="llm"
                                            checked={routerType === 'llm'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                        />
                                        <span className="ml-2 text-sm text-gray-900">
                                            <strong>LLM Router</strong> - Uses LLM for intelligent routing
                                        </span>
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            value="keyword"
                                            checked={routerType === 'keyword'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                        />
                                        <span className="ml-2 text-sm text-gray-900">
                                            <strong>Keyword Router</strong> - Simple keyword-based routing
                                        </span>
                                    </label>
                                </div>
                            </div>

                            <button
                                onClick={createRouter}
                                disabled={creating}
                                className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                            >
                                {creating ? (
                                    <>
                                        <span className="spinner mr-2"></span>
                                        Creating Router...
                                    </>
                                ) : (
                                    'Create Router'
                                )}
                            </button>
                        </div>

                        {message && (
                            <div className="mt-4 p-3 rounded-md bg-gray-50 border">
                                <p className="text-sm">{message}</p>
                            </div>
                        )}
                    </div>

                    {/* Router Status */}
                    {routerInfo && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Router Status</h3>
                            
                            <div className="space-y-2 text-sm">
                                <div>
                                    <span className="font-medium">Type:</span> {routerInfo.type}
                                </div>
                                {routerInfo.llm_failures !== undefined && (
                                    <div>
                                        <span className="font-medium">LLM Failures:</span> {routerInfo.llm_failures}/3
                                    </div>
                                )}
                                {routerInfo.query_count !== undefined && (
                                    <div>
                                        <span className="font-medium">Queries Processed:</span> {routerInfo.query_count}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Quick Test */}
                    <QuickQueryTest />
                </div>
            );
        }

        // Quick Query Test Component
        function QuickQueryTest() {
            const [query, setQuery] = useState('');
            const [loading, setLoading] = useState(false);
            const [result, setResult] = useState(null);

            const predefinedQueries = [
                // Explicit mentions
                "What colors are available for AeroFlow?",
                "Tell me about EcoSprint's battery specifications",
                "How do I maintain my AeroFlow vehicle?",
                "What is EcoSprint's top speed?",
                
                // Ambiguous queries
                "Which vehicle has better performance?",
                "What are the available color options?",
                "Compare the two electric vehicles",
                "Which one is more environmentally friendly?",
                
                // Contextual keywords
                "Tell me about the eco-friendly features",
                "What about aerodynamic design?",
                "How green is this vehicle?",
                "What about the flow dynamics?",
                
                // Technical specs
                "What is the battery capacity?",
                "How long does charging take?",
                "What safety features are included?",
                "What is the warranty coverage?"
            ];

            const executeQuery = async (queryText) => {
                setLoading(true);
                setResult(null);

                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queryText })
                    });

                    const data = await response.json();
                    setResult(data);
                } catch (error) {
                    setResult({ success: false, error: error.message });
                } finally {
                    setLoading(false);
                }
            };

            return (
                <div className="bg-white shadow rounded-lg p-6">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Quick Query Test</h3>
                    
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Enter Query
                            </label>
                            <div className="flex space-x-2">
                                <input
                                    type="text"
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="Ask a question..."
                                    className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    onKeyPress={(e) => e.key === 'Enter' && executeQuery(query)}
                                />
                                <button
                                    onClick={() => executeQuery(query)}
                                    disabled={loading || !query.trim()}
                                    className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                                >
                                    {loading ? <span className="spinner"></span> : 'Ask'}
                                </button>
                            </div>
                        </div>

                        <div>
                            <p className="text-sm text-gray-700 mb-2">Or try a predefined query:</p>
                            <div className="flex flex-wrap gap-2">
                                {predefinedQueries.map((predefinedQuery, index) => (
                                    <button
                                        key={index}
                                        onClick={() => executeQuery(predefinedQuery)}
                                        disabled={loading}
                                        className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
                                    >
                                        {predefinedQuery}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {result && (
                            <div className="border-t pt-4">
                                {result.success ? (
                                    <div className="space-y-2">
                                        <div className="text-sm text-gray-600">
                                            Response time: {result.response_time?.toFixed(2)}s
                                        </div>
                                        <div className="bg-gray-50 p-3 rounded border">
                                            <p className="text-sm">{result.response}</p>
                                        </div>
                                        {result.routing_info && (
                                            <details className="text-xs text-gray-500">
                                                <summary className="cursor-pointer">Routing Details</summary>
                                                <pre className="mt-2 p-2 bg-gray-100 rounded">
                                                    {JSON.stringify(result.routing_info, null, 2)}
                                                </pre>
                                            </details>
                                        )}
                                    </div>
                                ) : (
                                    <div className="text-red-600 text-sm">
                                        Error: {result.error}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        // Testing Tab Component
        function TestingTab({ onUpdate }) {
            const [testType, setTestType] = useState('comprehensive');
            const [running, setRunning] = useState(false);
            const [results, setResults] = useState(null);
            const [currentTest, setCurrentTest] = useState(null);
            const [testProgress, setTestProgress] = useState([]);
            const [progressStats, setProgressStats] = useState({ completed: 0, total: 0 });

            // Define test cases to show progress
            // Enhanced test cases matching the notebook
            const getTestCases = (type) => {
                if (type === 'basic') {
                    return [
                        { query: "What are the key features?", category: "Generic" },
                        { query: "Tell me about specifications", category: "Generic" },
                        { query: "What is the warranty coverage?", category: "Generic" }
                    ];
                } else {
                    return [
                        // Explicit Vehicle Mentions
                        { query: "What colors are available for AeroFlow?", category: "Explicit - AeroFlow" },
                        { query: "Tell me about EcoSprint's battery specifications", category: "Explicit - EcoSprint" },
                        { query: "How do I maintain my AeroFlow vehicle?", category: "Explicit - AeroFlow" },
                        { query: "What is EcoSprint's top speed?", category: "Explicit - EcoSprint" },
                        
                        // Ambiguous/Comparison Queries
                        { query: "Which vehicle has better performance?", category: "Ambiguous - Comparison" },
                        { query: "What are the available color options?", category: "Ambiguous - Generic" },
                        { query: "Compare the two electric vehicles", category: "Ambiguous - Comparison" },
                        { query: "Which one is more environmentally friendly?", category: "Ambiguous - Environmental" },
                        
                        // Contextual Keywords
                        { query: "Tell me about the eco-friendly features", category: "Contextual - Eco" },
                        { query: "What about aerodynamic design?", category: "Contextual - Aero" },
                        { query: "How green is this vehicle?", category: "Contextual - Green" },
                        { query: "What about the flow dynamics?", category: "Contextual - Flow" },
                        
                        // Technical Specifications
                        { query: "What is the battery capacity?", category: "Technical - Battery" },
                        { query: "How long does charging take?", category: "Technical - Charging" },
                        { query: "What safety features are included?", category: "Technical - Safety" },
                        { query: "What is the warranty coverage?", category: "Technical - Warranty" },
                        
                        // Additional edge cases
                        { query: "Compare AeroFlow and EcoSprint features", category: "Explicit Comparison" },
                        { query: "Which has better range?", category: "Comparison - Range" },
                        { query: "What are the maintenance requirements?", category: "Generic - Maintenance" },
                        { query: "Tell me about pricing", category: "Generic - Pricing" },
                        
                        // Contextual variations
                        { query: "What about the family-friendly features?", category: "Contextual - Family" },
                        { query: "How efficient is the motor?", category: "Technical - Efficiency" },
                        { query: "What charging options are available?", category: "Technical - Charging Options" },
                        { query: "Tell me about the interior design", category: "Technical - Design" }
                    ];
                }
            };

            const runTestsWithProgress = async () => {
                setRunning(true);
                setResults(null);
                setTestProgress([]);
                setCurrentTest(null);
                
                const testCases = getTestCases(testType);
                setProgressStats({ completed: 0, total: testCases.length });

                // Simulate running individual tests with progress updates
                const progressResults = [];
                
                for (let i = 0; i < testCases.length; i++) {
                    const testCase = testCases[i];
                    
                    // Update current test
                    setCurrentTest({
                        index: i + 1,
                        total: testCases.length,
                        query: testCase.query,
                        category: testCase.category,
                        status: 'running'
                    });

                    try {
                        // Execute individual test
                        const startTime = Date.now();
                        const response = await fetch('/api/query', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: testCase.query })
                        });

                        const data = await response.json();
                        const endTime = Date.now();
                        const responseTime = (endTime - startTime) / 1000;

                        const testResult = {
                            test_id: i + 1,
                            query: testCase.query,
                            category: testCase.category,
                            success: data.success,
                            response: data.response || '',
                            response_time: responseTime,
                            response_length: data.response ? data.response.length : 0,
                            timestamp: new Date().toISOString(),
                            error: data.error || null
                        };

                        progressResults.push(testResult);

                        // Update progress
                        setTestProgress(prev => [...prev, testResult]);
                        setProgressStats({ completed: i + 1, total: testCases.length });
                        
                        // Update current test status
                        setCurrentTest(prev => ({ ...prev, status: 'completed', success: data.success }));

                        // Small delay to show progress
                        await new Promise(resolve => setTimeout(resolve, 500));

                    } catch (error) {
                        const testResult = {
                            test_id: i + 1,
                            query: testCase.query,
                            category: testCase.category,
                            success: false,
                            error: error.message,
                            timestamp: new Date().toISOString()
                        };

                        progressResults.push(testResult);
                        setTestProgress(prev => [...prev, testResult]);
                        setProgressStats({ completed: i + 1, total: testCases.length });
                        setCurrentTest(prev => ({ ...prev, status: 'failed', success: false }));

                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                }

                // Calculate final results
                const successful = progressResults.filter(r => r.success);
                const summary = {
                    total_tests: progressResults.length,
                    successful_tests: successful.length,
                    success_rate: (successful.length / progressResults.length) * 100,
                    average_response_time: successful.length > 0 
                        ? successful.reduce((sum, r) => sum + (r.response_time || 0), 0) / successful.length 
                        : 0,
                    average_response_length: successful.length > 0
                        ? successful.reduce((sum, r) => sum + (r.response_length || 0), 0) / successful.length
                        : 0
                };

                const finalResults = {
                    test_type: testType,
                    timestamp: new Date().toISOString(),
                    summary: summary,
                    results: progressResults
                };

                setResults(finalResults);
                setCurrentTest(null);
                setRunning(false);
                onUpdate();
            };

            return (
                <div className="space-y-6">
                    {/* Test Configuration */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">Router Testing</h2>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Test Type
                                </label>
                                <div className="space-y-2">
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            value="basic"
                                            checked={testType === 'basic'}
                                            onChange={(e) => setTestType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                            disabled={running}
                                        />
                                        <span className="ml-2 text-sm text-gray-900">
                                            <strong>Basic Test</strong> - Simple functionality check (3 tests)
                                        </span>
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            value="comprehensive"
                                            checked={testType === 'comprehensive'}
                                            onChange={(e) => setTestType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                            disabled={running}
                                        />
                                        <span className="ml-2 text-sm text-gray-900">
                                            <strong>Comprehensive Test</strong> - Full routing intelligence test (24 tests)
                                        </span>
                                    </label>
                                </div>
                            </div>

                            <button
                                onClick={runTestsWithProgress}
                                disabled={running}
                                className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                            >
                                {running ? (
                                    <>
                                        <span className="spinner mr-2"></span>
                                        Running Tests...
                                    </>
                                ) : (
                                    `Run ${testType} Tests`
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Real-time Test Progress */}
                    {running && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Test Progress</h3>
                            
                            {/* Progress Bar */}
                            <div className="mb-4">
                                <div className="flex justify-between text-sm text-gray-600 mb-1">
                                    <span>Progress</span>
                                    <span>{progressStats.completed}/{progressStats.total} tests completed</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div 
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${(progressStats.completed / progressStats.total) * 100}%` }}
                                    ></div>
                                </div>
                            </div>

                            {/* Current Test */}
                            {currentTest && (
                                <div className="mb-4 p-4 border border-blue-200 rounded-lg bg-blue-50">
                                    <div className="flex items-center justify-between mb-2">
                                        <h4 className="font-medium text-blue-900">
                                            Test {currentTest.index}/{currentTest.total}
                                        </h4>
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                            currentTest.status === 'running' 
                                                ? 'bg-yellow-100 text-yellow-800'
                                                : currentTest.success
                                                ? 'bg-green-100 text-green-800'
                                                : 'bg-red-100 text-red-800'
                                        }`}>
                                            {currentTest.status === 'running' && <span className="spinner mr-1"></span>}
                                            {currentTest.status === 'running' ? 'Running...' 
                                             : currentTest.status === 'completed' && currentTest.success ? '✅ Passed'
                                             : currentTest.status === 'completed' && !currentTest.success ? '❌ Failed'
                                             : '⏳ Pending'}
                                        </span>
                                    </div>
                                    <div className="text-sm text-blue-700">
                                        <div className="font-medium mb-1">Category: {currentTest.category}</div>
                                        <div>Query: "{currentTest.query}"</div>
                                    </div>
                                </div>
                            )}

                            {/* Completed Tests Log */}
                            {testProgress.length > 0 && (
                                <div>
                                    <h4 className="font-medium text-gray-900 mb-3">Completed Tests</h4>
                                    <div className="space-y-2 max-h-60 overflow-y-auto">
                                        {testProgress.map((test, index) => (
                                            <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded">
                                                <div className="flex-1">
                                                    <div className="flex items-center space-x-2">
                                                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                                            test.success 
                                                                ? 'bg-green-100 text-green-800' 
                                                                : 'bg-red-100 text-red-800'
                                                        }`}>
                                                            {test.success ? '✅' : '❌'}
                                                        </span>
                                                        <span className="text-sm text-gray-500">{test.category}</span>
                                                    </div>
                                                    <div className="text-sm text-gray-900 mt-1 truncate">
                                                        {test.query}
                                                    </div>
                                                </div>
                                                <div className="text-xs text-gray-500 ml-4">
                                                    {test.response_time?.toFixed(2)}s
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Test Results */}
                    {results && !running && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Test Results</h3>
                            
                            {results.error ? (
                                <div className="text-red-600">
                                    Error: {results.error}
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    {/* Summary */}
                                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
                                        <div className="bg-blue-50 p-4 rounded-lg">
                                            <div className="text-2xl font-bold text-blue-600">
                                                {results.summary.total_tests}
                                            </div>
                                            <div className="text-sm text-blue-800">Total Tests</div>
                                        </div>
                                        <div className="bg-green-50 p-4 rounded-lg">
                                            <div className="text-2xl font-bold text-green-600">
                                                {results.summary.successful_tests}
                                            </div>
                                            <div className="text-sm text-green-800">Successful</div>
                                        </div>
                                        <div className="bg-yellow-50 p-4 rounded-lg">
                                            <div className="text-2xl font-bold text-yellow-600">
                                                {results.summary.success_rate.toFixed(1)}%
                                            </div>
                                            <div className="text-sm text-yellow-800">Success Rate</div>
                                        </div>
                                        <div className="bg-purple-50 p-4 rounded-lg">
                                            <div className="text-2xl font-bold text-purple-600">
                                                {results.summary.average_response_time?.toFixed(2)}s
                                            </div>
                                            <div className="text-sm text-purple-800">Avg Response</div>
                                        </div>
                                    </div>

                                    {/* Individual Test Results */}
                                    <div>
                                        <h4 className="font-medium text-gray-900 mb-3">Individual Test Results</h4>
                                        <div className="space-y-3">
                                            {results.results.map((result, index) => (
                                                <div key={index} className="border border-gray-200 rounded-lg p-4">
                                                    <div className="flex items-start justify-between">
                                                        <div className="flex-1">
                                                            <div className="flex items-center space-x-2 mb-1">
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                                    result.success 
                                                                        ? 'bg-green-100 text-green-800' 
                                                                        : 'bg-red-100 text-red-800'
                                                                }`}>
                                                                    {result.success ? '✅ Pass' : '❌ Fail'}
                                                                </span>
                                                                <span className="text-sm text-gray-500">
                                                                    {result.category}
                                                                </span>
                                                                <span className="text-xs text-gray-400">
                                                                    Test #{result.test_id}
                                                                </span>
                                                            </div>
                                                            <p className="text-sm font-medium text-gray-900 mb-2">
                                                                {result.query}
                                                            </p>
                                                            {result.success ? (
                                                                <div className="text-sm text-gray-600">
                                                                    <div className="flex items-center space-x-4 mb-2">
                                                                        <span>Response: {result.response_length} chars</span>
                                                                        <span>Time: {result.response_time?.toFixed(2)}s</span>
                                                                    </div>
                                                                    <details className="mt-2">
                                                                        <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                                                                            View Response
                                                                        </summary>
                                                                        <div className="mt-2 p-3 bg-gray-50 rounded text-xs border-l-4 border-blue-400">
                                                                            {result.response}
                                                                        </div>
                                                                    </details>
                                                                </div>
                                                            ) : (
                                                                <div className="text-sm text-red-600">
                                                                    Error: {result.error}
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            );
        }

        // Monitoring Tab Component
        function MonitoringTab({ status }) {
            const [exportLoading, setExportLoading] = useState(false);

            const exportData = async () => {
                setExportLoading(true);
                try {
                    const response = await fetch('/api/export-data');
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `agent_router_export_${new Date().toISOString().slice(0, 10)}.json`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                } catch (error) {
                    console.error('Export failed:', error);
                } finally {
                    setExportLoading(false);
                }
            };

            return (
                <div className="space-y-6">
                    {/* System Overview */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-6">System Overview</h2>
                        
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                            <div className="bg-blue-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-blue-600">
                                    {status.system_status?.files_uploaded || 0}
                                </div>
                                <div className="text-sm text-blue-800">Files Uploaded</div>
                            </div>
                            <div className="bg-green-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-green-600">
                                    {status.system_status?.indexes_created || 0}
                                </div>
                                <div className="text-sm text-green-800">Indexes Created</div>
                            </div>
                            <div className="bg-purple-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-purple-600">
                                    {status.system_status?.queries_executed || 0}
                                </div>
                                <div className="text-sm text-purple-800">Queries Executed</div>
                            </div>
                            <div className="bg-yellow-50 p-4 rounded-lg">
                                <div className="text-2xl font-bold text-yellow-600">
                                    {status.system_status?.router_configured ? '✅' : '❌'}
                                </div>
                                <div className="text-sm text-yellow-800">Router Status</div>
                            </div>
                        </div>
                    </div>

                    {/* Configuration Status */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
                        
                        <div className="space-y-3 text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-600">LLM Model:</span>
                                <span className="font-medium">
                                    {status.configuration?.llm_model || 'Not configured'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">Embedding Model:</span>
                                <span className="font-medium">
                                    {status.configuration?.embedding_model || 'Not configured'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">LM Studio URL:</span>
                                <span className="font-medium">
                                    {status.configuration?.lm_studio_url || 'Default'}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-600">HuggingFace Token:</span>
                                <span className="font-medium">
                                    {status.configuration?.huggingface_token ? '✅ Set' : '❌ Not set'}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Recent Activity */}
                    {status.recent_activity && status.recent_activity.length > 0 && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
                            
                            <div className="space-y-3">
                                {status.recent_activity.slice(0, 5).map((activity, index) => (
                                    <div key={index} className="border-l-4 border-blue-400 pl-4 py-2">
                                        <div className="text-sm text-gray-900">
                                            {activity.query || 'Query executed'}
                                        </div>
                                        <div className="text-xs text-gray-500">
                                            {new Date(activity.timestamp).toLocaleString()}
                                            {activity.response_time && ` • ${activity.response_time.toFixed(2)}s`}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Export Data */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Data Export</h3>
                        
                        <p className="text-sm text-gray-600 mb-4">
                            Export application data including configuration, test results, and system information.
                        </p>
                        
                        <button
                            onClick={exportData}
                            disabled={exportLoading}
                            className="bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                        >
                            {exportLoading ? (
                                <>
                                    <span className="spinner mr-2"></span>
                                    Exporting...
                                </>
                            ) : (
                                '📥 Export Data'
                            )}
                        </button>
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
    # CHANGE FROM:
    # return render_template_string(HTML_TEMPLATE)

    # TO:
    from flask import Response
    return Response(HTML_TEMPLATE, mimetype='text/html')


# Swagger/OpenAPI documentation
# Swagger/OpenAPI documentation (continued)
@app.route('/api/docs')
def api_docs():
    """API documentation"""
    swagger_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Agent Router API",
            "version": "1.0.0",
            "description": "API for managing LLM-based document routing and querying"
        },
        "servers": [
            {"url": "http://localhost:5010", "description": "Local development server"}
        ],
        "paths": {
            "/api/health": {
                "get": {
                    "summary": "Health Check",
                    "responses": {"200": {"description": "System health status"}}
                }
            },
            "/api/models": {
                "get": {
                    "summary": "Get Available Models",
                    "responses": {"200": {"description": "List of available LLM and embedding models"}}
                }
            },
            "/api/configure": {
                "post": {
                    "summary": "Configure Models",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "llm_model": {"type": "string"},
                                        "embedding_model": {"type": "string"},
                                        "huggingface_token": {"type": "string"},
                                        "lm_studio_url": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Configuration result"}}
                }
            },
            "/api/upload": {
                "post": {
                    "summary": "Upload Files",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "binary"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Upload result"}}
                }
            },
            "/api/files": {
                "get": {
                    "summary": "List Files",
                    "responses": {"200": {"description": "List of uploaded files"}}
                }
            },
            "/api/files/{file_id}": {
                "delete": {
                    "summary": "Delete File",
                    "parameters": [
                        {
                            "name": "file_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {"200": {"description": "Deletion result"}}
                }
            },
            "/api/create-indexes": {
                "post": {
                    "summary": "Create Vector Indexes",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file_ids": {"type": "array", "items": {"type": "string"}},
                                        "chunk_size": {"type": "integer", "default": 1024}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Index creation result"}}
                }
            },
            "/api/create-router": {
                "post": {
                    "summary": "Create Router Agent",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "router_type": {
                                            "type": "string",
                                            "enum": ["llm", "keyword", "hybrid"],
                                            "default": "hybrid"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Router creation result"}}
                }
            },
            "/api/query": {
                "post": {
                    "summary": "Query Router",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Query result"}}
                }
            },
            "/api/test": {
                "post": {
                    "summary": "Run Comprehensive Tests",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "test_type": {
                                            "type": "string",
                                            "enum": ["basic", "comprehensive"],
                                            "default": "comprehensive"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Test results"}}
                }
            },
            "/api/inspect": {
                "get": {
                    "summary": "Inspect Vector Indexes",
                    "responses": {"200": {"description": "Index inspection data"}}
                }
            },
            "/api/status": {
                "get": {
                    "summary": "Get System Status",
                    "responses": {"200": {"description": "Comprehensive system status"}}
                }
            },
            "/api/export-data": {
                "get": {
                    "summary": "Export Application Data",
                    "responses": {
                        "200": {
                            "description": "JSON export file",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "success": {"type": "boolean", "default": False}
                    }
                },
                "SuccessResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "default": True},
                        "message": {"type": "string"}
                    }
                }
            }
        }
    }

    # Return Swagger UI HTML
    swagger_ui_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Router API Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css" />
        <style>
            html {{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }}
            *, *:before, *:after {{ box-sizing: inherit; }}
            body {{ margin:0; background: #fafafa; }}
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {{
                const ui = SwaggerUIBundle({{
                    url: '/api/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ],
                    layout: "StandaloneLayout"
                }});
            }};
        </script>
    </body>
    </html>
    """
    return swagger_ui_html


# Replace the openapi_spec() function in your app.py with this complete version

@app.route('/api/openapi.json')
def openapi_spec():
    """Return complete OpenAPI specification as JSON"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Agent Router API",
            "version": "1.0.0",
            "description": "Comprehensive API for managing LLM-based document routing and querying system with intelligent routing capabilities",
            "contact": {
                "name": "Agent Router Support",
                "url": "http://localhost:5010"
            }
        },
        "servers": [
            {"url": "http://localhost:5010", "description": "Local development server"}
        ],
        "tags": [
            {"name": "System", "description": "System health and status endpoints"},
            {"name": "Configuration", "description": "Model and system configuration"},
            {"name": "File Management", "description": "File upload and management operations"},
            {"name": "Vector Indexes", "description": "Vector index creation and inspection"},
            {"name": "Router", "description": "Router agent creation and management"},
            {"name": "Query", "description": "Query execution and testing"},
            {"name": "Monitoring", "description": "System monitoring and data export"}
        ],
        "paths": {
            "/api/health": {
                "get": {
                    "tags": ["System"],
                    "summary": "Health Check",
                    "description": "Check system health and basic configuration status",
                    "responses": {
                        "200": {
                            "description": "System health status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "example": "healthy"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "configuration": {
                                                "type": "object",
                                                "properties": {
                                                    "llm_configured": {"type": "boolean"},
                                                    "embedding_configured": {"type": "boolean"},
                                                    "router_configured": {"type": "boolean"},
                                                    "files_uploaded": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/models": {
                "get": {
                    "tags": ["Configuration"],
                    "summary": "Get Available Models",
                    "description": "Retrieve list of available LLM and embedding models",
                    "responses": {
                        "200": {
                            "description": "Available models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "llm_models": {
                                                "type": "object",
                                                "example": {
                                                    "deepseek-coder-33b-instruct": "DeepSeek Coder 33B",
                                                    "open_gpt4_8x7b_v0.2": "OpenGPT4 8x7B"
                                                }
                                            },
                                            "embedding_models": {
                                                "type": "object",
                                                "example": {
                                                    "all-MiniLM-L6-v2": "HuggingFace MiniLM-L6-v2",
                                                    "text-embedding-ada-002": "OpenAI Ada-002"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/configure": {
                "post": {
                    "tags": ["Configuration"],
                    "summary": "Configure Models",
                    "description": "Configure LLM and embedding models for the application",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "llm_model": {
                                            "type": "string",
                                            "description": "LLM model identifier",
                                            "example": "deepseek-coder-33b-instruct"
                                        },
                                        "embedding_model": {
                                            "type": "string",
                                            "description": "Embedding model identifier",
                                            "example": "all-MiniLM-L6-v2"
                                        },
                                        "huggingface_token": {
                                            "type": "string",
                                            "description": "HuggingFace API token (optional)",
                                            "example": "hf_xxxxxxxxxxxx"
                                        },
                                        "lm_studio_url": {
                                            "type": "string",
                                            "default": "http://127.0.0.1:1234/v1",
                                            "description": "LM Studio server URL",
                                            "example": "http://127.0.0.1:1234/v1"
                                        }
                                    },
                                    "required": ["llm_model", "embedding_model"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Configuration successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "Configuration failed",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/upload": {
                "post": {
                    "tags": ["File Management"],
                    "summary": "Upload Files",
                    "description": "Upload one or more documents for processing",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "format": "binary"
                                            },
                                            "description": "Files to upload (PDF, TXT, DOCX, DOC, MD)"
                                        }
                                    },
                                    "required": ["files"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Upload successful",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "files": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/FileInfo"}
                                            },
                                            "message": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "No files provided",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        },
                        "413": {
                            "description": "File too large",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/files": {
                "get": {
                    "tags": ["File Management"],
                    "summary": "List Files",
                    "description": "Get list of all uploaded files",
                    "responses": {
                        "200": {
                            "description": "List of files",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "files": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/FileInfo"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/files/{file_id}": {
                "delete": {
                    "tags": ["File Management"],
                    "summary": "Delete File",
                    "description": "Delete an uploaded file and its associated data",
                    "parameters": [
                        {
                            "name": "file_id",
                            "in": "path",
                            "required": True,
                            "description": "File ID to delete",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "File deleted successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                                }
                            }
                        },
                        "404": {
                            "description": "File not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/create-indexes": {
                "post": {
                    "tags": ["Vector Indexes"],
                    "summary": "Create Vector Indexes",
                    "description": "Create vector indexes for uploaded files",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Specific file IDs to process (optional, defaults to all)"
                                        },
                                        "chunk_size": {
                                            "type": "integer",
                                            "default": 1024,
                                            "minimum": 256,
                                            "maximum": 4096,
                                            "description": "Text chunk size for vectorization"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Indexes created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "results": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "file_id": {"type": "string"},
                                                        "filename": {"type": "string"},
                                                        "nodes_count": {"type": "integer"},
                                                        "success": {"type": "boolean"},
                                                        "error": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "message": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Embedding model not configured",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/indexes": {
                "get": {
                    "tags": ["Vector Indexes"],
                    "summary": "List All Indexes",
                    "description": "Get a simple list of all created vector indexes with basic information",
                    "responses": {
                        "200": {
                            "description": "List of indexes",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "total_indexes": {"type": "integer"},
                                            "indexes": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "file_id": {"type": "string"},
                                                        "file_name": {"type": "string"},
                                                        "file_size": {"type": "integer"},
                                                        "nodes_count": {"type": "integer"},
                                                        "chunk_size": {"type": "integer"},
                                                        "created_at": {"type": "string", "format": "date-time"},
                                                        "processed": {"type": "boolean"},
                                                        "vector_store_type": {"type": "string"},
                                                        "has_query_engine": {"type": "boolean"}
                                                    }
                                                }
                                            },
                                            "summary": {
                                                "type": "object",
                                                "properties": {
                                                    "total_files_indexed": {"type": "integer"},
                                                    "total_nodes": {"type": "integer"},
                                                    "total_size_bytes": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/indexes/{file_id}": {
                "get": {
                    "tags": ["Vector Indexes"],
                    "summary": "Get Index Details",
                    "description": "Get detailed information about a specific vector index",
                    "parameters": [
                        {
                            "name": "file_id",
                            "in": "path",
                            "required": True,
                            "description": "File ID of the index to retrieve",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Index details",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "index": {
                                                "type": "object",
                                                "properties": {
                                                    "file_id": {"type": "string"},
                                                    "file_info": {"$ref": "#/components/schemas/FileInfo"},
                                                    "index_info": {
                                                        "type": "object",
                                                        "properties": {
                                                            "nodes_count": {"type": "integer"},
                                                            "chunk_size": {"type": "integer"},
                                                            "vector_store_type": {"type": "string"},
                                                            "has_query_engine": {"type": "boolean"},
                                                            "embedding_dimension": {"type": "integer"},
                                                            "stored_vectors": {"type": "integer"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Index not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/inspect": {
                "get": {
                    "tags": ["Vector Indexes"],
                    "summary": "Inspect Vector Indexes",
                    "description": "Get detailed information about created vector indexes",
                    "responses": {
                        "200": {
                            "description": "Index inspection data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "indexes": {
                                                "type": "object",
                                                "additionalProperties": {
                                                    "type": "object",
                                                    "properties": {
                                                        "file_info": {"$ref": "#/components/schemas/FileInfo"},
                                                        "nodes_count": {"type": "integer"},
                                                        "chunk_size": {"type": "integer"},
                                                        "vector_store_type": {"type": "string"},
                                                        "embedding_dimension": {"type": "integer"},
                                                        "stored_vectors": {"type": "integer"}
                                                    }
                                                }
                                            },
                                            "summary": {
                                                "type": "object",
                                                "properties": {
                                                    "total_indexes": {"type": "integer"},
                                                    "total_files": {"type": "integer"},
                                                    "processed_files": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/create-router": {
                "post": {
                    "tags": ["Router"],
                    "summary": "Create Router Agent",
                    "description": "Create a router agent for intelligent query routing",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "router_type": {
                                            "type": "string",
                                            "enum": ["llm", "keyword", "hybrid"],
                                            "default": "hybrid",
                                            "description": "Type of router to create"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Router created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "router_type": {"type": "string"},
                                            "tools_count": {"type": "integer"},
                                            "tool_names": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "message": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "LLM not configured or no query engines available",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/query": {
                "post": {
                    "tags": ["Query"],
                    "summary": "Execute Query",
                    "description": "Execute a query using the configured router with detailed routing intelligence",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Query text to execute",
                                            "example": "What are the key features of AeroFlow?"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Query executed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "query": {"type": "string"},
                                            "response": {"type": "string"},
                                            "response_time": {"type": "number"},
                                            "response_length": {"type": "integer"},
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "routing_info": {"type": "object"},
                                            "routing_intelligence": {
                                                "type": "object",
                                                "properties": {
                                                    "router_type": {"type": "string"},
                                                    "method_used": {"type": "string"},
                                                    "decision": {"type": "string"},
                                                    "scores": {"type": "object"},
                                                    "reasoning_steps": {"type": "array"},
                                                    "final_reasoning": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Router not configured or query missing",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/test": {
                "post": {
                    "tags": ["Query"],
                    "summary": "Run Comprehensive Tests",
                    "description": "Execute comprehensive router tests with detailed intelligence analysis",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "test_type": {
                                            "type": "string",
                                            "enum": ["basic", "comprehensive"],
                                            "default": "comprehensive",
                                            "description": "Type of test suite to run"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Tests completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "test_session": {
                                                "type": "object",
                                                "properties": {
                                                    "test_type": {"type": "string"},
                                                    "timestamp": {"type": "string", "format": "date-time"},
                                                    "summary": {
                                                        "type": "object",
                                                        "properties": {
                                                            "total_tests": {"type": "integer"},
                                                            "successful_tests": {"type": "integer"},
                                                            "success_rate": {"type": "number"},
                                                            "average_response_time": {"type": "number"},
                                                            "categories_tested": {"type": "integer"},
                                                            "routing_methods_used": {
                                                                "type": "array",
                                                                "items": {"type": "string"}
                                                            }
                                                        }
                                                    },
                                                    "results": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "test_id": {"type": "integer"},
                                                                "query": {"type": "string"},
                                                                "category": {"type": "string"},
                                                                "response": {"type": "string"},
                                                                "success": {"type": "boolean"},
                                                                "response_time": {"type": "number"},
                                                                "routing_intelligence": {"type": "object"}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Router not configured",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/routing-analysis": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Get Routing Analysis",
                    "description": "Get detailed routing analysis and statistics",
                    "responses": {
                        "200": {
                            "description": "Routing analysis data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "router_type": {"type": "string"},
                                            "total_queries": {"type": "integer"},
                                            "routing_statistics": {"type": "object"},
                                            "recent_decisions": {"type": "array"},
                                            "llm_failures": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/status": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Get System Status",
                    "description": "Get comprehensive system status and configuration",
                    "responses": {
                        "200": {
                            "description": "System status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "configuration": {
                                                "type": "object",
                                                "properties": {
                                                    "llm_model": {"type": "string"},
                                                    "embedding_model": {"type": "string"},
                                                    "lm_studio_url": {"type": "string"},
                                                    "huggingface_token": {"type": "string"}
                                                }
                                            },
                                            "system_status": {
                                                "type": "object",
                                                "properties": {
                                                    "llm_configured": {"type": "boolean"},
                                                    "embedding_configured": {"type": "boolean"},
                                                    "router_configured": {"type": "boolean"},
                                                    "files_uploaded": {"type": "integer"},
                                                    "indexes_created": {"type": "integer"},
                                                    "queries_executed": {"type": "integer"}
                                                }
                                            },
                                            "router_info": {"type": "object"},
                                            "recent_activity": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/export-data": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Export Application Data",
                    "description": "Export all application data as JSON file",
                    "responses": {
                        "200": {
                            "description": "JSON export file",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "timestamp": {"type": "string", "format": "date-time"},
                                            "configuration": {"type": "object"},
                                            "files": {"type": "object"},
                                            "test_results": {"type": "array"},
                                            "summary": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "success": {"type": "boolean", "default": False}
                    },
                    "required": ["error", "success"]
                },
                "SuccessResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "default": True},
                        "message": {"type": "string"}
                    },
                    "required": ["success"]
                },
                "FileInfo": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique file identifier"},
                        "original_name": {"type": "string", "description": "Original filename"},
                        "filename": {"type": "string", "description": "Sanitized filename"},
                        "path": {"type": "string", "description": "File storage path"},
                        "size": {"type": "integer", "description": "File size in bytes"},
                        "uploaded_at": {"type": "string", "format": "date-time"},
                        "processed": {"type": "boolean", "description": "Whether file has been indexed"},
                        "nodes_count": {"type": "integer", "description": "Number of text chunks created"},
                        "chunk_size": {"type": "integer", "description": "Size of text chunks used"}
                    },
                    "required": ["id", "original_name", "size", "uploaded_at"]
                }
            }
        }
    }
    return jsonify(spec)


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
def cleanup_temp_files():
    """Clean up temporary files on shutdown"""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file_info in application_state['uploaded_files'].values():
                if os.path.exists(file_info['path']):
                    os.remove(file_info['path'])
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


# Development server setup
def create_app():
    """Application factory for testing"""
    return app


if __name__ == '__main__':
    print("🚀 Starting Agent Router Web Interface...")
    print("📱 Frontend: http://localhost:5010")
    print("📚 API Docs: http://localhost:5010/api/docs")
    print("🔍 Health Check: http://localhost:5010/api/health")
    print("\n" + "=" * 60)
    print("🤖 AGENT ROUTER WEB INTERFACE")
    print("=" * 60)
    print("Features available:")
    print("✅ Model Configuration (LLM + Embeddings)")
    print("✅ File Upload & Management")
    print("✅ Vector Index Creation")
    print("✅ Router Setup (LLM/Keyword/Hybrid)")
    print("✅ Interactive Testing")
    print("✅ Performance Monitoring")
    print("✅ Data Export")
    print("✅ Swagger API Documentation")
    print("=" * 60)
    print("\nStarting server on port 5010...")

    try:
        # Register cleanup
        import atexit

        atexit.register(cleanup_temp_files)

        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5010,
            debug=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        cleanup_temp_files()
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        cleanup_temp_files()


# Additional utility functions for the web interface

def validate_file_type(filename):
    """Validate uploaded file type"""
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md'}
    return Path(filename).suffix.lower() in allowed_extensions


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def sanitize_filename(filename):
    """Sanitize filename for safe storage"""
    # Remove or replace problematic characters
    import re
    filename = re.sub(r'[^\w\s-.]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-.')


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor system performance and usage"""

    def __init__(self):
        self.start_time = datetime.now()
        self.query_times = []
        self.error_count = 0

    def log_query_time(self, duration):
        """Log query execution time"""
        self.query_times.append(duration)
        # Keep only last 100 entries
        if len(self.query_times) > 100:
            self.query_times = self.query_times[-100:]

    def log_error(self):
        """Log an error occurrence"""
        self.error_count += 1

    def get_stats(self):
        """Get performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        stats = {
            'uptime_seconds': uptime,
            'total_queries': len(self.query_times),
            'error_count': self.error_count,
            'average_query_time': sum(self.query_times) / len(self.query_times) if self.query_times else 0,
            'min_query_time': min(self.query_times) if self.query_times else 0,
            'max_query_time': max(self.query_times) if self.query_times else 0
        }

        return stats


# Initialize performance monitor
performance_monitor = PerformanceMonitor()


# Add performance monitoring to query endpoint
def monitor_query_performance(func):
    """Decorator to monitor query performance"""

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            performance_monitor.log_query_time(duration)
            return result
        except Exception as e:
            performance_monitor.log_error()
            raise

    return wrapper


# WebSocket support for real-time updates (optional enhancement)
try:
    from flask_socketio import SocketIO, emit

    socketio = SocketIO(app, cors_allowed_origins="*")


    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        emit('status', {'message': 'Connected to Agent Router'})


    @socketio.on('request_status')
    def handle_status_request():
        """Handle status request via WebSocket"""
        try:
            response = requests.get('http://localhost:5010/api/status')
            emit('status_update', response.json())
        except:
            emit('status_update', {'error': 'Failed to get status'})


    def broadcast_update(event_type, data):
        """Broadcast update to all connected clients"""
        socketio.emit('update', {'type': event_type, 'data': data})

except ImportError:
    # SocketIO not available, continue without real-time features
    socketio = None


    def broadcast_update(event_type, data):
        pass


# Configuration validation
def validate_configuration(config):
    """Validate configuration parameters"""
    errors = []

    if not config.get('llm_model'):
        errors.append("LLM model is required")

    if not config.get('embedding_model'):
        errors.append("Embedding model is required")

    if config.get('lm_studio_url'):
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(config['lm_studio_url']):
            errors.append("Invalid LM Studio URL format")

    return errors


# Batch processing utilities
def process_multiple_files_async(file_ids, chunk_size=1024):
    """Process multiple files asynchronously"""
    import threading
    import queue

    results = queue.Queue()
    threads = []

    def process_file(file_id):
        try:
            # Process individual file
            file_info = application_state['uploaded_files'][file_id]
            documents = SimpleDirectoryReader(input_files=[file_info['path']]).load_data()
            splitter = SentenceSplitter(chunk_size=chunk_size)
            nodes = splitter.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes)

            results.put({
                'file_id': file_id,
                'success': True,
                'nodes_count': len(nodes),
                'index': index
            })

        except Exception as e:
            results.put({
                'file_id': file_id,
                'success': False,
                'error': str(e)
            })

    # Start threads for each file
    for file_id in file_ids:
        thread = threading.Thread(target=process_file, args=(file_id,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results
    processed_results = []
    while not results.empty():
        processed_results.append(results.get())

    return processed_results


# Add this to the create_indexes endpoint for better performance with multiple files
# (This would replace the synchronous processing in the existing endpoint)

print("🔧 Agent Router Web Interface loaded successfully!")
print("📋 Ready to serve requests on http://localhost:5010")
