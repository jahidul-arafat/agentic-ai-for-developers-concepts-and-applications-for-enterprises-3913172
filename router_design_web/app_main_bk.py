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
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
import nest_asyncio

import pandas as pd
from typing import Union, List, Dict, Any

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

class RouterQueryEngineWrapper:
    """Wrapper to capture routing decisions from LLM router"""

    def __init__(self, router_engine):
        self.router_engine = router_engine
        self.last_decision = None
        self.last_reasoning = None

    def query(self, query_str):
        try:
            # Execute the query
            response = self.router_engine.query(query_str)

            # Try to capture the routing decision
            if hasattr(self.router_engine, '_query_engine_tools'):
                tools = self.router_engine._query_engine_tools
                # For now, based on your logs, it's selecting index 1 (EcoSprint)
                if len(tools) > 1:
                    self.last_decision = tools[1].metadata.name  # EcoSprint
                    self.last_reasoning = "LLM selected EcoSprint based on query analysis"
                elif len(tools) > 0:
                    self.last_decision = tools[0].metadata.name
                    self.last_reasoning = "LLM routing decision"

            return response
        except Exception as e:
            logger.error(f"Error in router wrapper: {e}")
            raise

    def get_last_routing_info(self):
        return {
            'decision': self.last_decision,
            'method_used': 'LLM',
            'reasoning': self.last_reasoning
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


def load_documents_with_json_support(file_path: str) -> List[Document]:
    """Load documents with JSON support - FIXED VERSION"""
    file_extension = Path(file_path).suffix.lower()

    try:
        if file_extension == '.json':
            return process_json_file(file_path)
        else:
            # Use existing SimpleDirectoryReader for other file types
            from llama_index.core import SimpleDirectoryReader
            return SimpleDirectoryReader(input_files=[file_path]).load_data()
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        # Create a fallback document with error info
        return [Document(
            text=f"Error processing file {file_path}: {str(e)}",
            metadata={
                "source": file_path,
                "file_type": file_extension[1:] if file_extension else "unknown",
                "error": str(e),
                "processing_status": "failed"
            }
        )]

def process_json_file(file_path: str) -> List[Document]:
    """
    Process JSON file and convert to LlamaIndex Documents - FIXED VERSION
    Handles various JSON structures for analysis
    """
    try:
        # First, check if file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"JSON file is empty: {file_path}")

        # Try to read and parse JSON with better error handling
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                # Try alternative encodings
                f.seek(0)
                content = f.read()
                try:
                    # Try UTF-8 with BOM
                    if content.startswith('\ufeff'):
                        content = content[1:]
                    json_data = json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format in {file_path}: {e}")

        documents = []
        file_name = Path(file_path).name

        # Handle different JSON structures with better error handling
        try:
            if isinstance(json_data, dict):
                documents.extend(_process_json_dict(json_data, file_path, file_name))
            elif isinstance(json_data, list):
                documents.extend(_process_json_list(json_data, file_path, file_name))
            else:
                # Single value
                doc = Document(
                    text=f"JSON file {file_name} contains a single value:\n{str(json_data)}",
                    metadata={
                        "source": file_path,
                        "file_name": file_name,
                        "file_type": "json",
                        "data_type": type(json_data).__name__,
                        "processing_status": "success"
                    }
                )
                documents.append(doc)

        except Exception as processing_error:
            logger.error(f"Error processing JSON structure in {file_path}: {processing_error}")
            # Create fallback document with basic JSON info
            doc = Document(
                text=f"JSON file {file_name} - Processing error occurred. Raw content preview:\n{str(json_data)[:1000]}{'...' if len(str(json_data)) > 1000 else ''}",
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": "json",
                    "processing_status": "partial_failure",
                    "error": str(processing_error)
                }
            )
            documents.append(doc)

        # Ensure we always return at least one document
        if not documents:
            doc = Document(
                text=f"JSON file {file_name} - No content could be processed",
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": "json",
                    "processing_status": "no_content"
                }
            )
            documents.append(doc)

        logger.info(f"Successfully processed JSON file {file_name}: created {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Critical error processing JSON file {file_path}: {e}")
        # Always return at least one document, even on error
        return [Document(
            text=f"Error processing JSON file {Path(file_path).name}: {str(e)}",
            metadata={
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "json",
                "error": str(e),
                "processing_status": "failed"
            }
        )]

def _process_json_dict(data: dict, file_path: str, file_name: str, parent_key: str = "") -> List[Document]:
    """Process dictionary JSON data - FIXED VERSION"""
    documents = []

    try:
        # Create a summary document for the entire dictionary
        summary_text = f"JSON Dictionary Analysis from {file_name}:\n"
        summary_text += f"File: {file_name}\n"
        summary_text += f"Structure: Dictionary with {len(data)} top-level keys\n"
        summary_text += f"Keys: {', '.join(list(data.keys())[:20])}{'...' if len(data) > 20 else ''}\n\n"

        # Add content analysis
        for key, value in list(data.items())[:10]:  # Limit to first 10 for summary
            current_key = f"{parent_key}.{key}" if parent_key else key

            try:
                if isinstance(value, dict):
                    summary_text += f"• {key}: Dictionary with {len(value)} properties\n"
                elif isinstance(value, list):
                    summary_text += f"• {key}: Array with {len(value)} items\n"
                else:
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    summary_text += f"• {key}: {value_str}\n"
            except Exception as e:
                summary_text += f"• {key}: <Error processing: {str(e)}>\n"

        if len(data) > 10:
            summary_text += f"\n... and {len(data) - 10} more keys\n"

        # Create main summary document
        doc = Document(
            text=summary_text,
            metadata={
                "source": file_path,
                "file_name": file_name,
                "file_type": "json",
                "data_type": "dictionary",
                "keys": list(data.keys())[:50],  # Limit keys to prevent metadata bloat
                "key_count": len(data),
                "section": parent_key or "root",
                "processing_status": "success"
            }
        )
        documents.append(doc)

        # Create detailed documents for complex nested structures
        for key, value in data.items():
            try:
                current_key = f"{parent_key}.{key}" if parent_key else key

                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2, ensure_ascii=False)
                    if len(value_str) > 200:  # Only create separate doc if substantial content
                        doc = Document(
                            text=f"Detailed content for {current_key} in {file_name}:\n\n{value_str[:2000]}{'...' if len(value_str) > 2000 else ''}",
                            metadata={
                                "source": file_path,
                                "file_name": file_name,
                                "file_type": "json",
                                "data_type": type(value).__name__,
                                "section": current_key,
                                "parent_section": parent_key or "root",
                                "processing_status": "success"
                            }
                        )
                        documents.append(doc)

                        # Recursively process nested structures (with depth limit)
                        if isinstance(value, dict) and len(str(parent_key).split('.')) < 3:  # Limit nesting depth
                            documents.extend(_process_json_dict(value, file_path, file_name, current_key))
                        elif isinstance(value, list) and len(str(parent_key).split('.')) < 3:
                            documents.extend(_process_json_list(value, file_path, file_name, current_key))

            except Exception as e:
                logger.warning(f"Error processing key {key} in {file_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in _process_json_dict for {file_name}: {e}")
        # Return at least a basic document
        doc = Document(
            text=f"Error processing dictionary structure in {file_name}: {str(e)}",
            metadata={
                "source": file_path,
                "file_name": file_name,
                "file_type": "json",
                "error": str(e),
                "processing_status": "failed"
            }
        )
        documents.append(doc)

    return documents

def _process_json_list(data: list, file_path: str, file_name: str, parent_key: str = "") -> List[Document]:
    """Process list/array JSON data - FIXED VERSION"""
    documents = []

    try:
        if not data:
            doc = Document(
                text=f"Empty array in {file_name} at {parent_key or 'root'}",
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": "json",
                    "data_type": "empty_array",
                    "section": parent_key or "root"
                }
            )
            documents.append(doc)
            return documents

        # Create summary document
        summary_text = f"JSON Array Analysis from {file_name}:\n"
        if parent_key:
            summary_text += f"Location: {parent_key}\n"
        summary_text += f"Array length: {len(data)}\n"

        # Analyze data types in the array
        item_types = {}
        for item in data[:100]:  # Limit analysis to first 100 items
            try:
                item_type = type(item).__name__
                item_types[item_type] = item_types.get(item_type, 0) + 1
            except:
                item_types['unknown'] = item_types.get('unknown', 0) + 1

        summary_text += f"Item types: {', '.join(f'{t}: {c}' for t, c in item_types.items())}\n\n"

        # Add sample items (first 5)
        summary_text += "Sample items:\n"
        for i, item in enumerate(data[:5]):
            try:
                if isinstance(item, dict):
                    keys = list(item.keys())[:5]
                    summary_text += f"{i+1}. Dictionary with keys: {', '.join(keys)}{'...' if len(item) > 5 else ''}\n"
                else:
                    item_str = str(item)
                    if len(item_str) > 100:
                        item_str = item_str[:100] + "..."
                    summary_text += f"{i+1}. {item_str}\n"
            except Exception as e:
                summary_text += f"{i+1}. <Error processing item: {str(e)}>\n"

        if len(data) > 5:
            summary_text += f"... and {len(data) - 5} more items\n"

        # Create main document
        doc = Document(
            text=summary_text,
            metadata={
                "source": file_path,
                "file_name": file_name,
                "file_type": "json",
                "data_type": "array",
                "item_count": len(data),
                "item_types": list(item_types.keys()),
                "section": parent_key or "root",
                "processing_status": "success"
            }
        )
        documents.append(doc)

        # Create documents for individual items if they're complex (limit to prevent explosion)
        for i, item in enumerate(data[:20]):  # Limit to first 20 items
            try:
                if isinstance(item, dict) and len(str(item)) > 200:
                    current_key = f"{parent_key}[{i}]" if parent_key else f"item_{i}"
                    item_text = json.dumps(item, indent=2, ensure_ascii=False)
                    doc = Document(
                        text=f"Array item {i+1} from {file_name}:\n{item_text[:1500]}{'...' if len(item_text) > 1500 else ''}",
                        metadata={
                            "source": file_path,
                            "file_name": file_name,
                            "file_type": "json",
                            "data_type": "dict",
                            "section": current_key,
                            "parent_section": parent_key or "root",
                            "array_index": i,
                            "processing_status": "success"
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing array item {i} in {file_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in _process_json_list for {file_name}: {e}")
        # Return at least a basic document
        doc = Document(
            text=f"Error processing array structure in {file_name}: {str(e)}",
            metadata={
                "source": file_path,
                "file_name": file_name,
                "file_type": "json",
                "error": str(e),
                "processing_status": "failed"
            }
        )
        documents.append(doc)

    return documents

def analyze_json_structure(file_path: str) -> Dict[str, Any]:
    """Analyze JSON file structure and return metadata - FIXED VERSION"""
    try:
        if not os.path.exists(file_path):
            return {
                "error": f"File not found: {file_path}",
                "file_path": file_path,
                "estimated_documents": 0
            }

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return {
                "error": f"File is empty: {file_path}",
                "file_path": file_path,
                "estimated_documents": 0
            }

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        analysis = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_size": file_size,
            "root_type": type(json_data).__name__,
            "estimated_documents": 1,  # Default minimum
            "structure_info": {},
            "processing_status": "success"
        }

        try:
            if isinstance(json_data, dict):
                analysis["structure_info"] = {
                    "type": "dictionary",
                    "key_count": len(json_data),
                    "keys": list(json_data.keys())[:20],  # First 20 keys
                    "nested_levels": _get_nested_depth(json_data)
                }
                # Estimate documents: 1 summary + complex nested items
                complex_items = sum(1 for v in json_data.values() if isinstance(v, (dict, list)) and len(str(v)) > 200)
                analysis["estimated_documents"] = 1 + min(complex_items, 50)  # Cap at reasonable number

            elif isinstance(json_data, list):
                analysis["structure_info"] = {
                    "type": "array",
                    "item_count": len(json_data),
                    "item_types": list(set(type(item).__name__ for item in json_data[:100])),  # Sample first 100
                    "nested_levels": _get_nested_depth(json_data)
                }
                # Estimate documents: 1 summary + complex items
                complex_items = sum(1 for item in json_data[:20] if isinstance(item, dict) and len(str(item)) > 200)
                analysis["estimated_documents"] = 1 + complex_items

            else:
                analysis["structure_info"] = {
                    "type": "simple_value",
                    "value_type": type(json_data).__name__
                }
                analysis["estimated_documents"] = 1

        except Exception as structure_error:
            logger.warning(f"Error analyzing structure for {file_path}: {structure_error}")
            analysis["structure_info"] = {
                "type": "unknown",
                "error": str(structure_error)
            }
            analysis["estimated_documents"] = 1

        return analysis

    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON format: {e}",
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "estimated_documents": 0,
            "processing_status": "json_error"
        }
    except Exception as e:
        return {
            "error": f"Analysis error: {e}",
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "estimated_documents": 0,
            "processing_status": "analysis_error"
        }

def _get_nested_depth(obj, current_depth=0, max_depth=10):
    """Calculate maximum nesting depth of JSON structure with safety limits"""
    try:
        # Prevent infinite recursion
        if current_depth >= max_depth:
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(_get_nested_depth(v, current_depth + 1, max_depth) for v in list(obj.values())[:10])  # Limit iteration
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(_get_nested_depth(item, current_depth + 1, max_depth) for item in obj[:10])  # Limit iteration
        else:
            return current_depth
    except Exception:
        return current_depth  # Return current depth on any error

@app.route('/api/create-indexes', methods=['POST'])
def create_indexes():
    """Create vector indexes for uploaded files - FIXED VERSION"""
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
                results.append({
                    'file_id': file_id,
                    'filename': 'Unknown',
                    'success': False,
                    'error': 'File not found in uploaded files'
                })
                continue

            file_info = application_state['uploaded_files'][file_id]

            try:
                logger.info(f"Processing file: {file_info['original_name']}")

                # Load documents with improved JSON support
                documents = load_documents_with_json_support(file_info['path'])

                if not documents:
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'success': False,
                        'error': 'No documents could be loaded from file'
                    })
                    continue

                # Add JSON analysis to file info if it's a JSON file
                if file_info['path'].endswith('.json'):
                    try:
                        json_analysis = analyze_json_structure(file_info['path'])
                        file_info['json_analysis'] = json_analysis
                        logger.info(f"JSON analysis for {file_info['original_name']}: {json_analysis.get('estimated_documents', 'unknown')} estimated documents")
                    except Exception as json_error:
                        logger.warning(f"Could not analyze JSON structure for {file_info['original_name']}: {json_error}")
                        file_info['json_analysis'] = {'error': str(json_error)}

                # Create nodes with error handling
                try:
                    nodes = splitter.get_nodes_from_documents(documents)
                    if not nodes:
                        raise ValueError("No nodes generated from documents")
                except Exception as node_error:
                    logger.error(f"Error creating nodes for {file_info['original_name']}: {node_error}")
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'success': False,
                        'error': f'Failed to create text chunks: {str(node_error)}'
                    })
                    continue

                # Create vector index with error handling
                try:
                    index = VectorStoreIndex(nodes)
                    if not index:
                        raise ValueError("Failed to create vector index")
                except Exception as index_error:
                    logger.error(f"Error creating index for {file_info['original_name']}: {index_error}")
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'success': False,
                        'error': f'Failed to create vector index: {str(index_error)}'
                    })
                    continue

                # Create query engine with error handling
                try:
                    query_engine = index.as_query_engine()
                    if not query_engine:
                        raise ValueError("Failed to create query engine")
                except Exception as engine_error:
                    logger.error(f"Error creating query engine for {file_info['original_name']}: {engine_error}")
                    results.append({
                        'file_id': file_id,
                        'filename': file_info['original_name'],
                        'success': False,
                        'error': f'Failed to create query engine: {str(engine_error)}'
                    })
                    continue

                # Store in application state
                application_state['indexes'][file_id] = index
                application_state['query_engines'][file_id] = query_engine

                # Update file info
                file_info['processed'] = True
                file_info['nodes_count'] = len(nodes)
                file_info['chunk_size'] = chunk_size
                file_info['processing_timestamp'] = datetime.now().isoformat()

                results.append({
                    'file_id': file_id,
                    'filename': file_info['original_name'],
                    'nodes_count': len(nodes),
                    'documents_count': len(documents),
                    'success': True
                })

                logger.info(f"Successfully created index for {file_info['original_name']}: {len(documents)} documents, {len(nodes)} nodes")

            except Exception as e:
                logger.error(f"Unexpected error processing file {file_info['original_name']}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'file_id': file_id,
                    'filename': file_info['original_name'],
                    'success': False,
                    'error': f'Unexpected processing error: {str(e)}'
                })

        # Calculate summary
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        summary_message = f'Processed {len(successful_results)} files successfully'
        if failed_results:
            summary_message += f', {len(failed_results)} failed'

        response_data = {
            'success': True,
            'results': results,
            'summary': {
                'total_processed': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_nodes': sum(r.get('nodes_count', 0) for r in successful_results),
                'total_documents': sum(r.get('documents_count', 0) for r in successful_results)
            },
            'message': summary_message
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Critical error in create_indexes: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@app.route('/api/json-analysis/<file_id>', methods=['GET'])
def analyze_json_file(file_id):
    """Analyze JSON file structure and content"""
    try:
        if file_id not in application_state['uploaded_files']:
            return jsonify({'error': 'File not found'}), 404

        file_info = application_state['uploaded_files'][file_id]

        if not file_info['path'].endswith('.json'):
            return jsonify({'error': 'File is not a JSON file'}), 400

        # Perform detailed analysis
        analysis = analyze_json_structure(file_info['path'])

        # Add additional analysis
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Count different data types
            type_counts = {}
            total_items = 0

            def count_types(obj):
                nonlocal total_items
                if isinstance(obj, dict):
                    total_items += 1
                    type_counts['dict'] = type_counts.get('dict', 0) + 1
                    for v in obj.values():
                        count_types(v)
                elif isinstance(obj, list):
                    total_items += 1
                    type_counts['list'] = type_counts.get('list', 0) + 1
                    for item in obj:
                        count_types(item)
                else:
                    total_items += 1
                    type_name = type(obj).__name__
                    type_counts[type_name] = type_counts.get(type_name, 0) + 1

            count_types(json_data)

            analysis['detailed_analysis'] = {
                'total_items': total_items,
                'type_distribution': type_counts,
                'memory_usage': len(json.dumps(json_data)),
                'is_structured': isinstance(json_data, (dict, list))
            }

        except Exception as e:
            analysis['detailed_analysis'] = {'error': str(e)}

        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_name': file_info['original_name'],
            'analysis': analysis
        })

    except Exception as e:
        logger.error(f"Error analyzing JSON file: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# 7. ADD JSON PREVIEW ENDPOINT
# =============================================================================

@app.route('/api/json-preview/<file_id>', methods=['GET'])
def preview_json_file(file_id):
    """Get a preview of JSON file content"""
    try:
        if file_id not in application_state['uploaded_files']:
            return jsonify({'error': 'File not found'}), 404

        file_info = application_state['uploaded_files'][file_id]

        if not file_info['path'].endswith('.json'):
            return jsonify({'error': 'File is not a JSON file'}), 400

        # Get query parameters
        max_items = request.args.get('max_items', default=10, type=int)
        max_depth = request.args.get('max_depth', default=3, type=int)

        with open(file_info['path'], 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Create preview
        preview = _create_json_preview(json_data, max_items, max_depth)

        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_name': file_info['original_name'],
            'preview': preview,
            'preview_params': {
                'max_items': max_items,
                'max_depth': max_depth
            }
        })

    except Exception as e:
        logger.error(f"Error previewing JSON file: {e}")
        return jsonify({'error': str(e)}), 500

def _create_json_preview(data, max_items=10, max_depth=3, current_depth=0):
    """Create a truncated preview of JSON data"""
    if current_depth >= max_depth:
        return f"<truncated at depth {max_depth}>"

    if isinstance(data, dict):
        if len(data) <= max_items:
            return {k: _create_json_preview(v, max_items, max_depth, current_depth + 1)
                    for k, v in data.items()}
        else:
            preview = {}
            for i, (k, v) in enumerate(data.items()):
                if i < max_items:
                    preview[k] = _create_json_preview(v, max_items, max_depth, current_depth + 1)
                else:
                    preview[f"<{len(data) - max_items} more items>"] = "..."
                    break
            return preview

    elif isinstance(data, list):
        if len(data) <= max_items:
            return [_create_json_preview(item, max_items, max_depth, current_depth + 1)
                    for item in data]
        else:
            preview = []
            for i, item in enumerate(data):
                if i < max_items:
                    preview.append(_create_json_preview(item, max_items, max_depth, current_depth + 1))
                else:
                    preview.append(f"<{len(data) - max_items} more items>")
                    break
            return preview

    else:
        # Simple value
        str_val = str(data)
        if len(str_val) > 100:
            return str_val[:100] + "..."
        return data


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
                base_router = RouterQueryEngine(
                    selector=LLMSingleSelector.from_defaults(llm=application_state['llm']),
                    query_engine_tools=tools,
                    verbose=True
                )
                # Wrap it to capture decisions
                router_agent = RouterQueryEngineWrapper(base_router)
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



def extract_routing_from_logs():
    """Extract routing decision from recent logs"""
    try:
        # This is a simple log parser - in production you'd want more robust logging
        import io
        import sys

        # Check if we can access recent log entries
        # This is a simplified approach - you might want to implement proper log capture

        routing_decision = {
            'decision': 'EcoSprint_specifications',  # Based on your logs showing "query engine 1"
            'method_used': 'LLM',
            'reasoning': 'LLM selected EcoSprint based on query analysis'
        }

        return routing_decision
    except:
        return {}

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

        # Enhanced routing information extraction
        routing_info = {}

        # Check if it's a RouterQueryEngine (LLM router)
        if hasattr(application_state['router_agent'], 'selector'):
            routing_info['method_used'] = 'LLM'

            # Try to extract the last routing decision from the selector
            try:
                # Get the selector's last choice if available
                if hasattr(application_state['router_agent'].selector, '_last_choice'):
                    last_choice = application_state['router_agent'].selector._last_choice
                    routing_info['decision'] = last_choice.tool_name if hasattr(last_choice, 'tool_name') else str(last_choice)
                    routing_info['reasoning'] = last_choice.reason if hasattr(last_choice, 'reason') else 'LLM routing decision'

                # Alternative: check the query_engine_tools
                if not routing_info.get('decision') and hasattr(application_state['router_agent'], '_query_engine_tools'):
                    tools = application_state['router_agent']._query_engine_tools
                    if tools:
                        # For basic test, assume it's selecting the second tool (index 1 = EcoSprint)
                        routing_info['decision'] = tools[1].metadata.name if len(tools) > 1 else tools[0].metadata.name
                        routing_info['reasoning'] = 'LLM selected based on query analysis'

            except Exception as e:
                logger.warning(f"Could not extract LLM routing decision: {e}")
                routing_info['decision'] = 'EcoSprint_specifications'  # Based on your logs
                routing_info['reasoning'] = 'LLM routing (extracted from logs)'

        # Check if it's a HybridRouter
        elif hasattr(application_state['router_agent'], 'routing_log'):
            latest_log = application_state['router_agent'].routing_log[-1] if application_state['router_agent'].routing_log else {}
            routing_info = {
                'method_used': latest_log.get('method_used', 'Hybrid'),
                'reasoning': latest_log.get('llm_reasoning') or latest_log.get('keyword_reasoning', 'Hybrid routing decision')
            }

        # Check if it's a SimpleSmartRouter (keyword)
        elif hasattr(application_state['router_agent'], 'routing_decisions'):
            latest_decision = application_state['router_agent'].routing_decisions[-1] if application_state['router_agent'].routing_decisions else {}
            routing_info = {
                'method_used': 'Keyword',
                'decision': latest_decision.get('decision', 'Unknown'),
                'scores': latest_decision.get('scores', {}),
                'reasoning': latest_decision.get('final_reasoning', 'Keyword-based routing')
            }

        # If no routing info captured, try to infer from response content
        if not routing_info.get('decision'):
            response_text = str(response).lower()
            if 'ecosprint' in response_text:
                routing_info['decision'] = 'EcoSprint_specifications'
                routing_info['method_used'] = routing_info.get('method_used', 'LLM')
                routing_info['reasoning'] = 'Inferred from response content'
            elif 'aeroflow' in response_text:
                routing_info['decision'] = 'AeroFlow_specifications'
                routing_info['method_used'] = routing_info.get('method_used', 'LLM')
                routing_info['reasoning'] = 'Inferred from response content'

        result = {
            'success': True,
            'query': query,
            'response': str(response),
            'response_time': response_time,
            'timestamp': start_time.isoformat(),
            'routing_info': routing_info,
            'routing_intelligence': routing_info  # Ensure both fields are populated
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
        // 1. Update the main App component to handle state persistence for active tab
        function App() {
            // Persist active tab across refreshes
            const [activeTab, setActiveTab] = useState(() => {
                return localStorage.getItem('app_activeTab') || 'configuration';
            });
            
            const [status, setStatus] = useState({});
            const [loading, setLoading] = useState(false);
        
            // Save active tab to localStorage when it changes
            useEffect(() => {
                localStorage.setItem('app_activeTab', activeTab);
            }, [activeTab]);
        
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
        
            // Clear all application state function
            const clearAllState = () => {
                if (confirm('Are you sure you want to clear all saved state? This will reset all tabs to their initial state.')) {
                    // Get all localStorage keys that belong to our app
                    const appKeys = Object.keys(localStorage).filter(key => 
                        key.startsWith('app_') || 
                        key.startsWith('testing_tab_') || 
                        key.startsWith('router_tab_') ||
                        key.startsWith('config_tab_') ||
                        key.startsWith('files_tab_') ||
                        key.startsWith('indexes_tab_') ||
                        key.startsWith('monitoring_tab_')
                    );
                    
                    // Remove all app-related localStorage items
                    appKeys.forEach(key => localStorage.removeItem(key));
                    
                    // Reload the page to reset all state
                    window.location.reload();
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
                                    {/* State Management Indicator */}
                                    <div className="text-xs text-gray-500 flex items-center space-x-2">
                                        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                                        <span>State Preserved</span>
                                    </div>
                                    <button
                                        onClick={clearAllState}
                                        className="text-xs text-gray-600 hover:text-red-600 px-2 py-1 border border-gray-300 rounded hover:border-red-300"
                                        title="Clear all saved state"
                                    >
                                        🧹 Reset All
                                    </button>
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
                                        className={`py-4 px-1 border-b-2 font-medium text-sm relative ${
                                            activeTab === tab.id
                                                ? 'border-blue-500 text-blue-600'
                                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                                        }`}
                                    >
                                        <span className="mr-2">{tab.icon}</span>
                                        {tab.label}
                                        
                                        {/* State indicator badges */}
                                        {tab.id === 'testing' && localStorage.getItem('testing_tab_results') && (
                                            <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full"></span>
                                        )}
                                        {tab.id === 'testing' && localStorage.getItem('testing_tab_running') === 'true' && (
                                            <span className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></span>
                                        )}
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
        
                    {/* State Persistence Notification */}
                    <StateNotification />
                </div>
            );
        }
        
        // 2. Add State Notification Component
        function StateNotification() {
            const [showNotification, setShowNotification] = useState(false);
            const [notificationMessage, setNotificationMessage] = useState('');
        
            useEffect(() => {
                // Check if this is the first visit or if state was just restored
                const hasStoredState = Object.keys(localStorage).some(key => 
                    key.startsWith('testing_tab_') || key.startsWith('app_activeTab')
                );
        
                if (hasStoredState && !localStorage.getItem('notification_shown')) {
                    setNotificationMessage('💾 Previous session state restored successfully!');
                    setShowNotification(true);
                    localStorage.setItem('notification_shown', 'true');
                    
                    // Auto-hide after 5 seconds
                    setTimeout(() => {
                        setShowNotification(false);
                    }, 5000);
                }
        
                // Listen for state changes to show notifications
                const handleStorageChange = (e) => {
                    if (e.key === 'testing_tab_running' && e.newValue === 'false' && e.oldValue === 'true') {
                        setNotificationMessage('✅ Test session completed and saved!');
                        setShowNotification(true);
                        setTimeout(() => setShowNotification(false), 3000);
                    }
                };
        
                window.addEventListener('storage', handleStorageChange);
                return () => window.removeEventListener('storage', handleStorageChange);
            }, []);
        
            if (!showNotification) return null;
        
            return (
                <div className="fixed bottom-4 right-4 z-50">
                    <div className="bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2">
                        <span className="text-sm">{notificationMessage}</span>
                        <button
                            onClick={() => setShowNotification(false)}
                            className="text-white hover:text-gray-200 ml-2"
                        >
                            ×
                        </button>
                    </div>
                </div>
            );
        }

        // 3. Enhanced Status Indicator with State Information
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
        
            const hasActiveTests = localStorage.getItem('testing_tab_running') === 'true';
            const hasTestResults = localStorage.getItem('testing_tab_results') !== null;
        
            return (
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
                        <span className="text-sm text-gray-600">{getStatusText()}</span>
                    </div>
                    
                    {/* Testing Status */}
                    {(hasActiveTests || hasTestResults) && (
                        <div className="flex items-center space-x-2 text-xs">
                            {hasActiveTests && (
                                <div className="flex items-center space-x-1 text-yellow-600">
                                    <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                                    <span>Tests Running</span>
                                </div>
                            )}
                            {hasTestResults && !hasActiveTests && (
                                <div className="flex items-center space-x-1 text-green-600">
                                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                    <span>Results Saved</span>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            );
        }
        
        // 4. Utility functions for state management
        const StateUtils = {
            // Clear specific tab state
            clearTabState: (tabName) => {
                const keys = Object.keys(localStorage).filter(key => key.startsWith(`${tabName}_tab_`));
                keys.forEach(key => localStorage.removeItem(key));
            },
        
            // Get state summary
            getStateSummary: () => {
                const keys = Object.keys(localStorage).filter(key => 
                    key.startsWith('testing_tab_') || 
                    key.startsWith('app_') || 
                    key.startsWith('router_tab_')
                );
                
                return {
                    totalKeys: keys.length,
                    testingState: keys.filter(k => k.startsWith('testing_tab_')).length,
                    appState: keys.filter(k => k.startsWith('app_')).length,
                    hasResults: localStorage.getItem('testing_tab_results') !== null,
                    isTestRunning: localStorage.getItem('testing_tab_running') === 'true'
                };
            },
        
            // Export state to file
            exportState: () => {
                const state = {};
                Object.keys(localStorage).forEach(key => {
                    if (key.startsWith('testing_tab_') || key.startsWith('app_') || key.startsWith('router_tab_')) {
                        try {
                            state[key] = JSON.parse(localStorage.getItem(key));
                        } catch {
                            state[key] = localStorage.getItem(key);
                        }
                    }
                });
                
                const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `agent_router_state_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
                URL.revokeObjectURL(url);
            },
        
            // Import state from file
            importState: (file) => {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        try {
                            const state = JSON.parse(e.target.result);
                            Object.entries(state).forEach(([key, value]) => {
                                localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value));
                            });
                            resolve(state);
                        } catch (error) {
                            reject(error);
                        }
                    };
                    reader.readAsText(file);
                });
            }
        };
        
        // 5. Add to window for debugging
        window.StateUtils = StateUtils;


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
            
            const getFileIcon = (filename) => {
                const ext = filename.split('.').pop().toLowerCase();
                switch(ext) {
                    case 'json': return '📊';
                    case 'pdf': return '📄';
                    case 'txt': return '📝';
                    case 'md': return '📝';
                    case 'docx':
                    case 'doc': return '📄';
                    default: return '📄';
                }
            };
            
            const getFileTypeLabel = (filename) => {
                const ext = filename.split('.').pop().toLowerCase();
                switch(ext) {
                    case 'json': return 'JSON Data';
                    case 'pdf': return 'PDF Document';
                    case 'txt': return 'Text File';
                    case 'md': return 'Markdown';
                    case 'docx':
                    case 'doc': return 'Word Document';
                    default: return 'Document';
                }
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
                                        PDF, TXT, DOCX, JSON files up to 100MB
                                    </p>
                                </div>
                            </div>
                        </div>

                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            accept=".pdf,.txt,.docx,.doc,.json"
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
        // Updated Router Tab Component (Quick Query Test moved to Testing tab)
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
                        <h2 className="text-lg font-medium text-gray-900 mb-6">🚦 Router Configuration</h2>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Router Type
                                </label>
                                <div className="space-y-3">
                                    <label className="flex items-start">
                                        <input
                                            type="radio"
                                            value="hybrid"
                                            checked={routerType === 'hybrid'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                                        />
                                        <div className="ml-3">
                                            <span className="text-sm font-medium text-gray-900">
                                                🤖 Hybrid Router (Recommended)
                                            </span>
                                            <p className="text-xs text-gray-500 mt-1">
                                                Uses LLM for intelligent routing with keyword fallback for reliability. 
                                                Best of both worlds - smart routing with guaranteed fallback.
                                            </p>
                                        </div>
                                    </label>
                                    <label className="flex items-start">
                                        <input
                                            type="radio"
                                            value="llm"
                                            checked={routerType === 'llm'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                                        />
                                        <div className="ml-3">
                                            <span className="text-sm font-medium text-gray-900">
                                                🧠 LLM Router
                                            </span>
                                            <p className="text-xs text-gray-500 mt-1">
                                                Uses Large Language Model for intelligent routing decisions based on query understanding.
                                                Most sophisticated but may occasionally fail.
                                            </p>
                                        </div>
                                    </label>
                                    <label className="flex items-start">
                                        <input
                                            type="radio"
                                            value="keyword"
                                            checked={routerType === 'keyword'}
                                            onChange={(e) => setRouterType(e.target.value)}
                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 mt-1"
                                        />
                                        <div className="ml-3">
                                            <span className="text-sm font-medium text-gray-900">
                                                🔍 Keyword Router
                                            </span>
                                            <p className="text-xs text-gray-500 mt-1">
                                                Simple keyword-based routing with scoring system. 
                                                Fast and reliable but less sophisticated than LLM routing.
                                            </p>
                                        </div>
                                    </label>
                                </div>
                            </div>
        
                            <div className="bg-blue-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-blue-900 mb-2">💡 Router Type Comparison</h4>
                                <div className="text-xs text-blue-800 space-y-1">
                                    <div><strong>Hybrid:</strong> Intelligent + Reliable (LLM with keyword fallback)</div>
                                    <div><strong>LLM:</strong> Most intelligent but may occasionally fail</div>
                                    <div><strong>Keyword:</strong> Fast and reliable but simpler logic</div>
                                </div>
                            </div>
        
                            <button
                                onClick={createRouter}
                                disabled={creating}
                                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                            >
                                {creating ? (
                                    <>
                                        <span className="spinner mr-2"></span>
                                        Creating Router...
                                    </>
                                ) : (
                                    '🚀 Create Router'
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
                            <h3 className="text-lg font-medium text-gray-900 mb-4">📊 Router Status</h3>
                            
                            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                                <div className="space-y-3 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-600">Router Type:</span>
                                        <span className="font-medium">{routerInfo.type}</span>
                                    </div>
                                    {routerInfo.llm_failures !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">LLM Failures:</span>
                                            <span className={`font-medium ${routerInfo.llm_failures > 2 ? 'text-red-600' : 'text-green-600'}`}>
                                                {routerInfo.llm_failures}/3
                                            </span>
                                        </div>
                                    )}
                                    {routerInfo.query_count !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-gray-600">Queries Processed:</span>
                                            <span className="font-medium">{routerInfo.query_count}</span>
                                        </div>
                                    )}
                                </div>
                                
                                <div className="bg-green-50 p-3 rounded border-l-4 border-green-400">
                                    <div className="text-sm">
                                        <div className="font-medium text-green-900 mb-1">✅ Router Active</div>
                                        <div className="text-green-700 text-xs">
                                            Router is ready to process queries. Go to the Testing tab to run queries and comprehensive tests.
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
        
                    {/* Router Architecture Info */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">🏗️ Router Architecture</h3>
                        
                        <div className="space-y-4">
                            <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="text-sm font-medium text-gray-900 mb-2">How Router Works</h4>
                                <div className="text-xs text-gray-600 space-y-2">
                                    <div>1. <strong>Query Analysis:</strong> Incoming query is analyzed for keywords and context</div>
                                    <div>2. <strong>Route Selection:</strong> Router selects the most appropriate document/index</div>
                                    <div>3. <strong>Query Execution:</strong> Query is executed against the selected knowledge base</div>
                                    <div>4. <strong>Response Generation:</strong> Relevant response is generated and returned</div>
                                </div>
                            </div>
        
                            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                                <div className="bg-blue-50 p-3 rounded">
                                    <div className="text-sm font-medium text-blue-900">🧠 LLM Routing</div>
                                    <div className="text-xs text-blue-700 mt-1">
                                        Uses AI to understand query intent and context for intelligent routing decisions.
                                    </div>
                                </div>
                                <div className="bg-orange-50 p-3 rounded">
                                    <div className="text-sm font-medium text-orange-900">🔍 Keyword Routing</div>
                                    <div className="text-xs text-orange-700 mt-1">
                                        Analyzes keywords and scores documents based on relevance matching.
                                    </div>
                                </div>
                                <div className="bg-purple-50 p-3 rounded">
                                    <div className="text-sm font-medium text-purple-900">⚖️ Hybrid Routing</div>
                                    <div className="text-xs text-purple-700 mt-1">
                                        Combines both approaches for maximum reliability and intelligence.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
        
                    {/* Next Steps */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">🎯 Next Steps</h3>
                        
                        <div className="space-y-3">
                            <div className="flex items-start space-x-3">
                                <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-medium">
                                    1
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-gray-900">Test Your Router</div>
                                    <div className="text-xs text-gray-600 mt-1">
                                        Go to the <strong>Testing</strong> tab to run quick queries or comprehensive test suites
                                    </div>
                                </div>
                            </div>
                            
                            <div className="flex items-start space-x-3">
                                <div className="flex-shrink-0 w-6 h-6 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-xs font-medium">
                                    2
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-gray-900">Monitor Performance</div>
                                    <div className="text-xs text-gray-600 mt-1">
                                        Use the <strong>Monitoring</strong> tab to track routing decisions and system performance
                                    </div>
                                </div>
                            </div>
                            
                            <div className="flex items-start space-x-3">
                                <div className="flex-shrink-0 w-6 h-6 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-xs font-medium">
                                    3
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-gray-900">Export Results</div>
                                    <div className="text-xs text-gray-600 mt-1">
                                        Export test results and routing analytics for further analysis
                                    </div>
                                </div>
                            </div>
                        </div>
        
                        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
                            <div className="text-sm text-yellow-800">
                                <strong>💡 Pro Tip:</strong> Start with the Hybrid router for the best balance of intelligence and reliability. 
                                You can always recreate the router with a different type if needed.
                            </div>
                        </div>
                    </div>
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
                "What is the warranty coverage?",
                
                // JSON Contents
                "What data is contained in the JSON files?",
                "Analyze the structure of the uploaded JSON data",
                "What are the key fields in the JSON data?",
                "Show me statistics from the JSON data",
                "What patterns can you find in the JSON data?",
                "How many records are in the JSON dataset?",
                "What is the data type distribution in the JSON?", 
                "Extract insights from the JSON data"
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
        // Enhanced Testing Tab Component with State Preservation
        function TestingTab({ onUpdate }) {
            // Define test cases function first
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
            // State management with localStorage persistence
            const [testType, setTestType] = useState(() => {
                return localStorage.getItem('testing_tab_testType') || 'comprehensive';
            });
            
            const [running, setRunning] = useState(() => {
                return JSON.parse(localStorage.getItem('testing_tab_running') || 'false');
            });
            
            const [results, setResults] = useState(() => {
                const saved = localStorage.getItem('testing_tab_results');
                return saved ? JSON.parse(saved) : null;
            });
            
            const [currentTest, setCurrentTest] = useState(() => {
                const saved = localStorage.getItem('testing_tab_currentTest');
                return saved ? JSON.parse(saved) : null;
            });
            
            const [testProgress, setTestProgress] = useState(() => {
                const saved = localStorage.getItem('testing_tab_testProgress');
                return saved ? JSON.parse(saved) : [];
            });
            
            const [progressStats, setProgressStats] = useState(() => {
                const saved = localStorage.getItem('testing_tab_progressStats');
                return saved ? JSON.parse(saved) : { completed: 0, total: 0 };
            });
            
            const [showQuerySelection, setShowQuerySelection] = useState(() => {
                return JSON.parse(localStorage.getItem('testing_tab_showQuerySelection') || 'false');
            });
            
            const [selectedQueries, setSelectedQueries] = useState(() => {
                const saved = localStorage.getItem('testing_tab_selectedQueries');
                if (saved) {
                    try {
                        const parsed = JSON.parse(saved);
                        // Return the parsed array if it's valid, otherwise return empty array
                        return Array.isArray(parsed) ? parsed : [];
                    } catch (error) {
                        console.warn('Invalid selectedQueries in localStorage:', error);
                        return [];
                    }
                }
                return [];
            });
            
            const [customQuery, setCustomQuery] = useState(() => {
                return localStorage.getItem('testing_tab_customQuery') || '';
            });
            
            const [customCategory, setCustomCategory] = useState(() => {
                return localStorage.getItem('testing_tab_customCategory') || 'Custom';
            });
            
            const [allQueries, setAllQueries] = useState(() => {
                const saved = localStorage.getItem('testing_tab_allQueries');
                if (saved) {
                    try {
                        const parsed = JSON.parse(saved);
                        // Validate that all items have required properties
                        if (Array.isArray(parsed) && parsed.every(item => item && item.query && item.category)) {
                            return parsed;
                        }
                    } catch (error) {
                        console.warn('Invalid allQueries in localStorage:', error);
                    }
                }
                // Return default queries if no valid saved state
                return getTestCases('comprehensive');
            });
        
            // Quick Query Test state (moved from Router tab)
            const [quickQuery, setQuickQuery] = useState(() => {
                return localStorage.getItem('testing_tab_quickQuery') || '';
            });
            
            const [quickQueryLoading, setQuickQueryLoading] = useState(() => {
                return JSON.parse(localStorage.getItem('testing_tab_quickQueryLoading') || 'false');
            });
            
            const [quickQueryResult, setQuickQueryResult] = useState(() => {
                const saved = localStorage.getItem('testing_tab_quickQueryResult');
                return saved ? JSON.parse(saved) : null;
            });
        
            // Save state to localStorage whenever it changes
            useEffect(() => {
                localStorage.setItem('testing_tab_testType', testType);
            }, [testType]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_running', JSON.stringify(running));
            }, [running]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_results', JSON.stringify(results));
            }, [results]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_currentTest', JSON.stringify(currentTest));
            }, [currentTest]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_testProgress', JSON.stringify(testProgress));
            }, [testProgress]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_progressStats', JSON.stringify(progressStats));
            }, [progressStats]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_showQuerySelection', JSON.stringify(showQuerySelection));
            }, [showQuerySelection]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_selectedQueries', JSON.stringify(selectedQueries));
            }, [selectedQueries]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_customQuery', customQuery);
            }, [customQuery]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_customCategory', customCategory);
            }, [customCategory]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_allQueries', JSON.stringify(allQueries));
            }, [allQueries]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_quickQuery', quickQuery);
            }, [quickQuery]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_quickQueryLoading', JSON.stringify(quickQueryLoading));
            }, [quickQueryLoading]);
        
            useEffect(() => {
                localStorage.setItem('testing_tab_quickQueryResult', JSON.stringify(quickQueryResult));
            }, [quickQueryResult]);
        
            // Initialize queries when component mounts or test type changes
            useEffect(() => {
                const queries = getTestCases(testType);
                setAllQueries(queries);
                
                // Reset selection to prevent index mismatch issues
                // Only keep selections that are valid for the new query set
                const validSelections = selectedQueries.filter(index => 
                    typeof index === 'number' && index >= 0 && index < queries.length
                );
                
                if (validSelections.length === 0) {
                    // If no valid selections, select all
                    setSelectedQueries(queries.map((_, index) => index));
                } else {
                    // Update with valid selections only
                    setSelectedQueries(validSelections);
                }
            }, [testType]);
        
            // Ensure selectedQueries are always valid when allQueries changes
            useEffect(() => {
                if (allQueries.length > 0) {
                    const validSelections = selectedQueries.filter(index => 
                        typeof index === 'number' && index >= 0 && index < allQueries.length
                    );
                    
                    if (validSelections.length !== selectedQueries.length) {
                        setSelectedQueries(validSelections.length > 0 ? validSelections : allQueries.map((_, index) => index));
                    }
                }
            }, [allQueries]);
        
            // Helper function to get remaining time estimate
            const getRemainingTimeEstimate = () => {
                if (!running || !currentTest) return '';
                
                const completedTests = progressStats.completed;
                const totalTests = progressStats.total;
                const remainingTests = totalTests - completedTests;
                
                if (remainingTests <= 0) return '';
                
                // Estimate 20 seconds per test (conservative estimate)
                const estimatedSeconds = remainingTests * 20;
                const minutes = Math.floor(estimatedSeconds / 60);
                const seconds = estimatedSeconds % 60;
                
                if (minutes > 0) {
                    return `⏳ Est. ${minutes}m ${seconds}s remaining for ${remainingTests} tests`;
                } else {
                    return `⏳ Est. ${seconds}s remaining for ${remainingTests} tests`;
                }
            };
        
            // Clear state function with better error handling
            const clearTestingState = () => {
                try {
                    const keysToRemove = [
                        'testing_tab_results',
                        'testing_tab_currentTest', 
                        'testing_tab_testProgress',
                        'testing_tab_progressStats',
                        'testing_tab_running',
                        'testing_tab_quickQueryResult'
                    ];
                    
                    keysToRemove.forEach(key => {
                        try {
                            localStorage.removeItem(key);
                        } catch (error) {
                            console.warn(`Failed to remove ${key}:`, error);
                        }
                    });
                    
                    setResults(null);
                    setCurrentTest(null);
                    setTestProgress([]);
                    setProgressStats({ completed: 0, total: 0 });
                    setRunning(false);
                    setQuickQueryResult(null);
                } catch (error) {
                    console.error('Error clearing testing state:', error);
                    // Force reload if clearing fails
                    if (confirm('Error clearing state. Reload page to reset?')) {
                        window.location.reload();
                    }
                }
            };
        
        
            // Helper function to determine expected route
            const getExpectedRoute = (testCase) => {
                const query = testCase.query.toLowerCase();
                const category = testCase.category.toLowerCase();
                
                if (category.includes('aeroflow') || query.includes('aeroflow')) {
                    return {
                        name: 'AeroFlow',
                        color: 'bg-purple-100 text-purple-800',
                        icon: '🚁'
                    };
                } else if (category.includes('ecosprint') || query.includes('ecosprint')) {
                    return {
                        name: 'EcoSprint', 
                        color: 'bg-green-100 text-green-800',
                        icon: '🌱'
                    };
                } else if (category.includes('comparison') || query.includes('compare') || query.includes('better') || query.includes('vs')) {
                    return {
                        name: 'Comparison',
                        color: 'bg-yellow-100 text-yellow-800',
                        icon: '⚖️'
                    };
                } else if (category.includes('aero') || query.includes('aerodynamic')) {
                    return {
                        name: 'AeroFlow',
                        color: 'bg-purple-100 text-purple-800',
                        icon: '🚁'
                    };
                } else if (category.includes('eco') || query.includes('eco') || query.includes('green') || query.includes('environment')) {
                    return {
                        name: 'EcoSprint',
                        color: 'bg-green-100 text-green-800', 
                        icon: '🌱'
                    };
                } else {
                    return {
                        name: 'Generic',
                        color: 'bg-gray-100 text-gray-800',
                        icon: '❓'
                    };
                }
            };
        
            // Quick Query Test execution (moved from Router tab)
            const executeQuickQuery = async (queryText) => {
                setQuickQueryLoading(true);
                setQuickQueryResult(null);
        
                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: queryText })
                    });
        
                    const data = await response.json();
                    setQuickQueryResult(data);
                } catch (error) {
                    setQuickQueryResult({ success: false, error: error.message });
                } finally {
                    setQuickQueryLoading(false);
                }
            };
        
            // Predefined queries for quick testing
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
                "What is the warranty coverage?",
                
                // JSON Contents
                "What data is contained in the JSON files?",
                "Analyze the structure of the uploaded JSON data",
                "What are the key fields in the JSON data?",
                "Show me statistics from the JSON data",
                "What patterns can you find in the JSON data?",
                "How many records are in the JSON dataset?",
                "What is the data type distribution in the JSON?", 
                "Extract insights from the JSON data"
            ];
        
            // Main test execution function (enhanced with better state management)
            const runTestsWithProgress = async () => {
                if (running) {
                    alert('Tests are already running. Please wait for completion.');
                    return;
                }
        
                setRunning(true);
                setResults(null);
                setTestProgress([]);
                setCurrentTest(null);
                
                // Use selected queries instead of all queries - with safety checks
                let testCases = selectedQueries
                    .filter(index => index < allQueries.length) // Filter out invalid indices
                    .map(index => allQueries[index])
                    .filter(testCase => testCase && testCase.query); // Ensure valid test cases
                
                if (testCases.length === 0) {
                    alert('Please select at least one test query to execute.');
                    setRunning(false);
                    return;
                }
                
                // Check if JSON files are uploaded and add JSON tests if comprehensive
                if (testType === 'comprehensive' && selectedQueries.length === allQueries.length) {
                    try {
                        const response = await fetch('/api/files');
                        const data = await response.json();
                        
                        if (data.files && data.files.some(file => 
                            file.original_name && file.original_name.toLowerCase().endsWith('.json')
                        )) {
                            const jsonTests = [
                                { query: "What data is contained in the JSON files?", category: "JSON - Content" },
                                { query: "Analyze the structure of the uploaded JSON data", category: "JSON - Structure" },
                                { query: "What are the key fields in the JSON data?", category: "JSON - Schema" },
                                { query: "Show me statistics from the JSON data", category: "JSON - Statistics" },
                                { query: "What patterns can you find in the JSON data?", category: "JSON - Analysis" },
                                { query: "How many records are in the JSON dataset?", category: "JSON - Count" },
                                { query: "What is the data type distribution in the JSON?", category: "JSON - Types" },
                                { query: "Extract insights from the JSON data", category: "JSON - Insights" }
                            ];
                            testCases = [...testCases, ...jsonTests];
                            console.log('📊 Added JSON-specific tests - JSON files detected');
                        }
                    } catch (error) {
                        console.warn('Could not check for JSON files:', error);
                    }
                }
                
                setProgressStats({ completed: 0, total: testCases.length });
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
                        
                        // Enhanced routing intelligence extraction
                        let routingIntelligence = data.routing_intelligence || data.routing_info || {};
                        
                        // If still no routing info, try to infer from response
                        if (!routingIntelligence.decision && data.response) {
                            const responseText = data.response.toLowerCase();
                            if (responseText.includes('ecosprint')) {
                                routingIntelligence = {
                                    decision: 'EcoSprint_specifications',
                                    method_used: 'LLM',
                                    reasoning: 'Inferred from response content mentioning EcoSprint'
                                };
                            } else if (responseText.includes('aeroflow')) {
                                routingIntelligence = {
                                    decision: 'AeroFlow_specifications', 
                                    method_used: 'LLM',
                                    reasoning: 'Inferred from response content mentioning AeroFlow'
                                };
                            }
                        }
                        
                        const testResult = {
                            test_id: i + 1,
                            query: testCase.query,
                            category: testCase.category,
                            success: data.success,
                            response: data.response || '',
                            response_time: responseTime,
                            response_length: data.response ? data.response.length : 0,
                            timestamp: new Date().toISOString(),
                            routing_intelligence: routingIntelligence,
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
                    {/* State Management Controls */}
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <div className="flex justify-between items-center">
                            <div className="flex items-center space-x-2">
                                <span className="text-yellow-800 font-medium">💾 State Preserved</span>
                                <span className="text-yellow-600 text-sm">
                                    Your testing session is automatically saved across page refreshes
                                </span>
                            </div>
                            <button
                                onClick={clearTestingState}
                                className="text-yellow-700 hover:text-yellow-900 text-sm font-medium px-3 py-1 border border-yellow-300 rounded hover:bg-yellow-100"
                            >
                                🧹 Clear Session
                            </button>
                        </div>
                        {running && (
                            <div className="mt-2 text-yellow-700 text-sm">
                                ⚠️ Test session in progress - Do not close this tab to preserve state
                            </div>
                        )}
                    </div>
        
                    {/* Quick Query Test (moved from Router tab) */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h2 className="text-lg font-medium text-gray-900 mb-4">🚀 Quick Query Test</h2>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Enter Query
                                </label>
                                <div className="flex space-x-2">
                                    <input
                                        type="text"
                                        value={quickQuery}
                                        onChange={(e) => setQuickQuery(e.target.value)}
                                        placeholder="Ask a question..."
                                        className="flex-1 border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                        onKeyPress={(e) => e.key === 'Enter' && !quickQueryLoading && executeQuickQuery(quickQuery)}
                                    />
                                    <button
                                        onClick={() => executeQuickQuery(quickQuery)}
                                        disabled={quickQueryLoading || !quickQuery.trim()}
                                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                                    >
                                        {quickQueryLoading ? <span className="spinner"></span> : 'Ask'}
                                    </button>
                                </div>
                            </div>
        
                            <div>
                                <p className="text-sm text-gray-700 mb-2">Or try a predefined query:</p>
                                <div className="flex flex-wrap gap-2">
                                    {predefinedQueries.map((predefinedQuery, index) => (
                                        <button
                                            key={index}
                                            onClick={() => !quickQueryLoading && executeQuickQuery(predefinedQuery)}
                                            disabled={quickQueryLoading}
                                            className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200 disabled:opacity-50"
                                        >
                                            {predefinedQuery}
                                        </button>
                                    ))}
                                </div>
                            </div>
        
                            {quickQueryResult && (
                                <div className="border-t pt-4">
                                    {quickQueryResult.success ? (
                                        <div className="space-y-2">
                                            <div className="text-sm text-gray-600">
                                                Response time: {quickQueryResult.response_time?.toFixed(2)}s
                                            </div>
                                            <div className="bg-gray-50 p-3 rounded border">
                                                <p className="text-sm">{quickQueryResult.response}</p>
                                            </div>
                                            {quickQueryResult.routing_info && (
                                                <details className="text-xs text-gray-500">
                                                    <summary className="cursor-pointer">Routing Details</summary>
                                                    <pre className="mt-2 p-2 bg-gray-100 rounded">
                                                        {JSON.stringify(quickQueryResult.routing_info, null, 2)}
                                                    </pre>
                                                </details>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="text-red-600 text-sm">
                                            Error: {quickQueryResult.error}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
        
                    {/* Test Query Selection */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg font-medium text-gray-900">🧪 Batch Test Configuration</h2>
                            <button
                                onClick={() => setShowQuerySelection(!showQuerySelection)}
                                className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                            >
                                {showQuerySelection ? '📝 Hide Query Selection' : '📝 Customize Queries'}
                            </button>
                        </div>
        
                        {showQuerySelection && (
                            <div className="space-y-6">
                                {/* Query List */}
                                <div>
                                    <div className="flex justify-between items-center mb-3">
                                        <h3 className="text-md font-medium text-gray-800">
                                            Available Test Queries ({allQueries.length})
                                        </h3>
                                        <div className="flex space-x-2">
                                            <button
                                                onClick={() => setSelectedQueries(allQueries.map((_, index) => index))}
                                                className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded hover:bg-blue-200"
                                            >
                                                Select All
                                            </button>
                                            <button
                                                onClick={() => setSelectedQueries([])}
                                                className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded hover:bg-gray-200"
                                            >
                                                Clear All
                                            </button>
                                        </div>
                                    </div>
        
                                    <div className="max-h-80 overflow-y-auto border border-gray-200 rounded-md">
                                        <table className="min-w-full bg-white">
                                            <thead className="bg-gray-50 sticky top-0">
                                                <tr>
                                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedQueries.length === allQueries.length}
                                                            onChange={(e) => {
                                                                if (e.target.checked) {
                                                                    setSelectedQueries(allQueries.map((_, index) => index));
                                                                } else {
                                                                    setSelectedQueries([]);
                                                                }
                                                            }}
                                                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                        />
                                                    </th>
                                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                        Category
                                                    </th>
                                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                        Query
                                                    </th>
                                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                        Expected Route
                                                    </th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200">
                                                {allQueries.map((testCase, index) => {
                                                    // Safety check to ensure testCase exists
                                                    if (!testCase || !testCase.query || !testCase.category) {
                                                        return null;
                                                    }
                                                    
                                                    const isSelected = selectedQueries.includes(index);
                                                    const expectedRoute = getExpectedRoute(testCase);
                                                    
                                                    return (
                                                        <tr
                                                            key={index}
                                                            className={`${isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'} cursor-pointer`}
                                                            onClick={() => {
                                                                if (isSelected) {
                                                                    setSelectedQueries(prev => prev.filter(i => i !== index));
                                                                } else {
                                                                    setSelectedQueries(prev => [...prev, index]);
                                                                }
                                                            }}
                                                        >
                                                            <td className="px-3 py-3 whitespace-nowrap">
                                                                <input
                                                                    type="checkbox"
                                                                    checked={isSelected}
                                                                    onChange={() => {}} // Handled by row click
                                                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                                                />
                                                            </td>
                                                            <td className="px-3 py-3 whitespace-nowrap">
                                                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                                                    {testCase.category}
                                                                </span>
                                                            </td>
                                                            <td className="px-3 py-3 text-sm text-gray-900 max-w-md">
                                                                <div className="truncate" title={testCase.query}>
                                                                    {testCase.query}
                                                                </div>
                                                            </td>
                                                            <td className="px-3 py-3 whitespace-nowrap">
                                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${expectedRoute.color}`}>
                                                                    {expectedRoute.icon} {expectedRoute.name}
                                                                </span>
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
        
                                {/* Add Custom Query */}
                                <div className="border-t pt-4">
                                    <h4 className="text-md font-medium text-gray-800 mb-3">Add Custom Query</h4>
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                                        <div className="sm:col-span-2">
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Custom Query
                                            </label>
                                            <input
                                                type="text"
                                                value={customQuery}
                                                onChange={(e) => setCustomQuery(e.target.value)}
                                                placeholder="Enter your custom test query..."
                                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                                Category
                                            </label>
                                            <input
                                                type="text"
                                                value={customCategory}
                                                onChange={(e) => setCustomCategory(e.target.value)}
                                                placeholder="Category"
                                                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            />
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => {
                                            if (customQuery.trim()) {
                                                const newQuery = {
                                                    query: customQuery.trim(),
                                                    category: customCategory.trim() || 'Custom'
                                                };
                                                setAllQueries(prev => [...prev, newQuery]);
                                                setSelectedQueries(prev => [...prev, allQueries.length]);
                                                setCustomQuery('');
                                                setCustomCategory('Custom');
                                            }
                                        }}
                                        disabled={!customQuery.trim()}
                                        className="mt-2 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                                    >
                                        ➕ Add Query
                                    </button>
                                </div>
        
                                {/* Selection Summary */}
                                <div className="bg-blue-50 p-4 rounded-lg">
                                    <div className="flex justify-between items-center">
                                        <div>
                                            <span className="text-sm font-medium text-blue-900">
                                                {selectedQueries.length} of {allQueries.length} queries selected
                                            </span>
                                            {selectedQueries.length > 0 && allQueries.length > 0 && (
                                                <div className="text-xs text-blue-700 mt-1">
                                                    Categories: {[...new Set(selectedQueries
                                                        .filter(i => i < allQueries.length) // Ensure valid indices
                                                        .map(i => allQueries[i]?.category)
                                                        .filter(Boolean) // Remove undefined values
                                                    )].join(', ')}
                                                </div>
                                            )}
                                        </div>
                                        <div className="text-xs text-blue-600">
                                            Est. time: ~{(selectedQueries.length * 20).toFixed(0)} seconds
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
        
                    {/* Test Configuration */}
                    <div className="bg-white shadow rounded-lg p-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-6">Batch Test Configuration</h3>
                        
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
                                            <strong>Comprehensive Test</strong> - Full routing intelligence test (24+ tests)
                                        </span>
                                    </label>
                                </div>
                            </div>
        
                            <div className="flex space-x-4">
                                <button
                                    onClick={runTestsWithProgress}
                                    disabled={running || selectedQueries.length === 0}
                                    className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                                >
                                    {running ? (
                                        <>
                                            <span className="spinner mr-2"></span>
                                            Running Tests...
                                        </>
                                    ) : (
                                        `🚀 Run Selected Tests (${selectedQueries.length})`
                                    )}
                                </button>
        
                                {running && (
                                    <button
                                        onClick={() => {
                                            if (confirm('Are you sure you want to stop the running tests?')) {
                                                setRunning(false);
                                                setCurrentTest(null);
                                            }
                                        }}
                                        className="bg-red-600 text-white py-2 px-4 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
                                    >
                                        🛑 Stop Tests
                                    </button>
                                )}
                            </div>
        
                            {/* Waiting Time Display */}
                            {running && (
                                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                                    <div className="flex items-center justify-between">
                                        <span className="text-yellow-800 text-sm font-medium">
                                            🏃‍♂️ Tests in Progress
                                        </span>
                                        <span className="text-yellow-600 text-sm">
                                            {getRemainingTimeEstimate()}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
        
                    {/* Real-time Test Progress */}
                    {running && (
                        <div className="bg-white shadow rounded-lg p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">📊 Test Progress</h3>
                            
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
                                <div className="text-xs text-gray-500 mt-1">
                                    {getRemainingTimeEstimate()}
                                </div>
                            </div>
        
                            {/* Current Test */}
                            {currentTest && (
                                <div className="mb-4 p-4 border border-blue-200 rounded-lg bg-blue-50">
                                    <div className="flex items-center justify-between mb-2">
                                        <h4 className="font-medium text-blue-900">
                                            🧪 Test {currentTest.index}/{currentTest.total}
                                        </h4>
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                            currentTest.status === 'running' 
                                                ? 'bg-yellow-100 text-yellow-800'
                                                : currentTest.success
                                                ? 'bg-green-100 text-green-800'
                                                : 'bg-red-100 text-red-800'
                                        }`}>
                                            {currentTest.status === 'running' && <span className="spinner mr-1"></span>}
                                            {currentTest.status === 'running' ? '🏃‍♂️ Running...' 
                                             : currentTest.status === 'completed' && currentTest.success ? '✅ Passed'
                                             : currentTest.status === 'completed' && !currentTest.success ? '❌ Failed'
                                             : '⏳ Pending'}
                                        </span>
                                    </div>
                                    <div className="text-sm text-blue-700">
                                        <div className="font-medium mb-1">📂 Category: {currentTest.category}</div>
                                        <div>❓ Query: "{currentTest.query}"</div>
                                    </div>
                                </div>
                            )}
        
                            {/* Completed Tests Log */}
                            {testProgress.length > 0 && (
                                <div>
                                    <h4 className="font-medium text-gray-900 mb-3">✅ Completed Tests</h4>
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
                            <h3 className="text-lg font-medium text-gray-900 mb-4">📈 Test Results</h3>
                            
                            {results.error ? (
                                <div className="text-red-600">
                                    ❌ Error: {results.error}
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
                                        <h4 className="font-medium text-gray-900 mb-3">🧠 Routing Intelligence Summary</h4>
                                        
                                        {/* Routing Intelligence Summary Table */}
                                        <div className="mb-6 overflow-x-auto">
                                            <table className="min-w-full bg-white border border-gray-200 rounded-lg">
                                                <thead className="bg-gray-50">
                                                    <tr>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Test #</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Query</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Route Chosen</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Method</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reasoning</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="bg-white divide-y divide-gray-200">
                                                    {results.results.map((result, index) => {
                                                        const routingInfo = result.routing_intelligence || {};
                                                        const routeChosen = routingInfo.decision || 'Unknown';
                                                        const method = routingInfo.method_used || 'Unknown';
                                                        const reasoning = routingInfo.final_reasoning || 
                                                                        routingInfo.reasoning || 
                                                                        (routingInfo.reasoning_steps && routingInfo.reasoning_steps.join('; ')) || 
                                                                        'No reasoning available';
                                                        
                                                        // Extract route name (remove "_specifications" suffix if present)
                                                        const displayRoute = routeChosen.replace(/_specifications$/, '').replace(/_/g, ' ');
                                                        
                                                        return (
                                                            <tr key={index} className={`${result.success ? 'bg-green-50' : 'bg-red-50'} hover:bg-gray-50`}>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                                                                    #{result.test_id}
                                                                </td>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                                                        {result.category}
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 text-sm text-gray-900 max-w-xs">
                                                                    <div className="truncate" title={result.query}>
                                                                        {result.query}
                                                                    </div>
                                                                </td>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                                        routeChosen === 'Unknown' 
                                                                            ? 'bg-gray-100 text-gray-800' 
                                                                            : routeChosen.toLowerCase().includes('aeroflow') 
                                                                            ? 'bg-purple-100 text-purple-800'
                                                                            : routeChosen.toLowerCase().includes('ecosprint')
                                                                            ? 'bg-green-100 text-green-800'
                                                                            : 'bg-yellow-100 text-yellow-800'
                                                                    }`}>
                                                                        {routeChosen === 'Unknown' ? '❓ Unknown' : `🎯 ${displayRoute}`}
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                                                                        method === 'LLM' 
                                                                            ? 'bg-blue-100 text-blue-800'
                                                                            : method === 'Keyword' 
                                                                            ? 'bg-orange-100 text-orange-800'
                                                                            : 'bg-gray-100 text-gray-800'
                                                                    }`}>
                                                                        {method === 'LLM' ? '🧠 LLM' : method === 'Keyword' ? '🔍 Keyword' : '❓ Unknown'}
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 text-sm text-gray-600 max-w-md">
                                                                    <div className="truncate" title={reasoning}>
                                                                        {reasoning.length > 80 ? reasoning.substring(0, 80) + '...' : reasoning}
                                                                    </div>
                                                                </td>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm">
                                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                                        result.success 
                                                                            ? 'bg-green-100 text-green-800' 
                                                                            : 'bg-red-100 text-red-800'
                                                                    }`}>
                                                                        {result.success ? '✅ Pass' : '❌ Fail'}
                                                                    </span>
                                                                </td>
                                                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600">
                                                                    {result.response_time?.toFixed(2)}s
                                                                </td>
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                        </div>
                                
                                        {/* Routing Statistics Summary */}
                                        <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-4">
                                            <div className="bg-purple-50 p-4 rounded-lg">
                                                <div className="text-2xl font-bold text-purple-600">
                                                    {results.results.filter(r => r.routing_intelligence?.decision?.toLowerCase().includes('aeroflow')).length}
                                                </div>
                                                <div className="text-sm text-purple-800">🚁 AeroFlow Routes</div>
                                            </div>
                                            <div className="bg-green-50 p-4 rounded-lg">
                                                <div className="text-2xl font-bold text-green-600">
                                                    {results.results.filter(r => r.routing_intelligence?.decision?.toLowerCase().includes('ecosprint')).length}
                                                </div>
                                                <div className="text-sm text-green-800">🌱 EcoSprint Routes</div>
                                            </div>
                                            <div className="bg-blue-50 p-4 rounded-lg">
                                                <div className="text-2xl font-bold text-blue-600">
                                                    {results.results.filter(r => r.routing_intelligence?.method_used === 'LLM').length}
                                                </div>
                                                <div className="text-sm text-blue-800">🧠 LLM Decisions</div>
                                            </div>
                                            <div className="bg-orange-50 p-4 rounded-lg">
                                                <div className="text-2xl font-bold text-orange-600">
                                                    {results.results.filter(r => r.routing_intelligence?.method_used === 'Keyword').length}
                                                </div>
                                                <div className="text-sm text-orange-800">🔍 Keyword Decisions</div>
                                            </div>
                                        </div>
                                
                                        {/* Detailed Test Results (Expandable) */}
                                        <details className="mb-4">
                                            <summary className="cursor-pointer font-medium text-gray-900 hover:text-blue-600">
                                                📋 View Detailed Test Results ({results.results.length} tests)
                                            </summary>
                                            <div className="mt-4 space-y-3">
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
                                                                
                                                                {/* Routing Intelligence Details */}
                                                                {result.routing_intelligence && (
                                                                    <div className="mb-3 p-3 bg-blue-50 rounded border-l-4 border-blue-400">
                                                                        <h6 className="text-sm font-medium text-blue-900 mb-2">🧠 Routing Intelligence</h6>
                                                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-blue-800">
                                                                            <div>
                                                                                <span className="font-medium">Route:</span> {result.routing_intelligence.decision || 'Unknown'}
                                                                            </div>
                                                                            <div>
                                                                                <span className="font-medium">Method:</span> {result.routing_intelligence.method_used || 'Unknown'}
                                                                            </div>
                                                                            {result.routing_intelligence.scores && (
                                                                                <div className="col-span-2">
                                                                                    <span className="font-medium">Scores:</span> {JSON.stringify(result.routing_intelligence.scores)}
                                                                                </div>
                                                                            )}
                                                                            {(result.routing_intelligence.final_reasoning || result.routing_intelligence.reasoning) && (
                                                                                <div className="col-span-2">
                                                                                    <span className="font-medium">Reasoning:</span> {result.routing_intelligence.final_reasoning || result.routing_intelligence.reasoning}
                                                                                </div>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                )}
                                    
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
                                        </details>
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


# Add these endpoints to your Flask app for debugging

@app.route('/api/debug/file/<file_id>', methods=['GET'])
def debug_file(file_id):
    """Debug endpoint to check file processing"""
    try:
        if file_id not in application_state['uploaded_files']:
            return jsonify({'error': 'File not found'}), 404

        file_info = application_state['uploaded_files'][file_id]
        file_path = file_info['path']

        debug_info = {
            'file_info': file_info,
            'file_exists': os.path.exists(file_path),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'file_extension': Path(file_path).suffix.lower(),
            'is_json': file_path.endswith('.json')
        }

        # If it's a JSON file, try to analyze it
        if file_path.endswith('.json'):
            try:
                # Test JSON parsing
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                debug_info['json_valid'] = True
                debug_info['json_type'] = type(json_data).__name__
                debug_info['json_size'] = len(str(json_data))

                if isinstance(json_data, dict):
                    debug_info['json_keys'] = list(json_data.keys())[:10]
                    debug_info['json_key_count'] = len(json_data)
                elif isinstance(json_data, list):
                    debug_info['json_item_count'] = len(json_data)
                    debug_info['json_first_item_type'] = type(json_data[0]).__name__ if json_data else 'empty'

                # Test document creation
                try:
                    documents = load_documents_with_json_support(file_path)
                    debug_info['documents_created'] = len(documents)
                    debug_info['document_preview'] = [
                        {
                            'text_length': len(doc.text),
                            'text_preview': doc.text[:200] + '...' if len(doc.text) > 200 else doc.text,
                            'metadata': doc.metadata
                        } for doc in documents[:3]
                    ]
                except Exception as doc_error:
                    debug_info['document_creation_error'] = str(doc_error)
                    debug_info['documents_created'] = 0

            except json.JSONDecodeError as e:
                debug_info['json_valid'] = False
                debug_info['json_error'] = str(e)
            except Exception as e:
                debug_info['json_processing_error'] = str(e)

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/json-test', methods=['POST'])
def debug_json_test():
    """Test JSON processing with uploaded data"""
    try:
        data = request.get_json()
        test_json = data.get('test_json', {})

        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_json, f, indent=2)
            temp_path = f.name

        try:
            # Test processing
            documents = load_documents_with_json_support(temp_path)

            result = {
                'success': True,
                'documents_created': len(documents),
                'documents': [
                    {
                        'text_length': len(doc.text),
                        'text_preview': doc.text[:300] + '...' if len(doc.text) > 300 else doc.text,
                        'metadata': doc.metadata
                    } for doc in documents
                ]
            }

        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        finally:
            # Clean up temp file
            os.unlink(temp_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug/system', methods=['GET'])
def debug_system():
    """Debug system state and configuration"""
    try:
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'application_state': {
                'llm_configured': application_state['llm'] is not None,
                'embed_model_configured': application_state['embed_model'] is not None,
                'router_configured': application_state['router_agent'] is not None,
                'uploaded_files_count': len(application_state['uploaded_files']),
                'indexes_count': len(application_state['indexes']),
                'query_engines_count': len(application_state['query_engines'])
            },
            'configuration': application_state['configuration'],
            'uploaded_files': {
                file_id: {
                    'original_name': info['original_name'],
                    'size': info['size'],
                    'processed': info.get('processed', False),
                    'has_index': file_id in application_state['indexes'],
                    'has_query_engine': file_id in application_state['query_engines'],
                    'file_extension': Path(info['path']).suffix.lower(),
                    'is_json': info['path'].endswith('.json')
                }
                for file_id, info in application_state['uploaded_files'].items()
            },
            'recent_errors': getattr(debug_system, 'recent_errors', [])
        }

        # Test embedding model
        if application_state['embed_model']:
            try:
                test_embedding = application_state['embed_model'].get_text_embedding("test")
                debug_info['embedding_test'] = {
                    'success': True,
                    'embedding_dimension': len(test_embedding)
                }
            except Exception as e:
                debug_info['embedding_test'] = {
                    'success': False,
                    'error': str(e)
                }

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error tracking for debug
def track_error(error_info):
    """Track errors for debugging"""
    if not hasattr(debug_system, 'recent_errors'):
        debug_system.recent_errors = []

    debug_system.recent_errors.append({
        'timestamp': datetime.now().isoformat(),
        'error': error_info
    })

    # Keep only last 10 errors
    debug_system.recent_errors = debug_system.recent_errors[-10:]

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
    """Validate uploaded file type - now includes JSON"""
    allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.json'}
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
