#!/usr/bin/env python3
"""
Customer Service AI Agent with Database Integration and Enhanced Tool Usage Tracking

INTEGRATION CHECKLIST:

✅ 1. Import the call graph visualizer module
✅ 2. Modify CustomerServiceAgent.__init__ to include execution tracker
✅ 3. Wrap tools with TrackedCustomerServiceTools in create_tools()
    ✅ 4. Replace run_query() with tracked version
✅ 5. Replace run_intelligent_query() with tracked version
✅ 6. Add call graph viewer methods
✅ 7. Update main menu to include call graph options
✅ 8. Enhance custom query method with tracking info

WHAT GETS TRACKED:
• Query start and complexity assessment
• Agent thinking processes
• Tool selection reasoning
• Database operations (with timing)
• Cache hits and misses
• Logical inference steps
• Error handling and recovery
• Result synthesis
• Final response generation

OUTPUT:
• Interactive HTML files in ./generated_callgraphs/
• Each query creates one complete call graph
• Graphs show execution flow with nodes and edges
    • Clickable nodes show detailed information
• Graphs can be opened in browser for interaction

USAGE:
1. Run any query (predefined scenarios or custom)
2. Query execution is automatically tracked
3. Call graph HTML file is generated
4. Use menu option 10 to view graphs
5. Use menu option 11 for statistics

"""

import functools
import subprocess
import sys
import os
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hashlib
import json


from query_callgraph_visualizer_v01 import (
    execution_tracker,
    EnhancedQueryExecutionTracker,
    TrackedCustomerServiceTools,
    integrate_with_agent,
    NodeType
)


load_dotenv()  # Load .env file if it exists


def performance_monitor(func):
    """Decorator to monitor function performance"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__}: {execution_time:.2f}s")
        return result

    return wrapper


def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_dependencies():
    """Check and install required packages"""
    print("🔍 Checking required dependencies...")

    required_packages = {
        'mysql-connector-python': 'mysql.connector',
        'llama-index==0.10.59': 'llama_index',
        'llama-index-llms-openai-like': 'llama_index.llms.openai_like',
        'llama-index-embeddings-huggingface': 'llama_index.embeddings.huggingface',
        'sentence-transformers': 'sentence_transformers',
        'nest-asyncio': 'nest_asyncio',
        'psutil': 'psutil',
        'python-dotenv': 'dotenv'  # ADD THIS LINE
    }

    missing_packages = []

    for package, module in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package.split('==')[0]} - Found")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package.split('==')[0]} - Missing")

    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing package(s)...")

        for package in missing_packages:
            print(f"\n🔧 Installing {package}...")
            if install_package(package):
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
                print(f"Please install manually: pip install {package}")
                return False

        print("\n🎉 All packages installed successfully!")
        print("Please restart the script to use the newly installed packages.")
        return False
    else:
        print("✅ All required packages are available!")
        return True


# Check dependencies before importing anything else
if not check_and_install_dependencies():
    print("\n🔄 Please restart the script after package installation.")
    sys.exit(0)

# Now import the packages after ensuring they're installed
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any

# LlamaIndex imports
try:
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import VectorStoreIndex
    from llama_index.core.tools import QueryEngineTool, FunctionTool
    from llama_index.core.agent import ReActAgentWorker, AgentRunner
    import nest_asyncio

    nest_asyncio.apply()
    LLAMAINDEX_AVAILABLE = True
    print("✅ LlamaIndex components loaded successfully!")
except ImportError as e:
    print(f"❌ LlamaIndex import error: {e}")
    print("Some packages may not have been installed correctly.")
    LLAMAINDEX_AVAILABLE = False

# Fix tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Performance optimizations
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # Use all CPU cores
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

# Memory optimizations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable for GPU

# Cache optimizations
os.environ["HF_HUB_CACHE"] = "./hf_cache"  # Local cache
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./st_cache"

# Mac GPU acceleration setup
import torch

# Enable Metal Performance Shaders on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Mac GPU (Metal Performance Shaders)")
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use all available GPU memory
else:
    device = torch.device("cpu")
    print("💻 Using CPU only")

# Set device for sentence transformers
os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = str(device)

# ----------------------------------- DataClass-------------------------------#
# Enhanced imports for optimizations (ADD AFTER LINE 7)
import gc
import logging
from dataclasses import dataclass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

    PSUTIL_AVAILABLE = True


# Configuration Management (ADD AFTER IMPORTS)
@dataclass
class Config:
    """Comprehensive environment-based configuration"""

    # Database Configuration
    database_host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    database_user: str = field(default_factory=lambda: os.getenv('DB_USER', 'root'))
    database_password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', 'auburn'))
    database_name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'customer_service_db'))
    database_pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '5')))

    # LLM Configuration
    llm_url: str = field(default_factory=lambda: os.getenv('LLM_URL', 'http://127.0.0.1:1234/v1'))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'open_gpt4_8x7b_v0.2'))
    llm_api_key: str = field(default_factory=lambda: os.getenv('LLM_API_KEY', 'lm-studio'))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv('LLM_TEMPERATURE', '0.1')))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv('LLM_MAX_TOKENS', '3000')))
    llm_timeout: int = field(default_factory=lambda: int(os.getenv('LLM_TIMEOUT', '45')))
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv('LLM_MAX_RETRIES', '2')))

    # Embedding Model Configuration
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    embedding_max_length: int = field(default_factory=lambda: int(os.getenv('EMBEDDING_MAX_LENGTH', '512')))
    embedding_device: str = field(default_factory=lambda: os.getenv('EMBEDDING_DEVICE', 'auto'))
    embedding_trust_remote_code: bool = field(default_factory=lambda: os.getenv('EMBEDDING_TRUST_REMOTE_CODE', 'true').lower() == 'true')

    # Performance Configuration
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    max_iterations: int = field(default_factory=lambda: int(os.getenv('MAX_ITERATIONS', '15')))
    query_timeout: int = field(default_factory=lambda: int(os.getenv('QUERY_TIMEOUT', '60')))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '300')))
    chunk_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '1024')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '50')))

    # Hardware Optimization
    use_gpu: bool = field(default_factory=lambda: os.getenv('USE_GPU', 'true').lower() == 'true')
    pytorch_mps_ratio: str = field(default_factory=lambda: os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0'))
    tokenizers_parallelism: bool = field(default_factory=lambda: os.getenv('TOKENIZERS_PARALLELISM', 'true').lower() == 'true')

    # Cache Directories
    hf_hub_cache: str = field(default_factory=lambda: os.getenv('HF_HUB_CACHE', '../src/hf_cache'))
    st_cache: str = field(default_factory=lambda: os.getenv('SENTENCE_TRANSFORMERS_HOME', '../src/st_cache'))

    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_file: str = field(default_factory=lambda: os.getenv('LOG_FILE', '../src/customer_service_agent.log'))
    log_format: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = field(default_factory=lambda: int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5')))
    circuit_breaker_recovery_timeout: int = field(default_factory=lambda: int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60')))

    # Agent Configuration
    agent_verbose: bool = field(default_factory=lambda: os.getenv('AGENT_VERBOSE', 'true').lower() == 'true')
    agent_allow_parallel_tool_calls: bool = field(default_factory=lambda: os.getenv('AGENT_ALLOW_PARALLEL_TOOL_CALLS', 'false').lower() == 'true')
    agent_memory_enabled: bool = field(default_factory=lambda: os.getenv('AGENT_MEMORY_ENABLED', 'false').lower() == 'true')

    # File Paths
    policy_files_dir: str = field(default_factory=lambda: os.getenv('POLICY_FILES_DIR', '../src/policy_files'))

    # FIXED: Use default_factory for mutable list
    support_files_default: List[str] = field(default_factory=lambda: (
        os.getenv('SUPPORT_FILES_DEFAULT', '').split(',')
        if os.getenv('SUPPORT_FILES_DEFAULT')
        else [
            'Customer Service.txt',
            'FAQ.txt',
            'Return Policy.txt',
            'Warranty Policy.txt',
            'Escalation Procedures.txt',
            'Technical Troubleshooting Guide.txt',
            'Business Policies and Procedures.txt',
            'Product Knowledge Database.txt',
            'Order Management and Fulfillment.txt'
        ]
    ))

    # Performance Monitoring
    enable_performance_monitoring: bool = field(default_factory=lambda: os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true')
    enable_memory_monitoring: bool = field(default_factory=lambda: os.getenv('ENABLE_MEMORY_MONITORING', 'true').lower() == 'true')
    memory_warning_threshold_mb: int = field(default_factory=lambda: int(os.getenv('MEMORY_WARNING_THRESHOLD_MB', '100')))
    enable_query_caching: bool = field(default_factory=lambda: os.getenv('ENABLE_QUERY_CACHING', 'true').lower() == 'true')

    # Development/Debug Settings
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG_MODE', 'false').lower() == 'true')
    profile_performance: bool = field(default_factory=lambda: os.getenv('PROFILE_PERFORMANCE', 'false').lower() == 'true')
    enable_detailed_logging: bool = field(default_factory=lambda: os.getenv('ENABLE_DETAILED_LOGGING', 'false').lower() == 'true')

    def __post_init__(self):
        """Post-initialization setup"""
        # Set environment variables for external libraries
        os.environ["OMP_NUM_THREADS"] = str(self.max_workers)
        os.environ["MKL_NUM_THREADS"] = str(self.max_workers)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.max_workers)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.max_workers)

        # Memory optimizations
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = self.pytorch_mps_ratio
        os.environ["TOKENIZERS_PARALLELISM"] = str(self.tokenizers_parallelism).lower()

        # Cache optimizations
        os.environ["HF_HUB_CACHE"] = self.hf_hub_cache
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.st_cache

        # Create cache directories if they don't exist
        os.makedirs(self.hf_hub_cache, exist_ok=True)
        os.makedirs(self.st_cache, exist_ok=True)
        os.makedirs(self.policy_files_dir, exist_ok=True)

        # Clean up support files list (remove empty strings)
        self.support_files_default = [f.strip() for f in self.support_files_default if f.strip()]


        # Set up logger after directories are created
        global logger
        if 'logger' not in globals():
            logger = setup_logging()

        logger.info(f"Configuration initialized with environment settings")
        logger.debug(f"LLM URL: {self.llm_url}, Model: {self.llm_model}")
        logger.debug(f"Database: {self.database_host}:{self.database_name}")
        logger.debug(f"Cache directories: HF={self.hf_hub_cache}, ST={self.st_cache}")
        logger.debug(f"Support files: {len(self.support_files_default)} files configured")


# ----------------------------------- Logger-------------------------------#
# Enhanced Logging Setup - MOVE THIS BEFORE Config initialization
def setup_logging():
    """Setup structured logging with file and console handlers"""
    # Use environment variables directly since config isn't created yet
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE', '../src/customer_service_agent.log')
    log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger FIRST
logger = setup_logging()

# NOW initialize configuration
config = Config()

# Log configuration after it's created
logger.info(f"Configuration initialized with environment settings")
logger.debug(f"LLM URL: {config.llm_url}, Model: {config.llm_model}")
logger.debug(f"Database: {config.database_host}:{config.database_name}")
logger.debug(f"Cache directories: HF={config.hf_hub_cache}, ST={config.st_cache}")
logger.debug(f"Support files: {len(config.support_files_default)} files configured")

# Memory monitoring decorator
def memory_monitor(func):
    """Memory monitoring decorator with garbage collection"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not PSUTIL_AVAILABLE:
            return func(*args, **kwargs)

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = mem_after - mem_before

        if memory_increase > 100:  # If memory increased by 100MB
            logger.warning(f"High memory usage in {func.__name__}: {memory_increase:.1f}MB")
            gc.collect()  # Force garbage collection
            mem_after_gc = process.memory_info().rss / 1024 / 1024
            print(f"⚠️  High memory usage: {memory_increase:.1f}MB, GC freed {mem_after - mem_after_gc:.1f}MB")

        return result

    return wrapper


# Query Result Caching (ADD AFTER memory_monitor)
def cached_query(ttl_seconds=300):
    """Decorator for caching query results with enhanced tracking"""
    def decorator(func):
        return cache_manager.cached_call(func, ttl_seconds)
    return decorator


# Enhanced Query Result Caching with detailed tracking
class CacheManager:
    """Advanced cache manager with detailed statistics and tracking"""

    def __init__(self):
        self.caches = {}  # Method-specific caches
        self.global_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cache_time_saved': 0,
            'methods_tracked': set()
        }
        self.call_history = []  # Track all calls

    def get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key"""
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def cached_call(self, func, ttl_seconds=300):
        """Enhanced caching decorator with detailed tracking"""
        func_name = func.__name__

        if func_name not in self.caches:
            self.caches[func_name] = {}

        self.global_stats['methods_tracked'].add(func_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            cache_key = self.get_cache_key(func_name, args, kwargs)
            now = datetime.now()

            # Record the call attempt
            call_record = {
                'timestamp': now,
                'function': func_name,
                'args': str(args)[:100] + ('...' if len(str(args)) > 100 else ''),
                'cache_key': cache_key[:8],  # Short version for display
                'hit': False,
                'execution_time': 0,
                'caller': self._get_caller_info()
            }

            self.global_stats['total_calls'] += 1

            # Check cache
            if cache_key in self.caches[func_name]:
                result, timestamp = self.caches[func_name][cache_key]
                if now - timestamp < timedelta(seconds=ttl_seconds):
                    # Cache hit
                    execution_time = time.time() - start_time
                    call_record['hit'] = True
                    call_record['execution_time'] = execution_time
                    call_record['source'] = 'cache'

                    self.global_stats['cache_hits'] += 1
                    self.global_stats['total_cache_time_saved'] += 0.1  # Estimated time saved
                    self.call_history.append(call_record)

                    return result
                else:
                    # Cache expired
                    del self.caches[func_name][cache_key]

            # Cache miss - execute function
            self.global_stats['cache_misses'] += 1
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Store in cache
            self.caches[func_name][cache_key] = (result, now)

            call_record['execution_time'] = execution_time
            call_record['source'] = 'database'
            self.call_history.append(call_record)

            # Cache size management
            if len(self.caches[func_name]) > 1000:
                oldest_key = min(self.caches[func_name].keys(),
                                 key=lambda k: self.caches[func_name][k][1])
                del self.caches[func_name][oldest_key]

            return result

        return wrapper

    def _get_caller_info(self) -> str:
        """Get information about who called the function"""
        import inspect

        # Look through the call stack to find the agent or query
        for frame_info in inspect.stack():
            filename = frame_info.filename
            function_name = frame_info.function

            # Skip our own frames
            if 'customer_service_agent' in filename.lower():
                if function_name in ['run_query', 'query', 'execute_decomposed_query']:
                    return f"Agent.{function_name}"
                elif 'scenario' in function_name.lower():
                    return f"Scenario.{function_name}"

        return "Unknown"

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_calls = self.global_stats['total_calls']
        hit_rate = (self.global_stats['cache_hits'] / total_calls * 100) if total_calls > 0 else 0

        # Method-specific stats
        method_stats = {}
        for method_name in self.global_stats['methods_tracked']:
            method_calls = [call for call in self.call_history if call['function'] == method_name]
            method_hits = [call for call in method_calls if call['hit']]

            method_stats[method_name] = {
                'total_calls': len(method_calls),
                'cache_hits': len(method_hits),
                'cache_misses': len(method_calls) - len(method_hits),
                'hit_rate': (len(method_hits) / len(method_calls) * 100) if method_calls else 0,
                'avg_execution_time': sum(call['execution_time'] for call in method_calls) / len(method_calls) if method_calls else 0,
                'cache_size': len(self.caches.get(method_name, {}))
            }

        return {
            'global': {
                'total_calls': total_calls,
                'cache_hits': self.global_stats['cache_hits'],
                'cache_misses': self.global_stats['cache_misses'],
                'hit_rate': hit_rate,
                'time_saved': self.global_stats['total_cache_time_saved'],
                'methods_tracked': len(self.global_stats['methods_tracked'])
            },
            'by_method': method_stats,
            'recent_calls': self.call_history[-20:] if self.call_history else []  # Last 20 calls
        }

    def clear_all_caches(self):
        """Clear all caches and reset stats"""
        self.caches.clear()
        self.call_history.clear()
        self.global_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_cache_time_saved': 0,
            'methods_tracked': set()
        }

# Global cache manager instance
cache_manager = CacheManager()


# Tool Usage Tracking System
class ToolUsageTracker:
    """Track all tool usage with detailed analytics"""

    def __init__(self):
        self.tool_calls = []
        self.tool_stats = {}
        self.session_start = datetime.now()

    def record_tool_call(self, tool_name: str, input_data: Any, result: Any,
                         execution_time: float, caller_context: str = None,
                         reasoning: str = None):
        """Record a tool call with comprehensive details"""
        call_record = {
            'timestamp': datetime.now(),
            'tool_name': tool_name,
            'input_data': str(input_data)[:200] + ('...' if len(str(input_data)) > 200 else ''),
            'result_summary': self._summarize_result(result),
            'execution_time': execution_time,
            'caller_context': caller_context or self._infer_caller_context(),
            'reasoning': reasoning or "Tool call initiated by agent",
            'success': not self._is_error_result(result),
            'result_type': type(result).__name__
        }

        self.tool_calls.append(call_record)

        # Update tool statistics
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_execution_time': 0,
                'avg_execution_time': 0,
                'first_used': call_record['timestamp'],
                'last_used': call_record['timestamp']
            }

        stats = self.tool_stats[tool_name]
        stats['total_calls'] += 1
        stats['total_execution_time'] += execution_time
        stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_calls']
        stats['last_used'] = call_record['timestamp']

        if call_record['success']:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1

    def _summarize_result(self, result: Any) -> str:
        """Create a brief summary of the result"""
        if isinstance(result, dict):
            if 'error' in result:
                return f"Error: {str(result['error'])[:100]}"
            elif 'orders' in result:
                return f"Found {len(result.get('orders', []))} orders"
            elif 'total_orders' in result:
                return f"Analysis: {result.get('total_orders', 0)} orders"
            else:
                return f"Dict with {len(result)} keys"
        elif isinstance(result, list):
            return f"List with {len(result)} items"
        elif isinstance(result, str):
            return result[:100] + ('...' if len(result) > 100 else '')
        else:
            return str(result)[:100]

    def _is_error_result(self, result: Any) -> bool:
        """Check if result indicates an error"""
        if isinstance(result, dict) and 'error' in result:
            return True
        if isinstance(result, str) and any(error_word in result.lower()
                                           for error_word in ['error', 'failed', 'not found']):
            return True
        return False

    def _infer_caller_context(self) -> str:
        """Infer the context of who called the tool"""
        import inspect

        for frame_info in inspect.stack():
            function_name = frame_info.function
            if 'scenario' in function_name.lower():
                return f"Predefined scenario: {function_name}"
            elif function_name in ['run_query', 'custom_query']:
                return "Custom user query"
            elif 'decompose' in function_name.lower():
                return "Query decomposition system"

        return "Unknown context"

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive tool usage report"""
        total_calls = len(self.tool_calls)
        successful_calls = sum(1 for call in self.tool_calls if call['success'])
        session_duration = (datetime.now() - self.session_start).total_seconds()

        # Most used tools
        tool_usage_ranking = sorted(
            [(name, stats['total_calls']) for name, stats in self.tool_stats.items()],
            key=lambda x: x[1], reverse=True
        )

        # Recent activity
        recent_calls = self.tool_calls[-10:] if self.tool_calls else []

        # Performance analysis
        performance_data = {}
        for tool_name, stats in self.tool_stats.items():
            performance_data[tool_name] = {
                'efficiency_score': (stats['successful_calls'] / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0,
                'speed_category': 'Fast' if stats['avg_execution_time'] < 0.5 else 'Medium' if stats['avg_execution_time'] < 2.0 else 'Slow',
                'usage_frequency': stats['total_calls'] / session_duration * 60 if session_duration > 0 else 0  # calls per minute
            }

        return {
            'session_summary': {
                'session_start': self.session_start,
                'session_duration_minutes': session_duration / 60,
                'total_tool_calls': total_calls,
                'successful_calls': successful_calls,
                'success_rate': (successful_calls / total_calls * 100) if total_calls > 0 else 0,
                'unique_tools_used': len(self.tool_stats),
                'calls_per_minute': total_calls / (session_duration / 60) if session_duration > 0 else 0
            },
            'tool_ranking': tool_usage_ranking,
            'tool_statistics': self.tool_stats,
            'performance_analysis': performance_data,
            'recent_activity': recent_calls,
            'detailed_call_history': self.tool_calls
        }

    def clear_history(self):
        """Clear all tool usage history"""
        self.tool_calls.clear()
        self.tool_stats.clear()
        self.session_start = datetime.now()

# Global tool usage tracker
tool_tracker = ToolUsageTracker()


# Circuit Breaker Pattern (ADD AFTER cached_query)
class CircuitBreaker:
    """Circuit breaker pattern for resilient operations"""

    def __init__(self, failure_threshold=5, recovery_timeout=60, name="CircuitBreaker"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == 'OPEN':
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            raise e

    def get_state(self):
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time
        }


class DatabaseManager:
    """Handles all database operations with connection pooling"""

    def __init__(self, host='localhost', user='root', password='auburn', database='customer_service_db'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connection_pool = None

        # ADD these lines in DatabaseManager.__init__ after existing assignments
        self.circuit_breaker = CircuitBreaker(name="DatabaseCircuitBreaker")
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_query_time': 0,
            'total_query_time': 0
        }
        logger.info(f"DatabaseManager initialized for {self.host}:{self.database}")

    def connect(self):
        """Establish database connection with optimizations"""
        try:
            import mysql.connector.pooling

            # Create connection pool for better performance
            pool_config = {
                'pool_name': 'customer_service_pool',
                'pool_size': 5,  # Number of connections in pool
                'pool_reset_session': True,
                'host': self.host,
                'user': self.user,
                'password': self.password,
                'database': self.database,
                'autocommit': True,  # Auto-commit for faster queries
                'use_unicode': True,
                'charset': 'utf8mb4',
                'buffered': True  # Buffer results for faster fetching
            }

            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)

            # Test connection
            test_conn = self.connection_pool.get_connection()
            test_conn.close()

            print(f"✅ Database connected with pool! Pool size: 5")
            return True

        except Error as e:
            print(f"❌ Database connection failed: {e}")
            # Fallback to single connection
            try:
                self.connection = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    autocommit=True,
                    buffered=True
                )
                return True
            except Error as e2:
                print(f"❌ Fallback connection also failed: {e2}")
                return False

    def disconnect(self):
        """Close database connections and cleanup connection pool"""
        try:
            if self.connection_pool:
                # Close all connections in the pool
                print("🔌 Closing connection pool...")
                # Note: mysql.connector pools don't have a direct close_all method
                # But connections will be closed when the pool object is destroyed
                self.connection_pool = None
                print("✅ Connection pool closed")

            if self.connection and self.connection.is_connected():
                print("🔌 Closing single database connection...")
                self.connection.close()
                print("✅ Database connection closed")

        except Error as e:
            print(f"⚠️  Warning during disconnect: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error during disconnect: {e}")
        finally:
            self.connection = None
            self.connection_pool = None

    @memory_monitor
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query with connection pooling, logging, and circuit breaker"""
        start_time = time.time()

        def _execute():
            try:
                if self.connection_pool:
                    conn = self.connection_pool.get_connection()
                    cursor = conn.cursor(buffered=True)
                    cursor.execute(query, params or ())
                    results = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    return results
                else:
                    cursor = self.connection.cursor(buffered=True)
                    cursor.execute(query, params or ())
                    results = cursor.fetchall()
                    cursor.close()
                    return results
            except Error as e:
                logger.error(f"Query execution failed: {e}")
                raise

        try:
            results = self.circuit_breaker.call(_execute)

            # Update stats
            execution_time = time.time() - start_time
            self.query_stats['total_queries'] += 1
            self.query_stats['successful_queries'] += 1
            self.query_stats['total_query_time'] += execution_time
            self.query_stats['avg_query_time'] = (
                    self.query_stats['total_query_time'] / self.query_stats['total_queries']
            )

            logger.info(f"Query executed successfully in {execution_time:.2f}s")
            return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.query_stats['total_queries'] += 1
            self.query_stats['failed_queries'] += 1
            logger.error(f"Query failed after {execution_time:.2f}s: {e}")
            return []

    def get_connection_info(self):
        """Get connection status information"""
        if self.connection_pool:
            return {
                'type': 'connection_pool',
                'pool_size': 5,
                'status': 'active'
            }
        elif self.connection and self.connection.is_connected():
            return {
                'type': 'single_connection',
                'status': 'connected'
            }
        else:
            return {
                'type': 'none',
                'status': 'disconnected'
            }

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.disconnect()
        except:
            pass  # Ignore errors during cleanup


class AsyncCustomerServiceTools:
    """Async version for parallel operations - complements CustomerServiceTools"""

    def __init__(self, db_manager: DatabaseManager, sync_tools):
        self.db = db_manager
        self.sync_tools = sync_tools
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        logger.info(f"AsyncCustomerServiceTools initialized with {config.max_workers} workers")

    async def get_multiple_orders_with_limit(self, order_ids: List[int]):
        """Get multiple orders with semaphore limiting"""
        async with self.semaphore:
            return await self.get_multiple_orders_parallel(order_ids)

    async def get_multiple_orders_parallel(self, order_ids: List[int]) -> List[Dict]:
        """Get multiple orders in parallel - faster for expert scenarios"""
        loop = asyncio.get_event_loop()

        # Create tasks for parallel execution
        tasks = []
        for order_id in order_ids:
            task = loop.run_in_executor(
                self.executor,
                self.sync_tools.get_order_details,  # Use existing method
                order_id
            )
            tasks.append(task)

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]  # Filter out None results

    async def analyze_multiple_customers_parallel(self, customer_emails: List[str]) -> List[Dict]:
        """Analyze multiple customers in parallel"""
        loop = asyncio.get_event_loop()

        tasks = []
        for email in customer_emails:
            task = loop.run_in_executor(
                self.executor,
                self.sync_tools.analyze_customer_orders_comprehensive,  # Use existing method
                email
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return [r for r in results if r and 'error' not in r]

    def close(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=True)


# Enhanced CustomerServiceTools with tracking
class CustomerServiceTools:
    """Enhanced customer service tool functions with usage tracking"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._cache = {}  # Simple caching for performance
        logger.info("CustomerServiceTools initialized with enhanced tracking")

    # Decorator to track tool usage
    def track_tool_usage(self, reasoning: str = None):
        """Decorator to automatically track tool usage"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                tool_name = func.__name__

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Record successful tool call
                    tool_tracker.record_tool_call(
                        tool_name=tool_name,
                        input_data=args[1:] + tuple(kwargs.values()) if args else kwargs,  # Skip 'self'
                        result=result,
                        execution_time=execution_time,
                        reasoning=reasoning or f"Standard {tool_name} operation"
                    )

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    error_result = {"error": str(e)}

                    # Record failed tool call
                    tool_tracker.record_tool_call(
                        tool_name=tool_name,
                        input_data=args[1:] + tuple(kwargs.values()) if args else kwargs,
                        result=error_result,
                        execution_time=execution_time,
                        reasoning=f"Failed {tool_name} operation: {str(e)}"
                    )

                    return error_result

            return wrapper
        return decorator

    @cached_query(ttl_seconds=600)
    @track_tool_usage("Retrieve items for order to help with return policy or order verification")
    def get_order_items(self, order_id: int) -> List[str]:
        """Given an order ID, returns the list of items purchased for that order"""
        query = """
                SELECT p.product_name
                FROM order_items oi
                         JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = %s
                """
        results = self.db.execute_query(query, (order_id,))
        return [item[0] for item in results] if results else []

    @track_tool_usage("Get delivery date to calculate return policy deadlines or track shipment status")
    def get_delivery_date(self, order_id: int) -> str:
        """Given an order ID, returns the delivery date for that order"""
        query = """
                SELECT DATE_FORMAT(delivery_date, '%d-%b')
                FROM orders
                WHERE order_id = %s
                """
        results = self.db.execute_query(query, (order_id,))
        return results[0][0] if results else ""

    @track_tool_usage("Look up return policy to inform customer of return window and procedures")
    def get_item_return_days(self, item: str) -> int:
        """Given an item name, returns the return policy in days"""
        query = """
                SELECT return_days
                FROM products
                WHERE product_name LIKE %s LIMIT 1
                """
        results = self.db.execute_query(query, (f"%{item}%",))
        return results[0][0] if results else 45  # Default 45 days

    @cached_query(ttl_seconds=300)
    @track_tool_usage("Get comprehensive order information for customer service inquiries")
    def get_order_details(self, order_id: int) -> Dict[str, Any]:
        """Get comprehensive order details with caching"""
        if isinstance(order_id, dict):
            if 'order_id' in order_id:
                order_id = order_id['order_id']
            else:
                return {"error": "Invalid input format. Expected order_id."}

        try:
            order_id = int(order_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid order_id format: {order_id}. Must be a number."}

        query = """
                SELECT o.order_id,
                       o.order_date,
                       o.delivery_date,
                       o.status,
                       o.total_amount,
                       c.first_name,
                       c.last_name,
                       c.email
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = %s
                """
        results = self.db.execute_query(query, (order_id,))
        if results:
            row = results[0]
            return {
                'order_id': row[0],
                'order_date': row[1],
                'delivery_date': row[2],
                'status': row[3],
                'total_amount': row[4],
                'customer_name': f"{row[5]} {row[6]}",
                'customer_email': row[7]
            }
        return {"error": f"Order {order_id} not found"}

    @track_tool_usage("Search customer order history for comprehensive service analysis")
    def search_orders_by_customer(self, email: str) -> List[Dict]:
        """Search orders by customer email"""
        query = """
                SELECT o.order_id, o.order_date, o.status, o.total_amount
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.email LIKE %s
                """
        results = self.db.execute_query(query, (f"%{email}%",))
        return [{'order_id': row[0], 'order_date': row[1], 'status': row[2], 'total_amount': row[3]}
                for row in results]

    # ============= ADD THESE NEW ADVANCED TOOLS =============
    @track_tool_usage("Perform comprehensive customer analysis to identify patterns and issues")
    def analyze_customer_orders_comprehensive(self, customer_email: str) -> Dict[str, Any]:
        """Comprehensive analysis of all customer orders with return eligibility"""
        query = """
                SELECT o.order_id,
                       o.order_date,
                       o.delivery_date,
                       o.status,
                       o.total_amount,
                       p.product_name,
                       p.return_days,
                       oi.quantity,
                       oi.unit_price,
                       DATEDIFF(CURDATE(), o.delivery_date) as days_since_delivery,
                       CASE
                           WHEN o.delivery_date IS NULL THEN 'Not delivered'
                           WHEN DATEDIFF(CURDATE(), o.delivery_date) <= p.return_days THEN 'Returnable'
                           ELSE 'Return expired'
                           END                              as return_status
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                         JOIN order_items oi ON o.order_id = oi.order_id
                         JOIN products p ON oi.product_id = p.product_id
                WHERE c.email LIKE %s
                ORDER BY o.order_date DESC \
                """
        results = self.db.execute_query(query, (f"%{customer_email}%",))

        if not results:
            return {"error": "No orders found for this customer"}

        orders = {}
        total_spent = 0
        returnable_items = []

        for row in results:
            order_id = row[0]
            if order_id not in orders:
                orders[order_id] = {
                    'order_id': order_id,
                    'order_date': row[1],
                    'delivery_date': row[2],
                    'status': row[3],
                    'total_amount': float(row[4]),
                    'items': []
                }
                total_spent += float(row[4])

            item_info = {
                'product_name': row[5],
                'return_days': row[6],
                'quantity': row[7],
                'unit_price': float(row[8]),
                'days_since_delivery': row[9],
                'return_status': row[10]
            }
            orders[order_id]['items'].append(item_info)

            if row[10] == 'Returnable':
                returnable_items.append({
                    'order_id': order_id,
                    'product': row[5],
                    'days_remaining': row[6] - (row[9] or 0)
                })

        return {
            'customer_email': customer_email,
            'total_orders': len(orders),
            'total_spent': total_spent,
            'orders': list(orders.values()),
            'returnable_items': returnable_items
        }

    def calculate_return_policy_and_deadlines(self, order_id: int) -> Dict[str, Any]:
        """Calculate exact return deadlines and policy details for an order"""
        query = """
                SELECT o.order_id,
                       o.delivery_date,
                       o.total_amount,
                       p.product_name,
                       p.return_days,
                       p.price,
                       oi.quantity,
                       oi.unit_price,
                       DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY)                      as return_deadline,
                       DATEDIFF(DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY), CURDATE()) as days_remaining,
                       CASE
                           WHEN DATEDIFF(DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY), CURDATE()) > 0
                               THEN 'Active'
                           ELSE 'Expired'
                           END                                                                    as return_window_status
                FROM orders o
                         JOIN order_items oi ON o.order_id = oi.order_id
                         JOIN products p ON oi.product_id = p.product_id
                WHERE o.order_id = %s
                  AND o.delivery_date IS NOT NULL \
                """
        results = self.db.execute_query(query, (order_id,))

        if not results:
            return {"error": f"Order {order_id} not found or not delivered"}

        order_info = {
            'order_id': order_id,
            'delivery_date': results[0][1],
            'total_amount': float(results[0][2]),
            'items': [],
            'return_summary': {
                'items_returnable': 0,
                'items_expired': 0,
                'earliest_deadline': None,
                'latest_deadline': None
            }
        }

        deadlines = []
        for row in results:
            item_data = {
                'product_name': row[3],
                'return_days': row[4],
                'price': float(row[5]),
                'quantity': row[6],
                'unit_price': float(row[7]),
                'return_deadline': row[8],
                'days_remaining': row[9],
                'status': row[10]
            }
            order_info['items'].append(item_data)
            deadlines.append(row[8])

            if row[10] == 'Active':
                order_info['return_summary']['items_returnable'] += 1
            else:
                order_info['return_summary']['items_expired'] += 1

        if deadlines:
            order_info['return_summary']['earliest_deadline'] = min(deadlines)
            order_info['return_summary']['latest_deadline'] = max(deadlines)

        return order_info

    def analyze_geographic_performance(self, state: str = None, city: str = None) -> Dict[str, Any]:
        """Analyze order patterns and performance by geographic location"""
        where_clause = "WHERE 1=1"
        params = []

        if state:
            where_clause += " AND c.state = %s"
            params.append(state)
        if city:
            where_clause += " AND c.city = %s"
            params.append(city)

        query = f"""
            SELECT 
                c.city, c.state,
                COUNT(DISTINCT o.order_id) as total_orders,
                COUNT(DISTINCT c.customer_id) as unique_customers,
                AVG(o.total_amount) as avg_order_value,
                SUM(o.total_amount) as total_revenue,
                AVG(DATEDIFF(o.delivery_date, o.order_date)) as avg_delivery_time,
                COUNT(CASE WHEN o.status = 'delivered' THEN 1 END) as delivered_orders,
                COUNT(CASE WHEN o.status = 'pending' THEN 1 END) as pending_orders,
                COUNT(CASE WHEN o.status = 'cancelled' THEN 1 END) as cancelled_orders
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            {where_clause}
            GROUP BY c.city, c.state
            ORDER BY total_revenue DESC
        """
        results = self.db.execute_query(query, params)

        locations = []
        total_analysis = {
            'total_locations': len(results),
            'total_orders': 0,
            'total_customers': 0,
            'total_revenue': 0,
            'issues_identified': []
        }

        for row in results:
            location_data = {
                'city': row[0],
                'state': row[1],
                'total_orders': row[2],
                'unique_customers': row[3],
                'avg_order_value': float(row[4] or 0),
                'total_revenue': float(row[5] or 0),
                'avg_delivery_time': float(row[6] or 0),
                'delivered_orders': row[7],
                'pending_orders': row[8],
                'cancelled_orders': row[9],
                'delivery_success_rate': (row[7] / row[2] * 100) if row[2] > 0 else 0
            }
            locations.append(location_data)

            total_analysis['total_orders'] += row[2]
            total_analysis['total_customers'] += row[3]
            total_analysis['total_revenue'] += float(row[5] or 0)

            # Identify issues
            if row[6] and row[6] > 7:
                total_analysis['issues_identified'].append(f"{row[0]}, {row[1]}: Slow delivery ({row[6]:.1f} days avg)")
            if row[9] > 0:
                total_analysis['issues_identified'].append(f"{row[0]}, {row[1]}: {row[9]} cancelled orders")

        return {
            'locations': locations,
            'summary': total_analysis
        }

    def generate_predictive_risk_analysis(self) -> Dict[str, Any]:
        """Generate predictive analysis for customer service risks"""
        from datetime import datetime

        query = """
                SELECT o.order_id,
                       o.customer_id,
                       o.order_date,
                       o.delivery_date,
                       o.status,
                       o.total_amount,
                       c.email,
                       c.city,
                       c.state,
                       DATEDIFF(CURDATE(), o.order_date)    as days_since_order,
                       DATEDIFF(CURDATE(), o.delivery_date) as days_since_delivery,
                       COUNT(o2.order_id)                   as customer_order_history
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                         LEFT JOIN orders o2 ON c.customer_id = o2.customer_id AND o2.order_date < o.order_date
                WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
                GROUP BY o.order_id
                ORDER BY o.order_date DESC \
                """
        results = self.db.execute_query(query, ())

        risk_customers = []
        risk_factors = {
            'late_deliveries': 0,
            'pending_too_long': 0,
            'high_value_at_risk': 0,
            'new_customers_with_issues': 0
        }

        for row in results:
            customer_risk = {
                'order_id': row[0],
                'customer_email': row[6],
                'location': f"{row[7]}, {row[8]}",
                'order_value': float(row[5]),
                'risk_factors': [],
                'risk_score': 0
            }

            # Late delivery risk
            if row[4] == 'shipped' and row[2] and (datetime.now().date() - row[2]).days > 10:
                customer_risk['risk_factors'].append("Order shipped but overdue for delivery")
                customer_risk['risk_score'] += 3
                risk_factors['late_deliveries'] += 1

            # Pending too long
            if row[4] == 'pending' and row[9] > 5:
                customer_risk['risk_factors'].append("Order pending for too long")
                customer_risk['risk_score'] += 2
                risk_factors['pending_too_long'] += 1

            # High value orders
            if row[5] > 1500:
                customer_risk['risk_factors'].append("High-value order requiring attention")
                customer_risk['risk_score'] += 1
                risk_factors['high_value_at_risk'] += 1

            # New customer with potential issues
            if row[11] == 0 and (row[4] in ['pending', 'processing'] and row[9] > 3):
                customer_risk['risk_factors'].append("New customer with delayed first order")
                customer_risk['risk_score'] += 2
                risk_factors['new_customers_with_issues'] += 1

            if customer_risk['risk_score'] > 0:
                risk_customers.append(customer_risk)

        # Sort by risk score
        risk_customers.sort(key=lambda x: x['risk_score'], reverse=True)

        return {
            'high_risk_customers': [c for c in risk_customers if c['risk_score'] >= 3],
            'medium_risk_customers': [c for c in risk_customers if 1 <= c['risk_score'] < 3],
            'risk_summary': risk_factors,
            'total_at_risk': len(risk_customers),
            'proactive_actions': [
                "Contact high-risk customers proactively",
                "Expedite late shipments",
                "Offer compensation for delays",
                "Implement order tracking improvements",
                "Set up automated delay notifications"
            ]
        }

    def analyze_orders_by_status_and_timeframe(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Analyze orders by status within a timeframe with delivery performance"""
        if not start_date:
            start_date = '2024-06-01'
        if not end_date:
            end_date = '2024-06-30'

        query = """
                SELECT o.status,
                       COUNT(*)                                                                as order_count,
                       AVG(o.total_amount)                                                     as avg_order_value,
                       SUM(o.total_amount)                                                     as total_value,
                       AVG(DATEDIFF(o.delivery_date, o.order_date))                            as avg_delivery_days,
                       COUNT(CASE WHEN DATEDIFF(o.delivery_date, o.order_date) > 7 THEN 1 END) as delayed_deliveries
                FROM orders o
                WHERE o.order_date BETWEEN %s AND %s
                GROUP BY o.status
                ORDER BY order_count DESC \
                """
        results = self.db.execute_query(query, (start_date, end_date))

        analysis = {
            'timeframe': f"{start_date} to {end_date}",
            'status_breakdown': [],
            'total_orders': 0,
            'total_revenue': 0,
            'performance_issues': []
        }

        for row in results:
            status_data = {
                'status': row[0],
                'count': row[1],
                'avg_order_value': float(row[2] or 0),
                'total_value': float(row[3] or 0),
                'avg_delivery_days': float(row[4] or 0),
                'delayed_deliveries': row[5]
            }
            analysis['status_breakdown'].append(status_data)
            analysis['total_orders'] += row[1]
            analysis['total_revenue'] += float(row[3] or 0)

            # Identify performance issues
            if row[4] and row[4] > 7:
                analysis['performance_issues'].append(
                    f"Status '{row[0]}': Average delivery time {row[4]:.1f} days (target: 7 days)")
            if row[5] > 0:
                analysis['performance_issues'].append(
                    f"Status '{row[0]}': {row[5]} delayed deliveries out of {row[1]} orders")

        return analysis

    def analyze_product_performance_and_issues(self, product_category: str = None) -> Dict[str, Any]:
        """Analyze product performance, popularity, and potential issues"""
        where_clause = ""
        params = []

        if product_category:
            where_clause = "WHERE c.category_name LIKE %s"
            params.append(f"%{product_category}%")

        query = f"""
            SELECT 
                p.product_name, c.category_name, p.return_days,
                COUNT(oi.order_item_id) as times_ordered,
                SUM(oi.quantity) as total_quantity_sold,
                AVG(oi.unit_price) as avg_selling_price,
                SUM(oi.quantity * oi.unit_price) as total_revenue,
                COUNT(DISTINCT o.customer_id) as unique_customers,
                AVG(DATEDIFF(o.delivery_date, o.order_date)) as avg_delivery_time
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            {where_clause}
            GROUP BY p.product_id, p.product_name, c.category_name, p.return_days
            ORDER BY total_revenue DESC
        """
        results = self.db.execute_query(query, params)

        products = []
        category_summary = {}

        for row in results:
            product_data = {
                'product_name': row[0],
                'category': row[1],
                'return_days': row[2],
                'times_ordered': row[3],
                'total_quantity_sold': row[4],
                'avg_selling_price': float(row[5]),
                'total_revenue': float(row[6]),
                'unique_customers': row[7],
                'avg_delivery_time': float(row[8] or 0),
                'popularity_rank': len(products) + 1
            }
            products.append(product_data)

            # Category aggregation
            category = row[1]
            if category not in category_summary:
                category_summary[category] = {
                    'total_revenue': 0,
                    'total_orders': 0,
                    'product_count': 0
                }
            category_summary[category]['total_revenue'] += float(row[6])
            category_summary[category]['total_orders'] += row[3]
            category_summary[category]['product_count'] += 1

        return {
            'products': products,
            'category_summary': category_summary,
            'top_products': products[:5] if products else []
        }

    def get_multiple_order_details(self, order_ids: str) -> Dict[str, Any]:
        """Get details for multiple orders - handles comma-separated order IDs"""
        # Parse order IDs from string
        try:
            if isinstance(order_ids, str):
                # Handle different formats: "1007, 1017, 1023" or "1007,1017,1023"
                ids = [int(x.strip()) for x in order_ids.replace(',', ' ').split()]
            elif isinstance(order_ids, list):
                ids = [int(x) for x in order_ids]
            else:
                ids = [int(order_ids)]
        except (ValueError, TypeError):
            return {"error": f"Invalid order IDs format: {order_ids}"}

        query = """
            SELECT o.order_id, o.order_date, o.delivery_date, o.status,
                   o.total_amount, c.first_name, c.last_name, c.email,
                   p.product_name, p.return_days
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE o.order_id IN ({})
            ORDER BY o.order_id
        """.format(','.join(['%s'] * len(ids)))

        results = self.db.execute_query(query, tuple(ids))

        if not results:
            return {"error": f"No orders found for IDs: {order_ids}"}

        orders = {}
        for row in results:
            order_id = row[0]
            if order_id not in orders:
                orders[order_id] = {
                    'order_id': order_id,
                    'order_date': row[1],
                    'delivery_date': row[2],
                    'status': row[3],
                    'total_amount': float(row[4]),
                    'customer_name': f"{row[5]} {row[6]}",
                    'customer_email': row[7],
                    'products': []
                }

            orders[order_id]['products'].append({
                'product_name': row[8],
                'return_days': row[9]
            })

        return {
            'orders_found': len(orders),
            'orders': list(orders.values())
        }


class QueryDecomposer:
    """Intelligent query decomposition for complex customer service queries - CLEANED VERSION"""

    def __init__(self, agent_runner):
        self.agent = agent_runner
        self.complexity_patterns = self._define_complexity_patterns()

    def _define_complexity_patterns(self):
        """Define patterns that indicate complex queries needing decomposition"""
        return {
            'multi_customer': [
                'three customers', 'multiple customers', 'several customers',
                'customers who', 'all customers', 'group of customers'
            ],
            'multi_analysis': [
                'analyze and compare', 'breakdown and explain', 'investigate and recommend',
                'calculate and propose', 'identify and suggest', 'examine and determine'
            ],
            'multi_timeframe': [
                'last week and next week', 'previous month and current',
                'compare timeframes', 'historical and current'
            ],
            'multi_criteria': [
                'orders that are', 'customers who have', 'products that',
                'based on multiple factors', 'considering various aspects'
            ],
            'predictive_analysis': [
                'predict', 'forecast', 'likely to', 'identify risks',
                'proactive', 'prevent problems', 'anticipate issues'
            ],
            'business_impact': [
                'revenue impact', 'business risk', 'financial analysis',
                'cost calculation', 'ROI analysis', 'profit impact'
            ]
        }

    def assess_query_complexity(self, query: str) -> Dict[str, Any]:
        """Assess if a query is complex and needs decomposition"""
        query_lower = query.lower()

        complexity_score = 0
        detected_patterns = []
        complexity_indicators = []

        # Check for complexity patterns
        for pattern_type, keywords in self.complexity_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    complexity_score += 1
                    detected_patterns.append(pattern_type)
                    complexity_indicators.append(keyword)
                    break  # Avoid double counting for same pattern type

        # Additional complexity indicators
        if len(query.split()) > 30:  # Long queries
            complexity_score += 1
            complexity_indicators.append("long_query")

        if query.count('?') > 1:  # Multiple questions
            complexity_score += 1
            complexity_indicators.append("multiple_questions")

        if any(word in query_lower for word in ['and', 'also', 'additionally', 'furthermore']):
            complexity_score += 1
            complexity_indicators.append("multiple_requirements")

        return {
            'is_complex': complexity_score >= 2,
            'complexity_score': complexity_score,
            'detected_patterns': list(set(detected_patterns)),
            'indicators': complexity_indicators,
            'requires_decomposition': complexity_score >= 3
        }

    def decompose_query(self, query: str) -> List[Dict[str, str]]:
        """Break down complex query into manageable subgoals (5-10 subgoals)"""
        assessment = self.assess_query_complexity(query)

        if not assessment['requires_decomposition']:
            # Force minimum decomposition even for simple queries
            return self._force_minimum_decomposition(query)

        # MODIFIED: Updated prompt for 5-10 subgoals
        decomposition_prompt = f"""
        Break down this complex customer service query into 5-10 specific, actionable subgoals that can be executed sequentially.
        Each subgoal should be focused on a single task and specific enough to be executed by one tool call.
        
        REQUIREMENTS:
        - Minimum 5 subgoals (even for simpler queries)
        - Maximum 10 subgoals (break complex tasks into smaller steps)
        - Each subgoal should be actionable and specific
        - Arrange in logical execution order
        
        Original Query: "{query}"
        
        Format your response as a numbered list of subgoals:
        1. [First specific data collection step]
        2. [Second data collection step]
        3. [Third data collection or validation step]
        4. [First analysis step]
        5. [Second analysis step]
        6. [Additional analysis if needed]
        7. [Synthesis or comparison step]
        8. [Recommendation generation]
        9. [Additional recommendations if complex]
        10. [Final summary or action plan]
        
        Adjust the number between 5-10 based on query complexity, but ensure each step is meaningful and necessary.
        """

        try:
            # Use the agent's LLM for decomposition
            from llama_index.core import Settings
            decomposition_response = Settings.llm.complete(decomposition_prompt)

            # Parse the response into subgoals
            subgoals = self._parse_decomposition_response(str(decomposition_response))

            # ADDED: Enforce 5-10 subgoal range
            subgoals = self._enforce_subgoal_range(subgoals, query)

            return subgoals

        except Exception as e:
            print(f"⚠️  Decomposition failed, using pattern-based fallback: {e}")
            return self._pattern_based_decomposition_extended(query, assessment)

    def _force_minimum_decomposition(self, query: str) -> List[Dict[str, str]]:
        """Force even simple queries into minimum 5 subgoals"""
        return [
            {'subgoal': f'Identify the primary entities mentioned in the query: "{query}"', 'type': 'data_collection', 'priority': 1},
            {'subgoal': 'Collect detailed information for the identified entities', 'type': 'data_collection', 'priority': 2},
            {'subgoal': 'Validate and cross-reference the collected data', 'type': 'data_collection', 'priority': 3},
            {'subgoal': 'Analyze the data to extract key insights and patterns', 'type': 'analysis', 'priority': 4},
            {'subgoal': 'Generate comprehensive response with actionable recommendations', 'type': 'synthesis', 'priority': 5}
        ]

    def _enforce_subgoal_range(self, subgoals: List[Dict], original_query: str) -> List[Dict]:
        """Ensure subgoals are between 5-10"""

        if len(subgoals) < 5:
            print(f"⚠️  Only {len(subgoals)} subgoals generated, expanding to minimum 5...")
            return self._expand_subgoals(subgoals, original_query, target_count=5)

        elif len(subgoals) > 10:
            print(f"⚠️  {len(subgoals)} subgoals generated, condensing to maximum 10...")
            return self._condense_subgoals(subgoals, target_count=10)

        else:
            print(f"✅ {len(subgoals)} subgoals generated (within 5-10 range)")
            return subgoals

    def _expand_subgoals(self, subgoals: List[Dict], query: str, target_count: int) -> List[Dict]:
        """Expand subgoals to reach minimum count"""
        expanded = subgoals.copy()

        # Add data validation steps
        if len(expanded) < target_count:
            expanded.insert(1, {
                'subgoal': 'Validate input parameters and check data availability',
                'type': 'data_collection',
                'priority': len(expanded) + 1
            })

        # Add cross-reference step
        if len(expanded) < target_count:
            expanded.insert(-1, {
                'subgoal': 'Cross-reference findings with related data sources',
                'type': 'analysis',
                'priority': len(expanded) + 1
            })

        # Add quality check step
        if len(expanded) < target_count:
            expanded.insert(-1, {
                'subgoal': 'Perform quality check on analysis results',
                'type': 'analysis',
                'priority': len(expanded) + 1
            })

        # Add alternative solutions step
        if len(expanded) < target_count:
            expanded.insert(-1, {
                'subgoal': 'Generate alternative solutions or recommendations',
                'type': 'synthesis',
                'priority': len(expanded) + 1
            })

        # Add final verification step
        if len(expanded) < target_count:
            expanded.append({
                'subgoal': 'Verify all recommendations align with business policies',
                'type': 'synthesis',
                'priority': len(expanded) + 1
            })

        # Update priorities
        for i, subgoal in enumerate(expanded):
            subgoal['priority'] = i + 1

        return expanded

    def _condense_subgoals(self, subgoals: List[Dict], target_count: int) -> List[Dict]:
        """Condense subgoals to maximum count by combining similar ones"""
        if len(subgoals) <= target_count:
            return subgoals

        # Group by type
        data_collection = [s for s in subgoals if s['type'] == 'data_collection']
        analysis = [s for s in subgoals if s['type'] == 'analysis']
        synthesis = [s for s in subgoals if s['type'] == 'synthesis']

        condensed = []

        # Keep first 3-4 data collection steps
        if len(data_collection) > 4:
            condensed.extend(data_collection[:2])
            # Combine remaining data collection
            combined_data = {
                'subgoal': f"Complete additional data collection: {'; '.join([s['subgoal'][:50] + '...' for s in data_collection[2:]])}",
                'type': 'data_collection',
                'priority': 3
            }
            condensed.append(combined_data)
        else:
            condensed.extend(data_collection)

        # Keep first 3-4 analysis steps
        remaining_slots = target_count - len(condensed) - min(2, len(synthesis))
        analysis_to_keep = min(remaining_slots, len(analysis))

        if analysis_to_keep < len(analysis):
            condensed.extend(analysis[:analysis_to_keep-1])
            # Combine remaining analysis
            combined_analysis = {
                'subgoal': f"Complete comprehensive analysis including: {'; '.join([s['subgoal'][:40] + '...' for s in analysis[analysis_to_keep-1:]])}",
                'type': 'analysis',
                'priority': len(condensed) + 1
            }
            condensed.append(combined_analysis)
        else:
            condensed.extend(analysis)

        # Keep final synthesis steps
        final_synthesis_count = min(2, len(synthesis), target_count - len(condensed))
        condensed.extend(synthesis[:final_synthesis_count])

        # Update priorities
        for i, subgoal in enumerate(condensed):
            subgoal['priority'] = i + 1

        return condensed[:target_count]

    def _pattern_based_decomposition_extended(self, query: str, assessment: Dict) -> List[Dict[str, str]]:
        """Extended pattern-based decomposition with 5-10 subgoals"""
        subgoals = []

        # Multi-customer pattern (7 subgoals)
        if 'multi_customer' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Parse and validate customer identifiers from the query', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Retrieve basic customer information and contact details', 'type': 'data_collection', 'priority': 2},
                {'subgoal': 'Get comprehensive order history for each customer', 'type': 'data_collection', 'priority': 3},
                {'subgoal': 'Analyze order patterns and customer behavior individually', 'type': 'analysis', 'priority': 4},
                {'subgoal': 'Compare customers and identify common patterns or issues', 'type': 'analysis', 'priority': 5},
                {'subgoal': 'Generate individual recommendations for each customer', 'type': 'synthesis', 'priority': 6},
                {'subgoal': 'Create summary report with actionable next steps', 'type': 'synthesis', 'priority': 7}
            ])

        # Predictive analysis pattern (8 subgoals)
        elif 'predictive_analysis' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Collect historical data relevant to prediction requirements', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Gather current state data and recent trends', 'type': 'data_collection', 'priority': 2},
                {'subgoal': 'Validate data quality and identify any gaps', 'type': 'data_collection', 'priority': 3},
                {'subgoal': 'Identify patterns and trends in historical data', 'type': 'analysis', 'priority': 4},
                {'subgoal': 'Assess current risk factors and warning indicators', 'type': 'analysis', 'priority': 5},
                {'subgoal': 'Generate predictions based on identified patterns', 'type': 'analysis', 'priority': 6},
                {'subgoal': 'Develop proactive recommendations and action plans', 'type': 'synthesis', 'priority': 7},
                {'subgoal': 'Create monitoring strategy for ongoing risk management', 'type': 'synthesis', 'priority': 8}
            ])

        # Business impact pattern (6 subgoals)
        elif 'business_impact' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Identify and quantify the business metrics involved', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Collect financial and operational data for impact analysis', 'type': 'data_collection', 'priority': 2},
                {'subgoal': 'Calculate direct financial impact and costs', 'type': 'analysis', 'priority': 3},
                {'subgoal': 'Assess indirect business risks and operational implications', 'type': 'analysis', 'priority': 4},
                {'subgoal': 'Develop strategies to minimize negative impact', 'type': 'synthesis', 'priority': 5},
                {'subgoal': 'Create implementation plan with timeline and resources', 'type': 'synthesis', 'priority': 6}
            ])

        # Default extended decomposition (5 subgoals minimum)
        else:
            subgoals = [
                {'subgoal': 'Identify and extract key entities and requirements from query', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Collect comprehensive data for all identified entities', 'type': 'data_collection', 'priority': 2},
                {'subgoal': 'Validate data completeness and cross-reference information', 'type': 'data_collection', 'priority': 3},
                {'subgoal': 'Analyze collected data and identify key insights', 'type': 'analysis', 'priority': 4},
                {'subgoal': 'Generate comprehensive response with actionable recommendations', 'type': 'synthesis', 'priority': 5}
            ]

        return subgoals

    def _parse_decomposition_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response into structured subgoals"""
        lines = response.strip().split('\n')
        subgoals = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the line
                clean_line = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-', '•']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        break

                if clean_line:
                    subgoal_type = 'data_collection' if i < 2 else 'analysis' if i < 4 else 'synthesis'
                    subgoals.append({
                        'subgoal': clean_line,
                        'type': subgoal_type,
                        'priority': i + 1
                    })

        return subgoals if subgoals else [{'subgoal': response, 'type': 'simple', 'priority': 1}]

    def execute_decomposed_query(self, query: str) -> str:
        """Execute query with automatic decomposition - SIMPLIFIED VERSION"""
        print(f"🔍 Analyzing query complexity...")

        assessment = self.assess_query_complexity(query)

        if not assessment['requires_decomposition']:
            print("✅ Simple query detected, but enforcing minimum decomposition...")

        print(f"🧩 Query complexity score: {assessment['complexity_score']}")
        print(f"📋 Patterns found: {', '.join(assessment['detected_patterns'])}")
        print("🔄 Breaking down into 5-10 subgoals...")

        subgoals = self.decompose_query(query)

        print(f"\n📝 Query decomposed into {len(subgoals)} subgoals:")
        for i, subgoal in enumerate(subgoals, 1):
            print(f"   {i}. {subgoal['subgoal']} [{subgoal['type']}]")

        # Execute subgoals sequentially
        results = []
        context = []

        for i, subgoal in enumerate(subgoals, 1):
            print(f"\n🎯 Executing subgoal {i}: {subgoal['subgoal']}")
            print("-" * 40)

            # Add context from previous subgoals
            contextual_query = subgoal['subgoal']
            if context:
                contextual_query = f"Based on previous analysis: {' '.join(context[-2:])}. Now: {subgoal['subgoal']}"

            try:
                # SIMPLIFIED: Use agent's method directly
                result = self._execute_simple_query(contextual_query)
                results.append({
                    'subgoal': subgoal['subgoal'],
                    'result': result,
                    'type': subgoal['type']
                })
                context.append(f"Subgoal {i} found: {str(result)[:200]}...")

            except Exception as e:
                print(f"⚠️  Subgoal {i} failed: {e}")
                results.append({
                    'subgoal': subgoal['subgoal'],
                    'result': f"Failed to execute: {e}",
                    'type': subgoal['type']
                })

        # SIMPLIFIED: Use agent's synthesis method directly
        return self._synthesize_results(query, results)

    def _execute_simple_query(self, query: str) -> str:
        """Execute a simple query using the agent - SIMPLIFIED"""
        try:
            # Use agent's method if available, otherwise fallback
            if hasattr(self.agent, '_execute_simple_query'):
                return self.agent._execute_simple_query(query)
            else:
                response = self.agent.query(query)
                return str(response)
        except Exception as e:
            return f"Query execution failed: {e}"

    def _synthesize_results(self, original_query: str, results: List[Dict]) -> str:
        """Synthesize results - SIMPLIFIED"""
        try:
            # Use agent's method if available, otherwise fallback
            if hasattr(self.agent, '_synthesize_results'):
                return self.agent._synthesize_results(original_query, results)
            else:
                return self._simple_fallback_synthesis(original_query, results)
        except Exception as e:
            return self._simple_fallback_synthesis(original_query, results)

    def _simple_fallback_synthesis(self, original_query: str, results: List[Dict]) -> str:
        """Simple fallback synthesis - MINIMAL VERSION"""
        final_response = f"## Comprehensive Analysis for: {original_query}\n\n"

        for i, result in enumerate(results, 1):
            final_response += f"### Step {i}: {result['subgoal']}\n"
            final_response += f"**Type:** {result['type']}\n"
            final_response += f"**Result:** {result['result']}\n\n"

        final_response += f"### Summary\n"
        final_response += f"Completed analysis of {len(results)} subgoals."

        return final_response

class CustomerServiceAgent:
    """Main AI Agent class with comprehensive tool usage tracking"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.tools = None
        self.agent = None
        self.support_index = None
        self.query_decomposer = None
        self.sync_tools = None
        self.async_tools = None

        # ADD THESE LINES for call graph tracking:
        self.execution_tracker = EnhancedQueryExecutionTracker()
        global execution_tracker
        execution_tracker = self.execution_tracker  # Use instance tracker

        print("🎯 Enhanced Customer Service Agent with Call Graph Tracking initialized")

    def setup_database(self):
        """Setup database connection"""
        print("🔌 Connecting to database...")
        if not self.db_manager.connect():
            return False

        # Test connection
        test_query = "SELECT COUNT(*) FROM orders"
        results = self.db_manager.execute_query(test_query)
        if results:
            print(f"✅ Database connected! Found {results[0][0]} orders.")
            return True
        return False

    def setup_llm(self):
        """Setup Local LLM connection with comprehensive configuration from .env"""
        if not LLAMAINDEX_AVAILABLE:
            logger.error("LlamaIndex not available")
            return False

        logger.info("Setting up Local LLM with configuration from .env file...")
        print("🤖 Setting up Local LLM with environment configuration...")

        try:
            # Determine device for embeddings
            device_str = self._determine_embedding_device()

            # Setup the LLM with all configurable parameters
            logger.info(f"Configuring LLM: {config.llm_model} at {config.llm_url}")
            Settings.llm = OpenAILike(
                model=config.llm_model,
                api_base=config.llm_url,
                api_key=config.llm_api_key,
                is_local=True,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                timeout=config.llm_timeout,
                max_retries=config.llm_max_retries
            )

            # Setup embedding model with comprehensive configuration
            logger.info(f"Configuring embeddings: {config.embedding_model} on {device_str}")
            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=config.embedding_model,
                    max_length=config.embedding_max_length,
                    device=device_str,
                    trust_remote_code=config.embedding_trust_remote_code
                )
                logger.info(f"Embeddings configured successfully on {device_str}")
            except Exception as embed_error:
                logger.warning(f"GPU embedding failed, falling back to CPU: {embed_error}")
                print(f"⚠️  GPU embedding failed, falling back to CPU: {embed_error}")

                # Fallback to CPU
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=config.embedding_model,
                    max_length=config.embedding_max_length,
                    device="cpu",
                    trust_remote_code=config.embedding_trust_remote_code
                )
                device_str = "cpu"

            # Set device for sentence transformers
            os.environ["SENTENCE_TRANSFORMERS_DEVICE"] = device_str

            # Log configuration summary
            self._log_llm_configuration(device_str)

            # Test LLM connection
            if self._test_llm_connection():
                logger.info("LLM setup completed successfully")
                return True
            else:
                logger.error("LLM connection test failed")
                return False

        except Exception as e:
            logger.error(f"LLM setup failed: {e}")
            print(f"❌ LLM setup failed: {e}")
            print("💡 Please check your .env configuration and ensure LM Studio is running")
            return False

    def _determine_embedding_device(self):
        """Determine the best device for embeddings based on configuration and availability"""
        if config.embedding_device.lower() == 'auto':
            # Auto-detect best device
            try:
                import torch
                if config.use_gpu and torch.backends.mps.is_available():
                    logger.info("Mac GPU (Metal Performance Shaders) detected and enabled")
                    print("🚀 Using Mac GPU (Metal Performance Shaders)")
                    return "mps"
                elif config.use_gpu and torch.cuda.is_available():
                    logger.info("CUDA GPU detected and enabled")
                    print("🚀 Using CUDA GPU")
                    return "cuda"
                else:
                    logger.info("Using CPU for embeddings (no GPU available or disabled)")
                    print("💻 Using CPU for embeddings")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not available, using CPU")
                print("💻 PyTorch not available, using CPU")
                return "cpu"
        else:
            # Use specified device
            logger.info(f"Using specified device for embeddings: {config.embedding_device}")
            print(f"🎯 Using specified device: {config.embedding_device}")
            return config.embedding_device

    def _log_llm_configuration(self, device_str):
        """Log comprehensive LLM configuration"""
        print("✅ LLM setup completed with configuration:")
        print(f"   🎯 Model: {config.llm_model}")
        print(f"   🌐 URL: {config.llm_url}")
        print(f"   🌡️  Temperature: {config.llm_temperature}")
        print(f"   📝 Max tokens: {config.llm_max_tokens}")
        print(f"   ⏱️  Timeout: {config.llm_timeout}s")
        print(f"   🔄 Retries: {config.llm_max_retries}")
        print(f"   🧠 Embedding model: {config.embedding_model}")
        print(f"   🚀 Embedding device: {device_str}")
        print(f"   📏 Max embedding length: {config.embedding_max_length}")

        logger.info(f"LLM Configuration Summary:")
        logger.info(f"  Model: {config.llm_model} at {config.llm_url}")
        logger.info(f"  Temperature: {config.llm_temperature}, Max tokens: {config.llm_max_tokens}")
        logger.info(f"  Timeout: {config.llm_timeout}s, Retries: {config.llm_max_retries}")
        logger.info(f"  Embedding: {config.embedding_model} on {device_str}")

    def _test_llm_connection(self):
        """Test LLM connection with a simple query"""
        try:
            logger.debug("Testing LLM connection...")
            print("🔬 Testing LLM connection...")

            # Simple test query
            test_response = Settings.llm.complete("Hello")

            if test_response and str(test_response).strip():
                print("✅ LLM connection test successful")
                logger.info("LLM connection test successful")
                return True
            else:
                print("❌ LLM connection test failed - empty response")
                logger.error("LLM connection test failed - empty response")
                return False

        except Exception as e:
            print(f"❌ LLM connection test failed: {e}")
            logger.error(f"LLM connection test failed: {e}")
            return False

    def show_configuration(self):
        """Display current configuration (add this new method to CustomerServiceAgent)"""
        print("\n⚙️  Current Configuration")
        print("=" * 40)

        print(f"🗄️  Database:")
        print(f"   Host: {config.database_host}")
        print(f"   Database: {config.database_name}")
        print(f"   Pool Size: {config.database_pool_size}")

        print(f"\n🤖 LLM:")
        print(f"   URL: {config.llm_url}")
        print(f"   Model: {config.llm_model}")
        print(f"   Temperature: {config.llm_temperature}")
        print(f"   Max Tokens: {config.llm_max_tokens}")
        print(f"   Timeout: {config.llm_timeout}s")

        print(f"\n🧠 Embeddings:")
        print(f"   Model: {config.embedding_model}")
        print(f"   Device: {config.embedding_device}")
        print(f"   Max Length: {config.embedding_max_length}")

        print(f"\n⚡ Performance:")
        print(f"   Max Workers: {config.max_workers}")
        print(f"   Cache TTL: {config.cache_ttl}s")
        print(f"   Chunk Size: {config.chunk_size}")
        print(f"   Memory Monitoring: {'✅' if config.enable_memory_monitoring else '❌'}")
        print(f"   Query Caching: {'✅' if config.enable_query_caching else '❌'}")

        print(f"\n📁 Paths:")
        print(f"   Policy Files: {config.policy_files_dir}")
        print(f"   HF Cache: {config.hf_hub_cache}")
        print(f"   ST Cache: {config.st_cache}")
        print(f"   Log File: {config.log_file}")

    def _get_file_description(self, filename):
        """Get description for support file"""
        descriptions = {
            'Customer Service.txt': 'Contact information and response times',
            'FAQ.txt': 'Frequently asked questions',
            'Return Policy.txt': 'Return and refund policies',
            'Warranty Policy.txt': 'Comprehensive warranty coverage',
            'Escalation Procedures.txt': 'Multi-level escalation framework',
            'Technical Troubleshooting Guide.txt': 'Hardware/software support',
            'Business Policies and Procedures.txt': 'Customer tiers and policies',
            'Product Knowledge Database.txt': 'Product specifications',
            'Order Management and Fulfillment.txt': 'Order lifecycle procedures'
        }
        return descriptions.get(filename, 'Support document')

    def get_support_files_from_user(self):
        """Get support files from user input with configurable policy_files directory"""
        print("\n📚 Support Document Configuration")
        print("=" * 50)
        print("Complete support document suite includes:")
        for i, file in enumerate(config.support_files_default, 1):
            print(f"{i}. {file.replace('.txt', '')} - {self._get_file_description(file)}")
        print()
        print(f"📁 Documents location: ./{config.policy_files_dir}/")
        print()

        # Define complete document suite with configurable path
        complete_suite = [
            f"{config.policy_files_dir}/{file}" for file in config.support_files_default
        ]

        # Basic document suite (first 3 from config)
        basic_suite = [
            f"{config.policy_files_dir}/{file}" for file in config.support_files_default[:3]
        ]

        while True:
            print("📋 Document Suite Options:")
            print("1. 🚀 Complete Suite (All 9 documents) - Recommended for all scenarios")
            print("2. 📝 Basic Suite (Original 3 documents) - Basic queries only")
            print("3. 🔧 Custom Selection - Choose specific documents")
            print("4. 🔍 Verify Files - Check which files exist in policy_files/")
            print()

            choice = input("Select option (1-4): ").strip()

            if choice == '1':
                print("✅ Selected: Complete Suite (9 documents)")
                print("   This enables all 15 test scenarios including Expert level!")
                return complete_suite

            elif choice == '2':
                print("⚠️  Selected: Basic Suite (3 documents)")
                print("   This supports Basic scenarios (1-3) only.")
                print("   Advanced/Expert scenarios (4-15) will have limited capability.")
                confirm = input("Continue with basic suite? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return basic_suite
                else:
                    continue

            elif choice == '3':
                return self._get_custom_file_selection()

            elif choice == '4':
                self._verify_policy_files()
                continue

            else:
                print("❌ Invalid choice. Please select 1-4.")

    def _verify_policy_files(self):
        """Verify which files exist in the policy_files directory"""
        import os

        print("\n🔍 Checking policy_files directory...")
        policy_dir = "../src/policy_files"

        if not os.path.exists(policy_dir):
            print(f"❌ Directory '{policy_dir}' not found!")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Please create the policy_files directory and add your documents.")
            return False

        # Expected files
        expected_files = [
            'Customer Service.txt',
            'FAQ.txt',
            'Return Policy.txt',
            'Warranty Policy.txt',
            'Escalation Procedures.txt',
            'Technical Troubleshooting Guide.txt',
            'Business Policies and Procedures.txt',
            'Product Knowledge Database.txt',
            'Order Management and Fulfillment.txt'
        ]

        print(f"📁 Found files in {policy_dir}:")

        found_files = []
        missing_files = []

        for file in expected_files:
            file_path = os.path.join(policy_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file} ({file_size:,} bytes)")
                found_files.append(file)
            else:
                print(f"   ❌ {file} - Missing")
                missing_files.append(file)

        # Check for any additional files
        all_files = os.listdir(policy_dir)
        additional_files = [f for f in all_files if
                            f not in expected_files and f.endswith(('.txt', '.pdf', '.doc', '.docx'))]

        if additional_files:
            print(f"\n📄 Additional files found:")
            for file in additional_files:
                file_path = os.path.join(policy_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   📎 {file} ({file_size:,} bytes)")

        print(f"\n📊 Summary:")
        print(f"   ✅ Found: {len(found_files)}/9 expected files")
        print(f"   ❌ Missing: {len(missing_files)} files")

        if missing_files:
            print(f"\n💡 Missing files needed for complete functionality:")
            for file in missing_files:
                print(f"   - {file}")
            print(f"\n   Create these files to enable full scenario support.")

        if len(found_files) >= 3:
            print(f"✅ Sufficient files for basic operation!")

        return len(found_files) >= 3

    def _get_custom_file_selection(self):
        """Allow user to select specific files"""
        import os

        policy_dir = "../src/policy_files"

        if not os.path.exists(policy_dir):
            print(f"❌ Directory '{policy_dir}' not found!")
            return self._fallback_file_selection()

        # Get all supported files in policy_files directory
        supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.pptx']
        available_files = []

        for file in os.listdir(policy_dir):
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(policy_dir, file)
                file_size = os.path.getsize(file_path)
                available_files.append((file, file_size))

        if not available_files:
            print(f"❌ No supported files found in {policy_dir}/")
            return self._fallback_file_selection()

        print(f"\n📁 Available files in {policy_dir}:")
        for i, (file, size) in enumerate(available_files, 1):
            print(f"   {i:2d}. {file} ({size:,} bytes)")

        print(f"\n🔧 Custom File Selection:")
        print(f"   • Enter file numbers separated by commas (e.g., 1,2,3)")
        print(f"   • Enter 'all' to select all available files")
        print(f"   • Enter 'recommended' for the core 9 documents")
        print(f"   • Enter 'back' to return to main menu")

        while True:
            selection = input("\nYour selection: ").strip().lower()

            if selection == 'back':
                return None

            elif selection == 'all':
                selected_files = [os.path.join(policy_dir, file) for file, _ in available_files]
                print(f"✅ Selected all {len(selected_files)} files")
                return selected_files

            elif selection == 'recommended':
                # Try to find the recommended 9 files
                recommended = [
                    'Customer Service.txt',
                    'FAQ.txt',
                    'Return Policy.txt',
                    'Warranty Policy.txt',
                    'Escalation Procedures.txt',
                    'Technical Troubleshooting Guide.txt',
                    'Business Policies and Procedures.txt',
                    'Product Knowledge Database.txt',
                    'Order Management and Fulfillment.txt'
                ]

                selected_files = []
                for rec_file in recommended:
                    file_path = os.path.join(policy_dir, rec_file)
                    if os.path.exists(file_path):
                        selected_files.append(file_path)
                    else:
                        print(f"⚠️  Recommended file not found: {rec_file}")

                if selected_files:
                    print(f"✅ Selected {len(selected_files)} recommended files")
                    return selected_files
                else:
                    print("❌ No recommended files found")
                    continue

            else:
                # Parse comma-separated numbers
                try:
                    indices = [int(x.strip()) for x in selection.split(',')]
                    selected_files = []

                    for idx in indices:
                        if 1 <= idx <= len(available_files):
                            file_name = available_files[idx - 1][0]
                            file_path = os.path.join(policy_dir, file_name)
                            selected_files.append(file_path)
                        else:
                            print(f"❌ Invalid file number: {idx}")
                            break
                    else:
                        if selected_files:
                            print(f"✅ Selected {len(selected_files)} files:")
                            for file_path in selected_files:
                                file_name = os.path.basename(file_path)
                                print(f"   - {file_name}")
                            return selected_files

                except ValueError:
                    print("❌ Invalid format. Use comma-separated numbers (e.g., 1,2,3)")

    def _fallback_file_selection(self):
        """Fallback when policy_files directory doesn't exist"""
        print("\n⚠️  Fallback Mode: Looking for files in current directory")

        fallback_files = [
            'Customer Service.txt',
            'FAQ.txt',
            'Return Policy.txt'
        ]

        existing_files = []
        for file in fallback_files:
            if os.path.exists(file):
                existing_files.append(file)
                print(f"   ✅ Found: {file}")
            else:
                print(f"   ❌ Missing: {file}")

        if existing_files:
            print(f"\n💡 Using {len(existing_files)} files from current directory")
            return existing_files
        else:
            print(f"\n❌ No support files found!")
            print(f"   Please create the policy_files directory with support documents.")
            return None

    def validate_support_files(self, files):
        """Validate that all support files exist"""
        print(f"\n🔍 Validating {len(files)} support file(s)...")

        missing_files = []
        valid_files = []

        supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.pptx']

        for file in files:
            # Check if file exists
            if not os.path.exists(file):
                missing_files.append(file)
                continue

            # Check if file has supported extension
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in supported_extensions:
                print(f"⚠️  Warning: {file} has unsupported format {file_ext}")
                print(f"   Supported formats: {', '.join(supported_extensions)}")
                continue

            valid_files.append(file)
            print(f"✅ Found: {file}")

        if missing_files:
            print(f"\n❌ Missing files:")
            for file in missing_files:
                print(f"   - {file}")
            print(f"\n💡 Please ensure all required files are in the current directory:")
            print(f"   Current directory: {os.getcwd()}")
            return None

        if not valid_files:
            print("❌ No valid support files found!")
            return None

        print(f"✅ All {len(valid_files)} support file(s) validated successfully!")
        return valid_files

    # UPDATE the setup_support_documents method to use config:
    def setup_support_documents(self):
        """Setup vector index for customer support documents with configurable chunk size"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("📚 Setting up support documents...")
        logger.info("Setting up support documents with configurable parameters")

        try:
            # Get support files from user
            requested_files = self.get_support_files_from_user()

            # Validate files exist
            valid_files = self.validate_support_files(requested_files)
            if not valid_files:
                print("❌ Required support files not found. Exiting program.")
                logger.error("Required support files not found")
                return False

            print(f"\n📁 Loading {len(valid_files)} support document(s):")
            for file in valid_files:
                file_size = os.path.getsize(file)
                print(f"   - {file} ({file_size:,} bytes)")

            # Setup vector index with configurable parameters
            try:
                support_docs = SimpleDirectoryReader(input_files=valid_files).load_data()
                print(f"📄 Loaded {len(support_docs)} document(s)")
                logger.info(f"Loaded {len(support_docs)} support documents")

                # Use configuration for chunking parameters
                splitter = SentenceSplitter(
                    chunk_size=config.chunk_size,      # From .env
                    chunk_overlap=config.chunk_overlap  # From .env
                )
                support_nodes = splitter.get_nodes_from_documents(support_docs)
                print(f"🔧 Created {len(support_nodes)} text chunks (size: {config.chunk_size}, overlap: {config.chunk_overlap})")
                logger.info(f"Created {len(support_nodes)} text chunks with configurable parameters")

                self.support_index = VectorStoreIndex(support_nodes)

                print(f"✅ Support documents indexed successfully!")
                print(f"   Documents: {len(valid_files)}")
                print(f"   Chunks: {len(support_nodes)}")
                print(f"   Chunk Size: {config.chunk_size} (configurable in .env)")
                print(f"   Ready for vector search!")
                logger.info("Support documents indexed successfully")

                return True

            except Exception as e:
                print(f"❌ Error processing support documents: {e}")
                logger.error(f"Error processing support documents: {e}")
                return False

        except Exception as e:
            print(f"❌ Support documents setup failed: {e}")
            logger.error(f"Support documents setup failed: {e}")
            return False

    def create_tools(self):
        """Create enhanced tools for the agent with call graph tracking"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("🛠️  Creating enhanced agent tools with call graph tracking...")
        try:
            # Create regular database-connected tools
            self.sync_tools = CustomerServiceTools(self.db_manager)

            # ADD THIS LINE: Wrap tools with tracking
            print("📊 Wrapping tools with call graph tracking...")
            self.sync_tools = TrackedCustomerServiceTools(self.sync_tools)

            # Create async tools for parallel operations
            self.async_tools = AsyncCustomerServiceTools(self.db_manager, self.sync_tools)


            # Specialized return policy tool
            def get_order_return_policy(order_id: int) -> str:
                """Get return policy for all items in a specific order"""
                try:
                    order_id = int(order_id)

                    # Get order items
                    items = self.sync_tools.get_order_items(order_id)
                    if not items:
                        return f"Order {order_id} not found or has no items."

                    # Get return policy for each item
                    policies = []
                    for item in items:
                        days = self.sync_tools.get_item_return_days(item)
                        policies.append(f"- {item}: {days} days return policy")

                    return f"Return policy for order {order_id}:\n" + "\n".join(policies)

                except Exception as e:
                    return f"Error getting return policy for order {order_id}: {str(e)}"

            # Parallel multi-order tool for expert scenarios
            def get_multiple_orders_sync(order_ids: str) -> Dict[str, Any]:
                """Sync wrapper for async parallel order retrieval"""
                try:
                    # Parse order IDs
                    if isinstance(order_ids, str):
                        ids = [int(x.strip()) for x in order_ids.replace(',', ' ').split()]
                    else:
                        ids = [int(order_ids)]

                    # Use async method with asyncio.run for sync compatibility
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(
                            self.async_tools.get_multiple_orders_parallel(ids)
                        )
                        return {
                            'orders_found': len(results),
                            'orders': results
                        }
                    finally:
                        loop.close()

                except Exception as e:
                    return {"error": f"Failed to get multiple orders: {str(e)}"}

            # Create all function tools
            order_return_policy_tool = FunctionTool.from_defaults(
                fn=get_order_return_policy,
                name="get_order_return_policy",
                description="Get return policy for all items in a specific order. Input: order_id as integer (e.g., 1001)"
            )

            multiple_orders_parallel_tool = FunctionTool.from_defaults(
                fn=get_multiple_orders_sync,
                name="get_multiple_orders_parallel",
                description="Get details for multiple orders in parallel (faster). Input: order_ids as comma-separated string (e.g., '1007,1017,1023')"
            )

            # Basic function tools
            order_item_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_order_items,
                name="get_order_items",
                description="Get list of items in a specific order. Input: order_id as integer (e.g., 1001)"
            )

            delivery_date_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_delivery_date,
                name="get_delivery_date",
                description="Get delivery date for a specific order. Input: order_id as integer (e.g., 1001)"
            )

            return_policy_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_item_return_days,
                name="get_item_return_days",
                description="Get return policy days for a specific product name. Input: item as string (e.g., 'Laptop' or 'Mouse')"
            )

            order_details_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.get_order_details,
                name="get_order_details",
                description="Get comprehensive details for a specific order including customer info. Input: order_id as integer (e.g., 1001)"
            )

            search_orders_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.search_orders_by_customer,
                name="search_orders_by_customer",
                description="Search all orders for a customer by email address. Input: email as string (e.g., 'john.smith@email.com')"
            )

            # Advanced analytics tools
            comprehensive_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_customer_orders_comprehensive,
                name="analyze_customer_orders_comprehensive",
                description="Comprehensive analysis of all customer orders with return eligibility. Input: customer_email as string"
            )

            return_calculation_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.calculate_return_policy_and_deadlines,
                name="calculate_return_policy_and_deadlines",
                description="Calculate exact return deadlines and policy details for a specific order. Input: order_id as integer"
            )

            geographic_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_geographic_performance,
                name="analyze_geographic_performance",
                description="Analyze order patterns and performance by geographic location. Input: state as string (optional), city as string (optional)"
            )

            predictive_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.generate_predictive_risk_analysis,
                name="generate_predictive_risk_analysis",
                description="Generate predictive analysis for customer service risks and proactive recommendations. No input required."
            )

            status_analysis_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_orders_by_status_and_timeframe,
                name="analyze_orders_by_status_and_timeframe",
                description="Analyze orders by status within timeframe with delivery performance metrics. Input: start_date as string (optional), end_date as string (optional)"
            )

            product_performance_tool = FunctionTool.from_defaults(
                fn=self.sync_tools.analyze_product_performance_and_issues,
                name="analyze_product_performance_and_issues",
                description="Analyze product performance, popularity, and identify potential issues. Input: product_category as string (optional)"
            )

            # Support document tool
            support_tool = QueryEngineTool.from_defaults(
                query_engine=self.support_index.as_query_engine(),
                name="support_policy_search",
                description="Search customer support policies, return policies, FAQ, and contact information from support documents"
            )

            self.tools = [
                # Basic tools (scenarios 1-3) - prioritize specialized tools
                order_return_policy_tool,  # Specialized for return policy queries
                order_item_tool,
                delivery_date_tool,
                return_policy_tool,
                order_details_tool,
                search_orders_tool,

                # Parallel processing tools for expert scenarios
                multiple_orders_parallel_tool,  # NEW: Faster for multi-order queries

                # Advanced analytics tools (scenarios 4-15)
                comprehensive_analysis_tool,
                return_calculation_tool,
                geographic_analysis_tool,
                predictive_analysis_tool,
                status_analysis_tool,
                product_performance_tool,

                # Support document search
                support_tool
            ]

            print(f"✅ Created {len(self.tools)} enhanced tools successfully!")
            print("   📊 Basic tools: 6 (including specialized return policy tool)")
            print("   ⚡ Parallel tools: 1 (for multi-order scenarios)")
            print("   🚀 Advanced analytics: 6 (comprehensive analysis)")
            print("   📚 Support search: 1 (policy and FAQ)")
            print(f"✅ Created {len(self.tools)} enhanced tools with call graph tracking!")
            return True

        except Exception as e:
            print(f"❌ Tool creation failed: {e}")
            return False

    # def create_agent(self):
    #     """Create the enhanced AI agent with configuration from .env"""
    #     if not LLAMAINDEX_AVAILABLE or not self.tools:
    #         return False
    #
    #     print("🤖 Creating enhanced AI agent with environment configuration...")
    #     logger.info("Creating AI agent with configurable parameters")
    #
    #     try:
    #         # Setup the Agent worker with configurable settings
    #         agent_worker = ReActAgentWorker.from_tools(
    #             self.tools,
    #             llm=Settings.llm,
    #             verbose=config.agent_verbose,                    # From .env
    #             max_iterations=config.max_iterations,            # From .env
    #             allow_parallel_tool_calls=config.agent_allow_parallel_tool_calls  # From .env
    #         )
    #
    #         # Create agent runner with configurable settings
    #         self.agent = AgentRunner(
    #             agent_worker,
    #             memory=None if not config.agent_memory_enabled else "buffer",  # From .env
    #             verbose=config.agent_verbose                     # From .env
    #         )
    #
    #         print("✅ Enhanced AI agent created successfully!")
    #         print(f"   🔧 Max iterations: {config.max_iterations} (configurable in .env)")
    #         print(f"   🧠 Verbose mode: {'Enabled' if config.agent_verbose else 'Disabled'}")
    #         print(f"   📝 Memory: {'Enabled' if config.agent_memory_enabled else 'Disabled'}")
    #         print(f"   ⚡ Parallel tools: {'Enabled' if config.agent_allow_parallel_tool_calls else 'Disabled'}")
    #
    #         logger.info(f"AI agent created with max_iterations={config.max_iterations}, verbose={config.agent_verbose}")
    #         return True
    #
    #     except Exception as e:
    #         print(f"❌ Agent creation failed: {e}")
    #         logger.error(f"Agent creation failed: {e}")
    #         return False

    def create_agent(self):
        """Create the enhanced AI agent with configuration from .env"""
        if not LLAMAINDEX_AVAILABLE or not self.tools:
            return False

        print("🤖 Creating enhanced AI agent with environment configuration...")
        logger.info("Creating AI agent with configurable parameters")

        try:
            # Setup the Agent worker with configurable settings
            agent_worker = ReActAgentWorker.from_tools(
                self.tools,
                llm=Settings.llm,
                verbose=config.agent_verbose,                    # From .env
                max_iterations=config.max_iterations,            # From .env
                allow_parallel_tool_calls=config.agent_allow_parallel_tool_calls  # From .env
            )

            # Create memory object if enabled
            memory = None
            if config.agent_memory_enabled:
                try:
                    from llama_index.core.memory import ChatMemoryBuffer
                    memory = ChatMemoryBuffer.from_defaults(
                        token_limit=2000,  # Reasonable token limit for memory
                        chat_store=None    # Use default in-memory store
                    )
                    logger.info("Agent memory enabled with ChatMemoryBuffer")
                except ImportError:
                    logger.warning("ChatMemoryBuffer not available, disabling memory")
                    memory = None
                except Exception as e:
                    logger.warning(f"Failed to create memory buffer: {e}, disabling memory")
                    memory = None

            # Create agent runner with configurable settings
            self.agent = AgentRunner(
                agent_worker,
                memory=memory,  # Now properly using memory object or None
                verbose=config.agent_verbose
            )

            print("✅ Enhanced AI agent created successfully!")
            print(f"   🔧 Max iterations: {config.max_iterations} (configurable in .env)")
            print(f"   🧠 Verbose mode: {'Enabled' if config.agent_verbose else 'Disabled'}")
            print(f"   📝 Memory: {'Enabled' if memory is not None else 'Disabled'}")
            print(f"   ⚡ Parallel tools: {'Enabled' if config.agent_allow_parallel_tool_calls else 'Disabled'}")

            logger.info(f"AI agent created with max_iterations={config.max_iterations}, verbose={config.agent_verbose}")
            return True

        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
            logger.error(f"Agent creation failed: {e}")
            return False

    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Customer Service AI Agent...")
        print("=" * 60)

        steps = [
            ("Database Connection", self.setup_database),
            ("LLM Setup", self.setup_llm),
            ("Support Documents", self.setup_support_documents),
            ("Agent Tools", self.create_tools),
            ("AI Agent", self.create_agent)
        ]

        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            if not step_func():
                print(f"❌ Failed at step: {step_name}")
                return False

        # Setup query decomposer
        print("\n📋 Step: Query Decomposition System")
        if self.setup_query_decomposer():
            print("✅ Query decomposition enabled!")
        else:
            print("⚠️  Query decomposition setup failed, using standard mode")

        print("\n" + "=" * 60)
        print("🎉 Customer Service AI Agent initialized successfully!")
        print("🚀 Performance optimizations active:")
        print("   ⚡ Parallel processing for multi-order scenarios")
        print("   🔧 Connection pooling for faster database queries")
        print("   💾 GPU acceleration for embeddings (if available)")
        return True

    def run_predefined_scenarios(self):
        """Run predefined test scenarios"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        scenarios = [
            # Basic scenarios (original)
            {
                "name": "Return Policy Query",
                "query": "What is the return policy for order number 1001?",
                "description": "Tests order lookup and return policy retrieval",
                "difficulty": "Basic"
            },
            {
                "name": "Multi-part Question",
                "query": "When is the delivery date and items shipped for order 1003 and how can I contact customer support?",
                "description": "Tests multiple tool usage in single query",
                "difficulty": "Basic"
            },
            {
                "name": "Invalid Order",
                "query": "What is the return policy for order number 9999?",
                "description": "Tests handling of non-existent orders",
                "difficulty": "Basic"
            },

            # Intermediate scenarios
            {
                "name": "Customer History Analysis",
                "query": "Show me all orders for customer john.smith@email.com and tell me which items can still be returned based on delivery dates",
                "description": "Tests customer search, multiple orders, date calculations, and policy application",
                "difficulty": "Intermediate"
            },
            {
                "name": "Product Category Return Policy",
                "query": "I bought a laptop, mouse, and HDMI cable in different orders. What are the return policies for each and which one expires first?",
                "description": "Tests product search across orders, policy comparison, and time analysis",
                "difficulty": "Intermediate"
            },
            {
                "name": "Order Status Investigation",
                "query": "Why hasn't order 1005 been delivered yet? Check the status and compare with similar orders from the same timeframe",
                "description": "Tests order details, status analysis, and comparative investigation",
                "difficulty": "Intermediate"
            },
            {
                "name": "Cross-Reference Analysis",
                "query": "Find all customers who ordered laptops in June 2024 and tell me their order statuses and delivery performance",
                "description": "Tests complex joins, date filtering, and batch analysis",
                "difficulty": "Intermediate"
            },

            # Advanced scenarios
            {
                "name": "Complex Return Calculation",
                "query": "For order 1022, calculate the exact return deadline for each item considering the delivery date was June 6th, and tell me what happens if I return only the mouse but keep the laptop",
                "description": "Tests date arithmetic, partial returns, policy calculations, and business logic",
                "difficulty": "Advanced"
            },
            {
                "name": "Customer Lifecycle Analysis",
                "query": "Analyze the complete order history for customers in Auburn, AL. Which products are most popular, what's the average order value, and identify any delivery issues",
                "description": "Tests geographic filtering, statistical analysis, trend identification, and problem detection",
                "difficulty": "Advanced"
            },
            {
                "name": "Escalation Scenario",
                "query": "Customer placed order 1013 on June 13th for a gaming laptop costing $1599.99 but it's still pending after a week. They're threatening to cancel. What are our options for resolution and how should we escalate this?",
                "description": "Tests crisis management, order prioritization, escalation procedures, and solution generation",
                "difficulty": "Advanced"
            },
            {
                "name": "Inventory Impact Analysis",
                "query": "If all pending orders from the last week need to be expedited due to customer complaints, which products would be affected and what's the total value at risk?",
                "description": "Tests order aggregation, financial calculations, business impact analysis, and risk assessment",
                "difficulty": "Advanced"
            },

            # Expert scenarios
            {
                "name": "Multi-Customer Dispute Resolution",
                "query": "Three customers (orders 1007, 1017, 1023) all received MacBook Pros but are reporting different issues: one has screen problems, one has battery issues, one has keyboard problems. Analyze their orders, determine warranty coverage, and recommend resolution strategy for each",
                "description": "Tests multi-order analysis, issue categorization, warranty determination, and personalized solutions",
                "difficulty": "Expert"
            },
            {
                "name": "Supply Chain Investigation",
                "query": "Several customers are complaining about late deliveries for orders placed in early June. Analyze delivery performance for orders 1001-1015, identify patterns, and determine if there's a systemic issue with specific product categories or shipping regions",
                "description": "Tests pattern recognition, performance analysis, root cause identification, and systemic problem detection",
                "difficulty": "Expert"
            },
            {
                "name": "Revenue Recovery Strategy",
                "query": "Customer wants to return their entire order 1007 ($2129.98) due to compatibility issues, but they've already used the laptop for 2 weeks. Analyze the order details, check our return policy exceptions, calculate potential restocking fees, and propose alternatives that minimize revenue loss while satisfying the customer",
                "description": "Tests policy interpretation, financial calculations, alternative solution generation, and business optimization",
                "difficulty": "Expert"
            },
            {
                "name": "Predictive Customer Service",
                "query": "Based on order patterns, delivery dates, and customer behavior, identify which current customers are most likely to have issues or complaints in the next week, and proactively suggest what we should do to prevent problems",
                "description": "Tests predictive analysis, risk modeling, proactive service, and prevention strategies",
                "difficulty": "Expert"
            }
        ]

        print("\n🎯 Available Predefined Scenarios:")
        print("=" * 60)

        # Group scenarios by difficulty
        difficulty_groups = {}
        for scenario in scenarios:
            difficulty = scenario['difficulty']
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(scenario)

        # Display scenarios grouped by difficulty
        scenario_index = 1
        for difficulty in ['Basic', 'Intermediate', 'Advanced', 'Expert']:
            if difficulty in difficulty_groups:
                print(f"\n🔸 {difficulty.upper()} SCENARIOS:")
                for scenario in difficulty_groups[difficulty]:
                    print(f"{scenario_index:2d}. {scenario['name']}")
                    print(f"    Query: '{scenario['query'][:80]}{'...' if len(scenario['query']) > 80 else ''}'")
                    print(f"    Description: {scenario['description']}")
                    print()
                    scenario_index += 1

        while True:
            print(f"💡 Options:")
            print(f"   • Enter scenario number (1-{len(scenarios)})")
            print(f"   • 'basic' for scenarios 1-3")
            print(f"   • 'intermediate' for scenarios 4-7")
            print(f"   • 'advanced' for scenarios 8-11")
            print(f"   • 'expert' for scenarios 12-15")
            print(f"   • 'all' for all scenarios")
            print(f"   • 'random' for a random challenging scenario")
            print(f"   • 'back' to return to main menu")

            choice = input("\nSelect option: ").strip().lower()

            if choice == 'back':
                break
            elif choice == 'all':
                for i, scenario in enumerate(scenarios, 1):
                    print(f"\n🔍 Running Scenario {i}: {scenario['name']} ({scenario['difficulty']})")
                    print("=" * 70)
                    self.run_query_with_options(scenario['query'])
                    if i < len(scenarios):
                        cont = input(
                            f"\nPress Enter to continue to next scenario (or 'stop' to finish): ").strip().lower()
                        if cont == 'stop':
                            break
                break
            elif choice == 'basic':
                self._run_difficulty_scenarios(scenarios, 'Basic')
                break
            elif choice == 'intermediate':
                self._run_difficulty_scenarios(scenarios, 'Intermediate')
                break
            elif choice == 'advanced':
                self._run_difficulty_scenarios(scenarios, 'Advanced')
                break
            elif choice == 'expert':
                self._run_difficulty_scenarios(scenarios, 'Expert')
                break
            elif choice == 'random':
                import random
                # Bias towards more challenging scenarios
                weights = [1 if s['difficulty'] == 'Basic' else
                           2 if s['difficulty'] == 'Intermediate' else
                           3 if s['difficulty'] == 'Advanced' else 4
                           for s in scenarios]
                scenario = random.choices(scenarios, weights=weights)[0]
                print(f"\n🎲 Random Scenario: {scenario['name']} ({scenario['difficulty']})")
                print("=" * 70)
                self.run_query_with_options(scenario['query'])
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(scenarios):
                    scenario = scenarios[idx]
                    print(f"\n🔍 Running Scenario: {scenario['name']} ({scenario['difficulty']})")
                    print("=" * 70)
                    self.run_query_with_options(scenario['query'])
                    break
                else:
                    print(f"❌ Invalid scenario number. Please select 1-{len(scenarios)}.")
            else:
                print("❌ Invalid choice. Please try again.")

    def _run_difficulty_scenarios(self, scenarios, difficulty):
        """Run all scenarios of a specific difficulty level"""
        difficulty_scenarios = [s for s in scenarios if s['difficulty'] == difficulty]

        print(f"\n🎯 Running All {difficulty.upper()} Scenarios ({len(difficulty_scenarios)} scenarios)")
        print("=" * 60)

        for i, scenario in enumerate(difficulty_scenarios, 1):
            print(f"\n🔍 {difficulty} Scenario {i}: {scenario['name']}")
            print("=" * 50)
            self.run_query_with_options(scenario['query'])

            if i < len(difficulty_scenarios):
                cont = input(
                    f"\nPress Enter to continue to next {difficulty.lower()} scenario (or 'stop' to finish): ").strip().lower()
                if cont == 'stop':
                    break

    def run_custom_query(self):
        """Enhanced custom query with call graph tracking"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        print("\n💬 Custom Query Mode with Call Graph Tracking")
        print("=" * 50)
        print("You can ask questions about:")
        print("- Order details (e.g., 'What items are in order 1001?')")
        print("- Delivery dates (e.g., 'When will order 1002 be delivered?')")
        print("- Return policies (e.g., 'What's the return policy for laptops?')")
        print("- Customer support information")
        print("- Multi-order analysis (e.g., 'Compare orders 1007, 1017, 1023')")
        print("- Geographic analysis (e.g., 'Orders in Auburn, AL')")
        print("- Predictive analysis (e.g., 'Which customers might have issues?')")
        print("- Any combination of the above")
        print("\n🎯 Each query will generate an interactive call graph showing:")
        print("   • Thinking processes  • Tool selections  • Database operations")
        print("   • Cache hits/misses  • Logical reasoning  • Error handling")
        print("\nSpecial commands:")
        print("- 'demo cache' - Demonstrate cache behavior")
        print("- 'back' - Return to main menu")
        print()

        while True:
            query = input("🤔 Your question: ").strip()

            if query.lower() == 'back':
                break
            elif query.lower() == 'demo cache':
                self.demonstrate_cache_behavior()
                print("\n" + "-" * 50)
            elif query:
                # Offer execution options
                print(f"\n📋 Execution options for: '{query[:60]}{'...' if len(query) > 60 else ''}'")
                print("1. 🚀 Standard execution (with call graph)")
                print("2. 📊 Enhanced tracking (detailed tool usage + call graph)")
                print("3. 🧩 Intelligent decomposition (complex query analysis + call graph)")

                exec_choice = input("Select execution method (1-3, default=1): ").strip() or "1"

                if exec_choice == "1":
                    self.run_query(query)
                elif exec_choice == "2":
                    self.run_query_with_enhanced_tracking(query)
                elif exec_choice == "3":
                    if hasattr(self, 'query_decomposer') and self.query_decomposer:
                        self.run_intelligent_query(query)
                    else:
                        print("⚠️  Query decomposer not available, using standard execution")
                        self.run_query(query)

                print(f"\n💡 View the call graph: Menu option 10 → Latest graph")
                print("\n" + "-" * 50)
            else:
                print("❌ Please enter a valid question.")


    @performance_monitor
    def run_query(self, query: str):
        """Execute a query using the agent with call graph tracking"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        # Start call graph tracking
        query_id = self.execution_tracker.track_agent_query_start(query)

        try:
            print(f"🤖 Processing query with call graph tracking: '{query}'")
            print(f"📊 Tracking ID: {query_id}")
            print("-" * 70)

            # Track initial agent thinking
            self.execution_tracker.add_thinking_process(
                "Received customer query, initializing response process",
                f"Query length: {len(query)} characters, analyzing requirements"
            )

            # Track tool availability assessment
            available_tools = [tool.metadata.name for tool in self.tools] if self.tools else []
            self.execution_tracker.track_tool_selection_process(available_tools, query)

            # Add timeout and retry logic with tracking
            import signal

            def timeout_handler(signum, frame):
                self.execution_tracker.add_error("Query execution timeout", "timeout")
                raise TimeoutError("Query execution timeout")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                # Track the main agent execution
                self.execution_tracker.add_thinking_process(
                    "Executing main agent query processing",
                    "Agent will now process the query using available tools and knowledge"
                )

                response = self.agent.query(query)
                signal.alarm(0)  # Cancel timeout

                # Track successful completion
                self.execution_tracker.add_thinking_process(
                    "Query processing completed successfully",
                    f"Generated response of {len(str(response))} characters"
                )

                # Finalize tracking
                final_response = str(response)
                self.execution_tracker.finalize_query(final_response, success=True)

                print("\n✅ Agent Response:")
                print("=" * 30)
                print(response)
                print()

                # Show execution summary
                summary = self.execution_tracker.export_execution_summary()
                print(f"📊 Execution Summary:")
                print(f"   Total time: {summary.get('total_execution_time', 0):.2f}s")
                print(f"   Tool calls: {summary.get('tool_calls', 0)}")
                print(f"   Cache hits: {summary.get('cache_hits', 0)}")
                print(f"   Reasoning steps: {summary.get('reasoning_steps', 0)}")
                print(f"   📈 Call graph saved to: ./generated_callgraphs/")

            except TimeoutError:
                signal.alarm(0)
                self.execution_tracker.add_error("Query timeout exceeded", "timeout")
                self.execution_tracker.finalize_query("Query timed out", success=False)
                print("\n⏰ Query timeout! The query took too long to process.")

            except Exception as query_error:
                signal.alarm(0)
                self.execution_tracker.add_error(str(query_error), "execution_error")
                self.execution_tracker.finalize_query(f"Query failed: {str(query_error)}", success=False)
                print(f"\n❌ Query execution error: {query_error}")

                # Provide helpful fallback with tracking
                if "max iterations" in str(query_error).lower():
                    self.execution_tracker.add_thinking_process(
                        "Query complexity exceeded iteration limit",
                        "The query was too complex and reached the maximum iteration limit"
                    )
                    print("\n🔄 The query was too complex and reached iteration limit.")
                    print("💡 Suggestions:")
                    print("   • Break down your question into simpler parts")
                    print("   • Ask about specific order IDs or customers")
                    print("   • Try using more specific keywords")

        except Exception as e:
            self.execution_tracker.add_error(str(e), "critical_error")
            self.execution_tracker.finalize_query(f"Critical error: {str(e)}", success=False)
            print(f"❌ Critical error during query execution: {e}")
            print("🔧 Please check your database connection and try again.")

    def show_database_stats(self):
        """Show database statistics"""
        print("\n📊 Database Statistics")
        print("=" * 30)

        # Show connection info
        conn_info = self.db_manager.get_connection_info()
        print(f"Connection Type: {conn_info['type']}")
        print(f"Status: {conn_info['status']}")
        if conn_info['type'] == 'connection_pool':
            print(f"Pool Size: {conn_info.get('pool_size', 'Unknown')}")

        stats_queries = [
            ("Total Orders", "SELECT COUNT(*) FROM orders"),
            ("Total Customers", "SELECT COUNT(*) FROM customers"),
            ("Total Products", "SELECT COUNT(*) FROM products"),
            ("Orders by Status", "SELECT status, COUNT(*) FROM orders GROUP BY status"),
            ("Recent Orders (Last 10)", """
                                        SELECT order_id, customer_id, order_date, status, total_amount
                                        FROM orders
                                        ORDER BY order_date DESC LIMIT 10
                                        """)
        ]

        for stat_name, query in stats_queries:
            print(f"\n{stat_name}:")
            results = self.db_manager.execute_query(query)
            if results:
                if stat_name == "Recent Orders (Last 10)":
                    for row in results:
                        print(f"  Order {row[0]}: Customer {row[1]}, {row[2]}, {row[3]}, ${row[4]}")
                elif stat_name == "Orders by Status":
                    for row in results:
                        print(f"  {row[0]}: {row[1]} orders")
                else:
                    print(f"  {results[0][0]}")

    def health_check(self) -> Dict[str, str]:
        """Comprehensive health check for production monitoring"""
        health = {
            'database': 'unknown',
            'llm': 'unknown',
            'embeddings': 'unknown',
            'memory': 'unknown',
            'circuit_breaker': 'unknown'
        }

        # Database health
        try:
            self.db_manager.execute_query("SELECT 1")
            health['database'] = 'healthy'
        except:
            health['database'] = 'unhealthy'

        # Circuit breaker status
        if hasattr(self.db_manager, 'circuit_breaker'):
            cb_state = self.db_manager.circuit_breaker.get_state()
            health['circuit_breaker'] = cb_state['state']

        # LLM health
        try:
            Settings.llm.complete("test")
            health['llm'] = 'healthy'
        except:
            health['llm'] = 'unhealthy'

        # Memory check
        if PSUTIL_AVAILABLE:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            health['memory'] = 'healthy' if memory_mb < 1000 else 'warning'
            health['memory_usage'] = f"{memory_mb:.1f}MB"

        return health

    def show_health_check(self):
        """Display system health check"""
        print("\n🏥 System Health Check")
        print("=" * 30)

        health = self.health_check()
        for component, status in health.items():
            if component == 'memory_usage':
                continue
            icon = "✅" if status in ['healthy', 'CLOSED'] else "⚠️" if status == 'warning' else "❌"
            print(f"{icon} {component.title()}: {status}")

        if 'memory_usage' in health:
            print(f"💾 Memory Usage: {health['memory_usage']}")

    def show_performance_metrics(self):
        """Enhanced performance metrics with tool usage integration"""
        print("\n📈 Enhanced Performance Metrics")
        print("=" * 40)

        # Get basic metrics
        metrics = self.get_performance_metrics()

        # Add tool usage metrics
        tool_report = tool_tracker.get_comprehensive_report()
        cache_stats = cache_manager.get_detailed_stats()

        print(f"🔧 System Configuration:")
        for key, value in metrics.get('config', {}).items():
            print(f"   {key}: {value}")

        print(f"\n📊 Database Performance:")
        if 'database_stats' in metrics:
            stats = metrics['database_stats']
            print(f"   Total Queries: {stats['total_queries']}")
            print(f"   Success Rate: {stats['successful_queries']}/{stats['total_queries']} ({stats['successful_queries']/stats['total_queries']*100:.1f}%)")
            print(f"   Avg Query Time: {stats['avg_query_time']:.3f}s")

        print(f"\n🔧 Tool Usage Performance:")
        session = tool_report['session_summary']
        print(f"   Session Duration: {session['session_duration_minutes']:.1f} minutes")
        print(f"   Total Tool Calls: {session['total_tool_calls']}")
        print(f"   Tool Success Rate: {session['success_rate']:.1f}%")
        print(f"   Activity Rate: {session['calls_per_minute']:.1f} calls/minute")
        print(f"   Unique Tools Used: {session['unique_tools_used']}")

        print(f"\n💾 Cache Performance:")
        global_cache = cache_stats['global']
        print(f"   Overall Hit Rate: {global_cache['hit_rate']:.1f}%")
        print(f"   Cache Hits: {global_cache['cache_hits']}")
        print(f"   Cache Misses: {global_cache['cache_misses']}")
        print(f"   Time Saved: ~{global_cache['time_saved']:.1f}s")

        if 'memory' in metrics:
            mem = metrics['memory']
            print(f"\n🧠 System Resources:")
            print(f"   Memory Usage: {mem['rss_mb']:.1f}MB")
            print(f"   CPU Usage: {mem['cpu_percent']:.1f}%")
            print(f"   Active Threads: {mem['num_threads']}")

        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")

        if global_cache['hit_rate'] < 50:
            print(f"   ⚠️  Low cache hit rate ({global_cache['hit_rate']:.1f}%) - Consider longer TTL")
        elif global_cache['hit_rate'] > 80:
            print(f"   ✅ Excellent cache performance ({global_cache['hit_rate']:.1f}%)")

        if session['success_rate'] < 95:
            print(f"   ⚠️  Tool failure rate is high ({100-session['success_rate']:.1f}%) - Check error handling")

        if session['calls_per_minute'] > 10:
            print(f"   🔥 High activity rate ({session['calls_per_minute']:.1f}/min) - Monitor resource usage")

        # Show top performing tools
        if tool_report['tool_ranking']:
            print(f"\n🏆 Top 3 Most Used Tools:")
            for i, (tool_name, call_count) in enumerate(tool_report['tool_ranking'][:3], 1):
                perf = tool_report['performance_analysis'][tool_name]
                print(f"   {i}. {tool_name}: {call_count} calls ({perf['efficiency_score']:.1f}% success)")

    def clear_caches(self):
        """Enhanced cache clearing with detailed feedback"""
        print("\n🧹 Cache Management")
        print("=" * 30)

        # Show current cache state
        cache_stats = cache_manager.get_detailed_stats()
        tool_calls = len(tool_tracker.tool_calls)

        print(f"Current cache state:")
        print(f"   Total cached methods: {len(cache_stats['by_method'])}")
        print(f"   Total cache entries: {sum(stats['cache_size'] for stats in cache_stats['by_method'].values())}")
        print(f"   Cache hit rate: {cache_stats['global']['hit_rate']:.1f}%")
        print(f"   Tool call history: {tool_calls} entries")

        if cache_stats['global']['total_calls'] == 0:
            print("\n💡 No cache data to clear.")
            return

        print(f"\n🧹 Clear options:")
        print(f"1. Clear method caches only")
        print(f"2. Clear tool usage history only")
        print(f"3. Clear everything (caches + tool history)")
        print(f"4. Show cache details first")
        print(f"5. Cancel")

        choice = input("Select option (1-5): ").strip()

        if choice == '1':
            cache_manager.clear_all_caches()
            print("✅ Method caches cleared!")

        elif choice == '2':
            tool_tracker.clear_history()
            print("✅ Tool usage history cleared!")

        elif choice == '3':
            cache_manager.clear_all_caches()
            tool_tracker.clear_history()
            print("✅ All caches and tool history cleared!")

        elif choice == '4':
            self._show_cache_details(cache_stats)
            return  # Don't clear, just show details

        elif choice == '5':
            print("❌ Clear operation cancelled")
            return

        # Force garbage collection after clearing
        if PSUTIL_AVAILABLE:
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            gc.collect()
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"🧠 Garbage collection freed {mem_before - mem_after:.1f}MB")

    def reload_configuration(self):
        """Reload configuration from .env file"""
        print("\n🔄 Reloading Configuration...")

        try:
            # Reload .env file
            from dotenv import load_dotenv
            load_dotenv(override=True)  # Override existing environment variables

            # Create new config instance
            global config
            config = Config()

            print("✅ Configuration reloaded from .env file")
            logger.info("Configuration reloaded from .env file")

            # Show updated configuration
            self.show_configuration()

        except Exception as e:
            print(f"❌ Failed to reload configuration: {e}")
            logger.error(f"Failed to reload configuration: {e}")

    def show_tool_usage_report(self):
        """Display comprehensive tool usage report - NEW METHOD"""
        print("\n🔧 Tool Usage Report")
        print("=" * 60)

        report = tool_tracker.get_comprehensive_report()
        session = report['session_summary']

        # Session overview
        print(f"📊 Session Overview:")
        print(f"   Start Time: {session['session_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {session['session_duration_minutes']:.1f} minutes")
        print(f"   Total Tool Calls: {session['total_tool_calls']}")
        print(f"   Success Rate: {session['success_rate']:.1f}%")
        print(f"   Tools Used: {session['unique_tools_used']}")
        print(f"   Activity Rate: {session['calls_per_minute']:.1f} calls/minute")

        # Tool ranking
        if report['tool_ranking']:
            print(f"\n🏆 Most Used Tools:")
            for i, (tool_name, call_count) in enumerate(report['tool_ranking'][:10], 1):
                stats = report['tool_statistics'][tool_name]
                perf = report['performance_analysis'][tool_name]
                print(f"   {i:2d}. {tool_name}")
                print(f"       Calls: {call_count} | Success: {perf['efficiency_score']:.1f}% | Speed: {perf['speed_category']}")
                print(f"       Avg Time: {stats['avg_execution_time']:.3f}s | Usage: {perf['usage_frequency']:.1f}/min")

        # Recent activity
        if report['recent_activity']:
            print(f"\n📋 Recent Tool Activity (Last 10 calls):")
            for i, call in enumerate(reversed(report['recent_activity']), 1):
                status = "✅" if call['success'] else "❌"
                print(f"   {i:2d}. {status} {call['tool_name']} ({call['execution_time']:.3f}s)")
                print(f"       Input: {call['input_data']}")
                print(f"       Result: {call['result_summary']}")
                print(f"       Context: {call['caller_context']}")
                print(f"       Time: {call['timestamp'].strftime('%H:%M:%S')}")
                print()

        # Cache performance
        cache_stats = cache_manager.get_detailed_stats()
        print(f"\n💾 Cache Performance:")
        print(f"   Global Hit Rate: {cache_stats['global']['hit_rate']:.1f}%")
        print(f"   Total Calls: {cache_stats['global']['total_calls']}")
        print(f"   Cache Hits: {cache_stats['global']['cache_hits']}")
        print(f"   Time Saved: {cache_stats['global']['time_saved']:.1f}s")

        if cache_stats['by_method']:
            print(f"\n📈 Cache Performance by Method:")
            for method, stats in cache_stats['by_method'].items():
                print(f"   {method}:")
                print(f"     Hit Rate: {stats['hit_rate']:.1f}% ({stats['cache_hits']}/{stats['total_calls']})")
                print(f"     Avg Time: {stats['avg_execution_time']:.3f}s | Cache Size: {stats['cache_size']}")

        # Query patterns
        print(f"\n🔍 Query Pattern Analysis:")

        # Analyze tool usage patterns
        tool_combinations = {}
        if len(report['detailed_call_history']) > 1:
            for i in range(len(report['detailed_call_history']) - 1):
                current_tool = report['detailed_call_history'][i]['tool_name']
                next_tool = report['detailed_call_history'][i + 1]['tool_name']
                combo = f"{current_tool} → {next_tool}"
                tool_combinations[combo] = tool_combinations.get(combo, 0) + 1

        if tool_combinations:
            print(f"   Common Tool Sequences:")
            sorted_combos = sorted(tool_combinations.items(), key=lambda x: x[1], reverse=True)
            for combo, count in sorted_combos[:5]:
                print(f"     {combo}: {count} times")

        # Performance insights
        print(f"\n💡 Performance Insights:")

        slow_tools = [name for name, perf in report['performance_analysis'].items()
                      if perf['speed_category'] == 'Slow']
        if slow_tools:
            print(f"   ⚠️  Slow tools detected: {', '.join(slow_tools)}")
            print(f"      Consider optimizing these tools or checking database performance")

        high_usage_tools = [name for name, perf in report['performance_analysis'].items()
                            if perf['usage_frequency'] > 5]  # More than 5 calls per minute
        if high_usage_tools:
            print(f"   🔥 High-usage tools: {', '.join(high_usage_tools)}")
            print(f"      These tools are frequently used - ensure they're optimized")

        low_success_tools = [name for name, perf in report['performance_analysis'].items()
                             if perf['efficiency_score'] < 80]
        if low_success_tools:
            print(f"   ❌ Tools with low success rate: {', '.join(low_success_tools)}")
            print(f"      These tools often fail - check error handling and input validation")

        print(f"\n📝 Recommendations:")
        print(f"   • Cache hit rate target: >70% (current: {cache_stats['global']['hit_rate']:.1f}%)")
        print(f"   • Tool success rate target: >95% (current: {session['success_rate']:.1f}%)")
        print(f"   • Consider optimizing tools with >2s average execution time")

        # Option to show detailed call history
        print(f"\n🔧 Options:")
        print(f"   1. Show detailed call history")
        print(f"   2. Show cache details")
        print(f"   3. Export report to file")
        print(f"   4. Clear tool usage history")
        print(f"   5. Return to main menu")

        choice = input("Select option (1-5): ").strip()

        if choice == '1':
            self._show_detailed_call_history(report['detailed_call_history'])
        elif choice == '2':
            self._show_cache_details(cache_stats)
        elif choice == '3':
            self._export_tool_report(report)
        elif choice == '4':
            self._clear_tool_history()
        # Option 5 or other: return to main menu

    def _show_detailed_call_history(self, call_history: List[Dict]):
        """Show detailed call history"""
        print(f"\n📋 Detailed Call History ({len(call_history)} calls)")
        print("=" * 80)

        for i, call in enumerate(call_history, 1):
            status = "✅ SUCCESS" if call['success'] else "❌ FAILED"
            print(f"\n{i:3d}. [{call['timestamp'].strftime('%H:%M:%S')}] {status}")
            print(f"     Tool: {call['tool_name']}")
            print(f"     Input: {call['input_data']}")
            print(f"     Result: {call['result_summary']}")
            print(f"     Time: {call['execution_time']:.3f}s")
            print(f"     Context: {call['caller_context']}")
            print(f"     Reasoning: {call['reasoning']}")

            if i % 10 == 0 and i < len(call_history):
                cont = input(f"\nShowing {i}/{len(call_history)} calls. Continue? (y/n): ").strip().lower()
                if cont != 'y':
                    break

    def _show_cache_details(self, cache_stats: Dict):
        """Show detailed cache information"""
        print(f"\n💾 Detailed Cache Information")
        print("=" * 50)

        print(f"Global Cache Statistics:")
        global_stats = cache_stats['global']
        for key, value in global_stats.items():
            if key != 'methods_tracked':
                print(f"   {key.replace('_', ' ').title()}: {value}")

        print(f"\nMethod-Specific Cache Performance:")
        for method, stats in cache_stats['by_method'].items():
            print(f"\n📊 {method}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"   {key.replace('_', ' ').title()}: {value}")

        if cache_stats['recent_calls']:
            print(f"\n📋 Recent Cache Activity:")
            for call in cache_stats['recent_calls'][-10:]:
                hit_status = "🎯 HIT" if call['hit'] else "❌ MISS"
                print(f"   {hit_status} {call['function']} | {call['execution_time']:.3f}s | {call['source']}")

    def _export_tool_report(self, report: Dict):
        """Export tool usage report to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tool_usage_report_{timestamp}.json"

            # Convert datetime objects to strings for JSON serialization
            serializable_report = self._make_json_serializable(report)

            with open(filename, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)

            print(f"✅ Report exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export report: {e}")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj

    def _clear_tool_history(self):
        """Clear all tool usage history"""
        print(f"\n🧹 Clear Tool Usage History")
        print("=" * 40)

        current_calls = len(tool_tracker.tool_calls)
        print(f"Current history contains {current_calls} tool calls")

        if current_calls == 0:
            print("No history to clear.")
            return

        confirm = input("Are you sure you want to clear all tool usage history? (y/n): ").strip().lower()

        if confirm == 'y':
            tool_tracker.clear_history()
            cache_manager.clear_all_caches()
            print("✅ Tool usage history and caches cleared!")
            print("📊 Statistics reset - starting fresh session")
        else:
            print("❌ Clear operation cancelled")

    def _execute_simple_query(self, query: str) -> str:
        """Execute a simple query using the agent - required by QueryDecomposer"""
        try:
            # Track the simple query execution if tracker is available
            if hasattr(self, 'execution_tracker'):
                self.execution_tracker.add_thinking_process(
                    f"Executing simple query: {query[:100]}...",
                    "Processing subgoal through agent query interface"
                )

            response = self.agent.query(query)
            return str(response)

        except Exception as e:
            error_msg = f"Simple query execution failed: {e}"

            # Track the error if tracker is available
            if hasattr(self, 'execution_tracker'):
                self.execution_tracker.add_error(error_msg, "simple_query_error")

            return error_msg

    def _synthesize_results(self, original_query: str, results: List[Dict]) -> str:
        """Combine results from subgoals into comprehensive response - required by QueryDecomposer"""
        try:
            print(f"\n🔄 Synthesizing results from {len(results)} subgoals...")

            # Track synthesis start if tracker is available
            if hasattr(self, 'execution_tracker'):
                self.execution_tracker.add_thinking_process(
                    f"Starting result synthesis for {len(results)} subgoal results",
                    f"Original query: {original_query[:100]}..."
                )

            # Separate results by type
            data_results = [r for r in results if r['type'] == 'data_collection']
            analysis_results = [r for r in results if r['type'] == 'analysis']
            synthesis_results = [r for r in results if r['type'] == 'synthesis']

            # Build comprehensive response
            final_response = f"## Comprehensive Analysis for: {original_query}\n\n"

            if data_results:
                final_response += "### 📊 Data Collection Results:\n"
                for i, result in enumerate(data_results, 1):
                    final_response += f"{i}. **{result['subgoal']}**\n{result['result']}\n\n"

            if analysis_results:
                final_response += "### 🔍 Analysis Results:\n"
                for i, result in enumerate(analysis_results, 1):
                    final_response += f"{i}. **{result['subgoal']}**\n{result['result']}\n\n"

            if synthesis_results:
                final_response += "### 💡 Recommendations & Conclusions:\n"
                for i, result in enumerate(synthesis_results, 1):
                    final_response += f"{i}. **{result['subgoal']}**\n{result['result']}\n\n"

            # Add summary
            final_response += "### 📋 Executive Summary:\n"
            final_response += f"Successfully analyzed complex query through {len(results)} focused subgoals. "
            final_response += "Each aspect was thoroughly examined to provide comprehensive insights and actionable recommendations."

            # Track synthesis completion if tracker is available
            if hasattr(self, 'execution_tracker'):
                self.execution_tracker.add_logical_inference(
                    "Result synthesis completed successfully",
                    [f"Processed {len(results)} subgoal results"],
                    f"Generated comprehensive response with {len(final_response)} characters"
                )

            return final_response

        except Exception as e:
            error_msg = f"Result synthesis failed: {e}"

            # Track synthesis error if tracker is available
            if hasattr(self, 'execution_tracker'):
                self.execution_tracker.add_error(error_msg, "synthesis_error")

            # Return a fallback response
            fallback_response = f"## Analysis Results for: {original_query}\n\n"
            fallback_response += "⚠️ Result synthesis encountered an error, but here are the individual subgoal results:\n\n"

            for i, result in enumerate(results, 1):
                fallback_response += f"### Subgoal {i}: {result['subgoal']}\n"
                fallback_response += f"**Result:** {result['result']}\n\n"

            return fallback_response

    def main_menu(self):
        """Main interactive menu with call graph visualization"""
        if not self.initialize():
            print("❌ Initialization failed. Exiting.")
            return

        while True:
            print("\n" + "=" * 60)
            print("🤖 CUSTOMER SERVICE AI AGENT WITH CALL GRAPH TRACKING")
            print("=" * 60)
            print("1. 🎯 Run Predefined Scenarios")
            print("2. 💬 Custom Query")
            print("3. 📊 Database Statistics")
            print("4. 🏥 Health Check")
            print("5. 📈 Performance Metrics")
            print("6. 🧹 Clear Caches")
            print("7. ⚙️  Show Configuration")
            print("8. 🔄 Reload Configuration")
            print("9. 🔧 Tool Usage Report")
            print("10. 📊 View Call Graphs")         # NEW OPTION
            print("11. 📈 Call Graph Statistics")    # NEW OPTION
            print("12. 🚪 Exit")
            print("=" * 60)

            choice = input("Select an option (1-12): ").strip()

            if choice == '1':
                self.run_predefined_scenarios()
            elif choice == '2':
                self.run_custom_query()
            elif choice == '3':
                self.show_database_stats()
            elif choice == '4':
                self.show_health_check()
            elif choice == '5':
                self.show_performance_metrics()
            elif choice == '6':
                self.clear_caches()
            elif choice == '7':
                self.show_configuration()
            elif choice == '8':
                self.reload_configuration()
            elif choice == '9':
                self.show_tool_usage_report()
            elif choice == '10':              # NEW: View call graphs
                self.view_call_graphs()
            elif choice == '11':              # NEW: Call graph statistics
                self.show_call_graph_stats()
            elif choice == '12':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-12.")

        # Cleanup
        print("🧹 Cleaning up resources...")
        self.db_manager.disconnect()
        if hasattr(self, 'async_tools') and self.async_tools:
            self.async_tools.close()
            print("✅ Async resources cleaned up")

    def demonstrate_cache_behavior(self):
        """Demonstrate how caching works with repeated queries - DEMO METHOD"""
        print("\n🔬 Cache Behavior Demonstration")
        print("=" * 50)

        # Test query
        test_order_id = 1001

        print(f"Testing cache behavior with order {test_order_id}...")
        print(f"This will call get_order_details multiple times to show cache effects.\n")

        for i in range(3):
            print(f"Call #{i+1}:")
            start_time = time.time()

            result = self.sync_tools.get_order_details(test_order_id)

            execution_time = time.time() - start_time
            print(f"   Execution time: {execution_time:.4f}s")

            if isinstance(result, dict) and 'error' not in result:
                print(f"   Result: Order for {result.get('customer_name', 'Unknown')}")
            else:
                print(f"   Result: {result}")

            # Check if this was served from cache
            cache_stats = cache_manager.get_detailed_stats()
            method_stats = cache_stats['by_method'].get('get_order_details', {})
            hit_rate = method_stats.get('hit_rate', 0)

            print(f"   Cache hit rate for this method: {hit_rate:.1f}%")
            print()

            if i < 2:
                time.sleep(1)  # Small delay between calls

        print("💡 Observations:")
        print("   • First call: Database query (slower)")
        print("   • Subsequent calls: Served from cache (faster)")
        print("   • Cache TTL: 5 minutes for this method")
        print("   • Cache keys are based on function name and parameters")

    @performance_monitor
    def run_query_with_enhanced_tracking(self, query: str):
        """Enhanced query execution with detailed tracking"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        query_start_time = time.time()

        # Record the query attempt
        print(f"🤖 Processing query with enhanced tracking: '{query}'")
        print("-" * 70)

        # Clear recent tracking for this query context
        initial_tool_calls = len(tool_tracker.tool_calls)

        try:
            # Execute the query
            response = self.agent.query(query)

            query_execution_time = time.time() - query_start_time
            tools_used_in_query = len(tool_tracker.tool_calls) - initial_tool_calls

            print(f"\n✅ Query completed successfully!")
            print(f"⏱️  Total execution time: {query_execution_time:.2f}s")
            print(f"🔧 Tools called during this query: {tools_used_in_query}")

            # Show tools used for this specific query
            if tools_used_in_query > 0:
                print(f"\n🔧 Tools used in this query:")
                recent_calls = tool_tracker.tool_calls[-tools_used_in_query:]
                for i, call in enumerate(recent_calls, 1):
                    status = "✅" if call['success'] else "❌"
                    source = "💾 Cache" if 'cache' in call.get('reasoning', '').lower() else "🗄️  Database"
                    print(f"   {i}. {status} {call['tool_name']} ({call['execution_time']:.3f}s) {source}")
                    print(f"      └─ {call['reasoning']}")

            print(f"\n📋 Agent Response:")
            print("=" * 50)
            print(response)

            # Check for cache benefits
            cache_stats = cache_manager.get_detailed_stats()
            recent_cache_hits = sum(1 for call in tool_tracker.tool_calls[-tools_used_in_query:]
                                    if 'cache' in call.get('reasoning', '').lower())

            if recent_cache_hits > 0:
                print(f"\n💾 Cache Performance for this query:")
                print(f"   Cache hits: {recent_cache_hits}/{tools_used_in_query}")
                print(f"   Time saved: ~{recent_cache_hits * 0.1:.1f}s (estimated)")

        except Exception as e:
            query_execution_time = time.time() - query_start_time
            print(f"\n❌ Query failed after {query_execution_time:.2f}s")
            print(f"Error: {e}")

            # Still show tools that were attempted
            tools_attempted = len(tool_tracker.tool_calls) - initial_tool_calls
            if tools_attempted > 0:
                print(f"\n🔧 Tools attempted before failure:")
                recent_calls = tool_tracker.tool_calls[-tools_attempted:]
                for i, call in enumerate(recent_calls, 1):
                    status = "✅" if call['success'] else "❌"
                    print(f"   {i}. {status} {call['tool_name']} ({call['execution_time']:.3f}s)")

    def setup_query_decomposer(self):
        """Setup intelligent query decomposition"""
        if self.agent:
            self.query_decomposer = QueryDecomposer(self.agent)
            return True
        return False

    def run_intelligent_query(self, query: str):
        """Enhanced intelligent query execution with decomposition tracking"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            print("⚠️  Query decomposer not initialized, using standard execution...")
            return self.run_query(query)

        # Start call graph tracking
        query_id = self.execution_tracker.track_agent_query_start(query, "IntelligentAgent")

        try:
            print(f"🤖 Processing query with intelligent decomposition and tracking: '{query}'")
            print(f"📊 Tracking ID: {query_id}")
            print("=" * 70)

            # Track complexity assessment
            assessment = self.query_decomposer.assess_query_complexity(query)
            self.execution_tracker.track_complexity_assessment(
                query, assessment['complexity_score'], assessment['detected_patterns']
            )

            # Track decomposition process
            self.execution_tracker.add_thinking_process(
                "Starting intelligent query decomposition",
                f"Complexity: {assessment['complexity_score']}, indicators: {assessment['indicators']}"
            )

            # Get subgoals and track them
            subgoals = self.query_decomposer.decompose_query(query)
            decomp_node = self.execution_tracker.add_query_decomposition(
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
                print("-" * 40)

                # Track subgoal execution start
                subgoal_thinking = self.execution_tracker.add_thinking_process(
                    f"Processing subgoal {i}/{len(subgoals)}",
                    f"Type: {subgoal['type']}, Priority: {subgoal['priority']}"
                )

                # Add context from previous subgoals
                contextual_query = subgoal['subgoal']
                if context:
                    contextual_query = f"Based on previous analysis: {' '.join(context[-2:])}. Now: {subgoal['subgoal']}"
                    self.execution_tracker.add_logical_inference(
                        f"Incorporating context from {len(context)} previous subgoals",
                        context[-2:] if len(context) >= 2 else context,
                        f"Enhanced query: {contextual_query[:100]}..."
                    )

                try:
                    # Track the individual subgoal execution
                    self.execution_tracker.add_thinking_process(
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
                    self.execution_tracker.add_logical_inference(
                        f"Subgoal {i} completed successfully",
                        [f"Query: {subgoal['subgoal'][:50]}..."],
                        f"Result obtained: {str(result)[:100]}..."
                    )

                except Exception as e:
                    error_msg = f"Subgoal {i} failed: {e}"
                    print(f"⚠️  {error_msg}")
                    self.execution_tracker.add_error(error_msg, f"subgoal_{i}_error")

                    results.append({
                        'subgoal': subgoal['subgoal'],
                        'result': f"Failed to execute: {e}",
                        'type': subgoal['type']
                    })

            # Track result synthesis
            self.execution_tracker.add_thinking_process(
                f"Synthesizing results from {len(results)} subgoals",
                "Combining partial results into comprehensive response"
            )

            synthesis_node = self.execution_tracker.track_response_synthesis(
                results, "subgoal_based_synthesis"
            )

            # Generate final response
            final_response = self._synthesize_results(query, results)

            self.execution_tracker.complete_result_synthesis(synthesis_node, final_response)
            self.execution_tracker.finalize_query(final_response, success=True)

            print(f"\n✅ Complete Response:")
            print("=" * 50)
            print(final_response)

            # Show execution summary
            summary = self.execution_tracker.export_execution_summary()
            print(f"\n📊 Intelligent Execution Summary:")
            print(f"   Subgoals processed: {len(subgoals)}")
            print(f"   Total time: {summary.get('total_execution_time', 0):.2f}s")
            print(f"   Reasoning steps: {summary.get('reasoning_steps', 0)}")
            print(f"   Thinking processes: {summary.get('thinking_steps', 0)}")
            print(f"   📈 Call graph saved to: ./generated_callgraphs/")

            return final_response

        except Exception as e:
            self.execution_tracker.add_error(str(e), "intelligent_query_error")
            self.execution_tracker.finalize_query(f"Intelligent query failed: {str(e)}", success=False)
            print(f"❌ Intelligent query execution failed: {e}")
            print("🔄 Falling back to standard execution...")
            self.run_query(query)

    def run_query_with_options(self, query: str):
        """Give user choice between standard and intelligent execution"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            return self.run_query(query)

        # Quick complexity check
        assessment = self.query_decomposer.assess_query_complexity(query)

        if assessment['is_complex']:
            print(f"🧩 Complex query detected (score: {assessment['complexity_score']})")
            print(f"📋 Complexity indicators: {', '.join(assessment['indicators'])}")
            print("\n📋 Execution options:")
            print("1. 🚀 Standard execution (single attempt)")
            print("2. 🧩 Intelligent decomposition (break into subgoals)")
            print("3. 🤖 Auto-decide based on complexity")

            choice = input("Select execution method (1-3, default=3): ").strip() or "3"

            if choice == "1":
                self.run_query(query)
            elif choice == "2":
                self.run_intelligent_query(query)
            else:  # Auto-decide
                if assessment['requires_decomposition']:
                    print("🧩 Auto-selected: Intelligent decomposition")
                    self.run_intelligent_query(query)
                else:
                    print("🚀 Auto-selected: Standard execution")
                    self.run_query(query)
        else:
            print("✅ Simple query detected, using standard execution...")
            self.run_query(query)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Enhanced performance metrics"""
        metrics = {
            'database_type': 'Unknown',
            'embedding_device': 'Unknown',
            'tools_count': len(self.tools) if self.tools else 0,
            'parallel_processing': hasattr(self, 'async_tools') and self.async_tools is not None,
            'config': {
                'cache_ttl': config.cache_ttl,
                'max_workers': config.max_workers,
                'query_timeout': config.query_timeout
            }
        }

        # Database metrics
        if hasattr(self, 'db_manager') and hasattr(self.db_manager, 'query_stats'):
            metrics['database_stats'] = self.db_manager.query_stats.copy()
            conn_info = self.db_manager.get_connection_info()
            metrics['database_type'] = conn_info.get('type', 'Unknown')

        # Cache metrics
        if hasattr(self, 'sync_tools'):
            cache_stats = {}
            for method_name in ['get_order_items', 'get_order_details']:
                method = getattr(self.sync_tools, method_name, None)
                if method and hasattr(method, 'cache_stats'):
                    cache_stats[method_name] = method.cache_stats()
            metrics['cache_stats'] = cache_stats

        # Memory metrics
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            metrics['memory'] = {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }

        return metrics

    def view_call_graphs(self):
        """View generated call graphs"""
        import os
        import webbrowser
        from pathlib import Path

        graph_dir = "../src/generated_callgraphs"
        if not os.path.exists(graph_dir):
            print("📊 No call graphs directory found.")
            print("   Call graphs will be created after running queries.")
            return

        html_files = [f for f in os.listdir(graph_dir) if f.endswith('.html')]

        if not html_files:
            print("📊 No HTML call graphs found.")
            print("   Execute some queries first to generate call graphs.")
            return

        # Sort by modification time (newest first)
        html_files.sort(key=lambda f: os.path.getmtime(os.path.join(graph_dir, f)), reverse=True)

        print(f"\n📊 Generated Call Graphs ({len(html_files)} files):")
        print("=" * 50)

        for i, file in enumerate(html_files[:10], 1):  # Show only last 10
            file_path = os.path.join(graph_dir, file)
            file_size = os.path.getsize(file_path)
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

            print(f"   {i:2d}. {file}")
            print(f"       Size: {file_size:,} bytes | Modified: {mod_date}")

        if len(html_files) > 10:
            print(f"   ... and {len(html_files) - 10} more files")

        print(f"\n📋 Options:")
        print(f"   • Enter file number (1-{min(10, len(html_files))})")
        print(f"   • 'latest' for most recent graph")
        print(f"   • 'all' to see all files")
        print(f"   • 'clean' to delete old graphs")
        print(f"   • 'back' to return to main menu")

        choice = input("\nSelect option: ").strip().lower()

        if choice == 'back':
            return
        elif choice == 'latest':
            latest_file = html_files[0]  # Already sorted by newest first
            filepath = os.path.join(graph_dir, latest_file)
            self._open_graph_file(filepath)
        elif choice == 'all':
            self._show_all_graphs(html_files, graph_dir)
        elif choice == 'clean':
            self._clean_old_graphs(html_files, graph_dir)
        elif choice.isdigit() and 1 <= int(choice) <= min(10, len(html_files)):
            selected_file = html_files[int(choice)-1]
            filepath = os.path.join(graph_dir, selected_file)
            self._open_graph_file(filepath)
        else:
            print("❌ Invalid choice.")

    def _open_graph_file(self, filepath):
        """Open a call graph file in the browser"""
        import webbrowser
        import os

        try:
            # Convert to absolute path for browser
            abs_path = os.path.abspath(filepath)
            file_url = f'file://{abs_path}'

            print(f"🌐 Opening call graph in browser...")
            print(f"   File: {os.path.basename(filepath)}")

            webbrowser.open(file_url)
            print(f"✅ Call graph opened successfully!")
            print(f"💡 If browser didn't open, manually open: {abs_path}")

        except Exception as e:
            print(f"❌ Failed to open graph: {e}")
            print(f"💡 Manually open this file: {filepath}")

    def _show_all_graphs(self, html_files, graph_dir):
        """Show all available graphs with detailed info"""
        print(f"\n📊 All Generated Call Graphs ({len(html_files)} files):")
        print("=" * 70)

        total_size = 0
        for i, file in enumerate(html_files, 1):
            file_path = os.path.join(graph_dir, file)
            file_size = os.path.getsize(file_path)
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

            # Extract query info from filename if possible
            query_info = "Unknown query"
            if "_" in file:
                parts = file.replace('.html', '').split('_')
                if len(parts) >= 3:
                    query_info = f"Query {parts[1]}"

            total_size += file_size

            print(f"{i:3d}. {file}")
            print(f"     Query: {query_info}")
            print(f"     Size: {file_size:,} bytes | Modified: {mod_date}")
            print()

        print(f"📈 Summary: {len(html_files)} graphs, {total_size:,} bytes total")

        choice = input(f"\nEnter file number to open (1-{len(html_files)}) or 'back': ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(html_files):
            selected_file = html_files[int(choice)-1]
            filepath = os.path.join(graph_dir, selected_file)
            self._open_graph_file(filepath)

    def _clean_old_graphs(self, html_files, graph_dir):
        """Clean up old call graph files"""
        import os

        if len(html_files) <= 5:
            print("📊 Only 5 or fewer graphs exist. No cleanup needed.")
            return

        print(f"\n🧹 Call Graph Cleanup")
        print(f"Current graphs: {len(html_files)}")
        print(f"Options:")
        print(f"  1. Keep latest 10, delete {len(html_files) - 10} older files")
        print(f"  2. Keep latest 5, delete {len(html_files) - 5} older files")
        print(f"  3. Delete all except latest 3")
        print(f"  4. Delete ALL graphs (careful!)")
        print(f"  5. Cancel cleanup")

        choice = input("Select cleanup option (1-5): ").strip()

        files_to_delete = []

        if choice == '1' and len(html_files) > 10:
            files_to_delete = html_files[10:]
        elif choice == '2' and len(html_files) > 5:
            files_to_delete = html_files[5:]
        elif choice == '3' and len(html_files) > 3:
            files_to_delete = html_files[3:]
        elif choice == '4':
            files_to_delete = html_files
        elif choice == '5':
            print("❌ Cleanup cancelled.")
            return
        else:
            print("❌ Invalid choice or no files to delete.")
            return

        if files_to_delete:
            print(f"\n⚠️  About to delete {len(files_to_delete)} graph files:")
            for file in files_to_delete[:5]:  # Show first 5
                print(f"   - {file}")
            if len(files_to_delete) > 5:
                print(f"   ... and {len(files_to_delete) - 5} more")

            confirm = input(f"\nConfirm deletion of {len(files_to_delete)} files? (yes/no): ").strip().lower()

            if confirm == 'yes':
                deleted_count = 0
                for file in files_to_delete:
                    try:
                        os.remove(os.path.join(graph_dir, file))
                        deleted_count += 1
                    except Exception as e:
                        print(f"❌ Failed to delete {file}: {e}")

                print(f"✅ Successfully deleted {deleted_count} graph files")
            else:
                print("❌ Deletion cancelled.")

    def show_call_graph_stats(self):
        """Show statistics about call graph generation"""
        import os

        graph_dir = "../src/generated_callgraphs"

        if not os.path.exists(graph_dir):
            print("📊 No call graphs directory found.")
            return

        html_files = [f for f in os.listdir(graph_dir) if f.endswith('.html')]

        if not html_files:
            print("📊 No call graphs generated yet.")
            return

        print(f"\n📊 Call Graph Statistics")
        print("=" * 40)

        total_size = sum(os.path.getsize(os.path.join(graph_dir, f)) for f in html_files)

        # Get file ages
        now = datetime.now()
        file_ages = []
        for file in html_files:
            file_path = os.path.join(graph_dir, file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            age_hours = (now - mod_time).total_seconds() / 3600
            file_ages.append(age_hours)

        print(f"📈 Overview:")
        print(f"   Total graphs: {len(html_files)}")
        print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        print(f"   Average size: {total_size/len(html_files):,.0f} bytes")
        print(f"   Newest: {min(file_ages):.1f} hours ago")
        print(f"   Oldest: {max(file_ages):.1f} hours ago")

        # Size distribution
        file_sizes = [os.path.getsize(os.path.join(graph_dir, f)) for f in html_files]
        print(f"\n📏 Size Distribution:")
        print(f"   Smallest: {min(file_sizes):,} bytes")
        print(f"   Largest: {max(file_sizes):,} bytes")
        print(f"   Median: {sorted(file_sizes)[len(file_sizes)//2]:,} bytes")

        # Recent activity
        recent_files = [f for f, age in zip(html_files, file_ages) if age < 24]
        print(f"\n⏰ Recent Activity (last 24 hours):")
        print(f"   Graphs generated: {len(recent_files)}")

        if recent_files:
            print(f"   Most recent files:")
            sorted_recent = sorted(recent_files, key=lambda f: os.path.getmtime(os.path.join(graph_dir, f)), reverse=True)
            for file in sorted_recent[:3]:
                mod_time = os.path.getmtime(os.path.join(graph_dir, file))
                mod_date = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
                print(f"     - {file} (generated at {mod_date})")

    def __del__(self):
        """Cleanup async resources and database connections"""
        try:
            if hasattr(self, 'async_tools') and self.async_tools:
                self.async_tools.close()
        except:
            pass

        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                self.db_manager.disconnect()
        except:
            pass


def main():
    """Main function with enhanced tracking"""
    print("🚀 Customer Service AI Agent with Enhanced Tool Tracking")
    print("Connecting to MySQL database and setting up AI tools...")

    try:
        agent = CustomerServiceAgent()
        agent.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")

        # Show final statistics
        if len(tool_tracker.tool_calls) > 0:
            print("\n📊 Final Session Statistics:")
            report = tool_tracker.get_comprehensive_report()
            session = report['session_summary']
            print(f"   Total Tool Calls: {session['total_tool_calls']}")
            print(f"   Session Duration: {session['session_duration_minutes']:.1f} minutes")
            print(f"   Success Rate: {session['success_rate']:.1f}%")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()