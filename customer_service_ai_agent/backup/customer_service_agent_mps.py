#!/usr/bin/env python3
"""
Customer Service AI Agent with Database Integration
Equivalent to the Jupyter notebook but with MySQL database connectivity
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps

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
        'nest-asyncio': 'nest_asyncio'
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
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

class DatabaseManager:
    """Handles all database operations with connection pooling"""

    def __init__(self, host='localhost', user='root', password='auburn', database='customer_service_db'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connection_pool = None

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

    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query with connection pooling"""
        try:
            if self.connection_pool:
                # Use connection from pool
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor(buffered=True)
                cursor.execute(query, params or ())
                results = cursor.fetchall()
                cursor.close()
                conn.close()  # Return connection to pool
                return results
            else:
                # Fallback to single connection
                cursor = self.connection.cursor(buffered=True)
                cursor.execute(query, params or ())
                results = cursor.fetchall()
                cursor.close()
                return results

        except Error as e:
            print(f"❌ Query execution failed: {e}")
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
        self.sync_tools = sync_tools  # Reference to existing tools
        self.executor = ThreadPoolExecutor(max_workers=4)

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

class CustomerServiceTools:
    """Enhanced customer service tool functions with sophisticated analytics"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._cache = {}  # Simple caching for performance

    # ============= BASIC TOOLS (Keep existing) =============
    def get_order_items(self, order_id: int) -> List[str]:
        """Given an order ID, returns the list of items purchased for that order"""
        cache_key = f"order_items_{order_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        query = """
                SELECT p.product_name
                FROM order_items oi
                         JOIN products p ON oi.product_id = p.product_id
                WHERE oi.order_id = %s \
                """
        results = self.db.execute_query(query, (order_id,))
        items = [item[0] for item in results] if results else []
        self._cache[cache_key] = items
        return items

    def get_delivery_date(self, order_id: int) -> str:
        """Given an order ID, returns the delivery date for that order"""
        query = """
                SELECT DATE_FORMAT(delivery_date, '%d-%b')
                FROM orders
                WHERE order_id = %s \
                """
        results = self.db.execute_query(query, (order_id,))
        return results[0][0] if results else ""

    def get_item_return_days(self, item: str) -> int:
        """Given an item name, returns the return policy in days"""
        query = """
                SELECT return_days
                FROM products
                WHERE product_name LIKE %s
                    LIMIT 1 \
                """
        results = self.db.execute_query(query, (f"%{item}%",))
        return results[0][0] if results else 45  # Default 45 days

    def get_order_details(self, order_id: int) -> Dict[str, Any]:
        """Get comprehensive order details"""
        # Handle different input formats
        if isinstance(order_id, dict):
            if 'order_id' in order_id:
                order_id = order_id['order_id']
            else:
                return {"error": "Invalid input format. Expected order_id."}

        try:
            order_id = int(order_id)  # Ensure it's an integer
        except (ValueError, TypeError):
            return {"error": f"Invalid order_id format: {order_id}. Must be a number."}

        query = """
                SELECT o.order_id, o.order_date, o.delivery_date, o.status,
                       o.total_amount, c.first_name, c.last_name, c.email
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = %s \
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

    def search_orders_by_customer(self, email: str) -> List[Dict]:
        """Search orders by customer email"""
        query = """
                SELECT o.order_id, o.order_date, o.status, o.total_amount
                FROM orders o
                         JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.email LIKE %s \
                """
        results = self.db.execute_query(query, (f"%{email}%",))
        return [{'order_id': row[0], 'order_date': row[1], 'status': row[2], 'total_amount': row[3]}
                for row in results]

    # ============= ADD THESE NEW ADVANCED TOOLS =============
    def analyze_customer_orders_comprehensive(self, customer_email: str) -> Dict[str, Any]:
        """Comprehensive analysis of all customer orders with return eligibility"""
        query = """
                SELECT
                    o.order_id, o.order_date, o.delivery_date, o.status, o.total_amount,
                    p.product_name, p.return_days, oi.quantity, oi.unit_price,
                    DATEDIFF(CURDATE(), o.delivery_date) as days_since_delivery,
                    CASE
                        WHEN o.delivery_date IS NULL THEN 'Not delivered'
                        WHEN DATEDIFF(CURDATE(), o.delivery_date) <= p.return_days THEN 'Returnable'
                        ELSE 'Return expired'
                        END as return_status
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
                SELECT
                    o.order_id, o.delivery_date, o.total_amount,
                    p.product_name, p.return_days, p.price,
                    oi.quantity, oi.unit_price,
                    DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY) as return_deadline,
                    DATEDIFF(DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY), CURDATE()) as days_remaining,
                    CASE
                        WHEN DATEDIFF(DATE_ADD(o.delivery_date, INTERVAL p.return_days DAY), CURDATE()) > 0 THEN 'Active'
                        ELSE 'Expired'
                        END as return_window_status
                FROM orders o
                         JOIN order_items oi ON o.order_id = oi.order_id
                         JOIN products p ON oi.product_id = p.product_id
                WHERE o.order_id = %s AND o.delivery_date IS NOT NULL \
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
                SELECT
                    o.order_id, o.customer_id, o.order_date, o.delivery_date, o.status, o.total_amount,
                    c.email, c.city, c.state,
                    DATEDIFF(CURDATE(), o.order_date) as days_since_order,
                    DATEDIFF(CURDATE(), o.delivery_date) as days_since_delivery,
                    COUNT(o2.order_id) as customer_order_history
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
                SELECT
                    o.status,
                    COUNT(*) as order_count,
                    AVG(o.total_amount) as avg_order_value,
                    SUM(o.total_amount) as total_value,
                    AVG(DATEDIFF(o.delivery_date, o.order_date)) as avg_delivery_days,
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
                analysis['performance_issues'].append(f"Status '{row[0]}': Average delivery time {row[4]:.1f} days (target: 7 days)")
            if row[5] > 0:
                analysis['performance_issues'].append(f"Status '{row[0]}': {row[5]} delayed deliveries out of {row[1]} orders")

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
    """Intelligent query decomposition for complex customer service queries"""

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
        """Break down complex query into manageable subgoals"""
        assessment = self.assess_query_complexity(query)

        if not assessment['requires_decomposition']:
            return [{'subgoal': query, 'type': 'simple', 'priority': 1}]

        # Use LLM to intelligently decompose the query
        decomposition_prompt = f"""
        Break down this complex customer service query into 3-5 simple, specific subgoals that can be executed sequentially.
        Each subgoal should be actionable and focused on a single task.
        
        Original Query: "{query}"
        
        Format your response as a numbered list of subgoals:
        1. [Specific action with clear parameters]
        2. [Next logical step]
        3. [Analysis or calculation step]
        4. [Final synthesis or recommendation]
        
        Make each subgoal specific enough to be executed by a single tool call.
        """

        try:
            # Use the agent's LLM for decomposition
            from llama_index.core import Settings
            decomposition_response = Settings.llm.complete(decomposition_prompt)

            # Parse the response into subgoals
            subgoals = self._parse_decomposition_response(str(decomposition_response))

            return subgoals

        except Exception as e:
            print(f"⚠️  Decomposition failed, using pattern-based fallback: {e}")
            return self._pattern_based_decomposition(query, assessment)

    def _parse_decomposition_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response into structured subgoals"""
        lines = response.strip().split('\n')
        subgoals = []

        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Clean up the line
                clean_line = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•']:
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

    def _pattern_based_decomposition(self, query: str, assessment: Dict) -> List[Dict[str, str]]:
        """Fallback pattern-based decomposition when LLM fails"""
        subgoals = []

        # Multi-customer pattern
        if 'multi_customer' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Identify and list all customers mentioned in the query', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Get detailed order information for each customer', 'type': 'data_collection', 'priority': 2},
                {'subgoal': 'Analyze patterns and issues across the customer group', 'type': 'analysis', 'priority': 3}
            ])

        # Predictive analysis pattern
        if 'predictive_analysis' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Analyze recent order and customer behavior patterns', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Identify risk factors and potential issues', 'type': 'analysis', 'priority': 2},
                {'subgoal': 'Generate proactive recommendations and action plans', 'type': 'synthesis', 'priority': 3}
            ])

        # Business impact pattern
        if 'business_impact' in assessment['detected_patterns']:
            subgoals.extend([
                {'subgoal': 'Calculate financial metrics and business impact', 'type': 'analysis', 'priority': 1},
                {'subgoal': 'Assess risk levels and operational implications', 'type': 'analysis', 'priority': 2},
                {'subgoal': 'Recommend strategies to minimize impact', 'type': 'synthesis', 'priority': 3}
            ])

        # Default decomposition if no specific patterns
        if not subgoals:
            subgoals = [
                {'subgoal': 'Collect relevant data based on the query requirements', 'type': 'data_collection', 'priority': 1},
                {'subgoal': 'Analyze the collected data and identify key insights', 'type': 'analysis', 'priority': 2},
                {'subgoal': 'Provide comprehensive response with recommendations', 'type': 'synthesis', 'priority': 3}
            ]

        return subgoals

    def execute_decomposed_query(self, query: str) -> str:
        """Execute query with automatic decomposition if needed"""
        print(f"🔍 Analyzing query complexity...")

        assessment = self.assess_query_complexity(query)

        if not assessment['requires_decomposition']:
            print("✅ Simple query detected, executing directly...")
            return self._execute_simple_query(query)

        print(f"🧩 Complex query detected (score: {assessment['complexity_score']})")
        print(f"📋 Patterns found: {', '.join(assessment['detected_patterns'])}")
        print("🔄 Breaking down into subgoals...")

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

        # Synthesize final response
        return self._synthesize_results(query, results)

    def _execute_simple_query(self, query: str) -> str:
        """Execute a simple query using the agent"""
        try:
            response = self.agent.query(query)
            return str(response)
        except Exception as e:
            return f"Query execution failed: {e}"

    def _synthesize_results(self, original_query: str, results: List[Dict]) -> str:
        """Combine results from subgoals into comprehensive response"""
        print(f"\n🔄 Synthesizing results from {len(results)} subgoals...")

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

        return final_response


class CustomerServiceAgent:
    """Main AI Agent class with hybrid sync/async capabilities"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.tools = None
        self.agent = None
        self.support_index = None
        self.query_decomposer = None
        self.sync_tools = None      # Regular tools
        self.async_tools = None     # Async tools for parallel operations

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
        """Setup Local LLM connection with enhanced configuration"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("🤖 Setting up Local LLM...")
        try:
            # Check for Mac GPU availability
            try:
                import torch
                if torch.backends.mps.is_available():
                    device_str = "mps"
                    print("🚀 Mac GPU (Metal) detected and will be used for embeddings")
                else:
                    device_str = "cpu"
                    print("💻 Using CPU for embeddings")
            except ImportError:
                device_str = "cpu"
                print("💻 PyTorch not available, using CPU for embeddings")

            # Local server configuration
            local_llm_url = "http://127.0.0.1:1234/v1"

            # Setup the LLM to use local server with optimized settings
            Settings.llm = OpenAILike(
                model="open_gpt4_8x7b_v0.2",
                api_base=local_llm_url,
                api_key="lm-studio",
                is_local=True,
                temperature=0.1,
                max_tokens=3000,
                timeout=45,
                max_retries=2
            )

            # Setup local embedding model for RAG with GPU acceleration
            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    max_length=512,
                    device=device_str,
                    trust_remote_code=True
                )
            except Exception as embed_error:
                print(f"⚠️  GPU embedding failed, falling back to CPU: {embed_error}")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    max_length=512
                )
                device_str = "cpu"

            print("✅ LLM setup completed with optimizations!")
            print(f"   🎯 Temperature: 0.1 (focused responses)")
            print(f"   📝 Max tokens: 3000 (detailed answers)")
            print(f"   ⏱️  Timeout: 45s per call")
            print(f"   🔄 Retries: 2 attempts")
            print(f"   🚀 Embedding device: {device_str}")
            return True

        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            print("💡 Make sure LM Studio is running on http://127.0.0.1:1234/v1")
            return False

    def get_support_files_from_user(self):
        """Get support files from user input with policy_files directory support"""
        print("\n📚 Support Document Configuration")
        print("=" * 50)
        print("Complete support document suite includes:")
        print("1. Customer Service.txt - Contact information and response times")
        print("2. FAQ.txt - Frequently asked questions")
        print("3. Return Policy.txt - Return and refund policies")
        print("4. Warranty Policy.txt - Comprehensive warranty coverage")
        print("5. Escalation Procedures.txt - Multi-level escalation framework")
        print("6. Technical Troubleshooting Guide.txt - Hardware/software support")
        print("7. Business Policies and Procedures.txt - Customer tiers and policies")
        print("8. Product Knowledge Database.txt - Product specifications")
        print("9. Order Management and Fulfillment.txt - Order lifecycle procedures")
        print()
        print("📁 Documents location: ./policy_files/")
        print()

        # Define complete document suite with policy_files path
        complete_suite = [
            'policy_files/Customer Service.txt',
            'policy_files/FAQ.txt',
            'policy_files/Return Policy.txt',
            'policy_files/Warranty Policy.txt',
            'policy_files/Escalation Procedures.txt',
            'policy_files/Technical Troubleshooting Guide.txt',
            'policy_files/Business Policies and Procedures.txt',
            'policy_files/Product Knowledge Database.txt',
            'policy_files/Order Management and Fulfillment.txt'
        ]

        # Basic document suite (original 3)
        basic_suite = [
            'policy_files/Customer Service.txt',
            'policy_files/FAQ.txt',
            'policy_files/Return Policy.txt'
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
        policy_dir = "policy_files"

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
        additional_files = [f for f in all_files if f not in expected_files and f.endswith(('.txt', '.pdf', '.doc', '.docx'))]

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

        policy_dir = "policy_files"

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
                            file_name = available_files[idx-1][0]
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

    def setup_support_documents(self):
        """Setup vector index for customer support documents"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("📚 Setting up support documents...")

        try:
            # Get support files from user
            requested_files = self.get_support_files_from_user()

            # Validate files exist
            valid_files = self.validate_support_files(requested_files)
            if not valid_files:
                print("❌ Required support files not found. Exiting program.")
                print("Please ensure all support documents are available and try again.")
                return False

            print(f"\n📁 Loading {len(valid_files)} support document(s):")
            for file in valid_files:
                file_size = os.path.getsize(file)
                print(f"   - {file} ({file_size:,} bytes)")

            # Setup vector index for support documents
            try:
                support_docs = SimpleDirectoryReader(input_files=valid_files).load_data()
                print(f"📄 Loaded {len(support_docs)} document(s)")

                splitter = SentenceSplitter(
                    chunk_size=1024,
                    chunk_overlap=50  # Add overlap for better context
                )
                support_nodes = splitter.get_nodes_from_documents(support_docs)
                print(f"🔧 Created {len(support_nodes)} text chunks")

                self.support_index = VectorStoreIndex(support_nodes)

                print(f"✅ Support documents indexed successfully!")
                print(f"   Documents: {len(valid_files)}")
                print(f"   Chunks: {len(support_nodes)}")
                print(f"   Ready for vector search!")

                return True

            except Exception as e:
                print(f"❌ Error processing support documents: {e}")
                print("Please check document formats and content.")
                return False

        except Exception as e:
            print(f"❌ Support documents setup failed: {e}")
            return False

    def create_tools(self):
        """Create enhanced tools for the agent with hybrid sync/async capabilities"""
        if not LLAMAINDEX_AVAILABLE:
            return False

        print("🛠️  Creating enhanced agent tools...")
        try:
            # Create regular database-connected tools
            self.sync_tools = CustomerServiceTools(self.db_manager)

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
            return True

        except Exception as e:
            print(f"❌ Tool creation failed: {e}")
            return False

    def create_agent(self):
        """Create the enhanced AI agent with better configuration"""
        if not LLAMAINDEX_AVAILABLE or not self.tools:
            return False

        print("🤖 Creating enhanced AI agent...")
        try:
            # Setup the Agent worker with optimized settings
            agent_worker = ReActAgentWorker.from_tools(
                self.tools,
                llm=Settings.llm,
                verbose=True,
                max_iterations=15,  # Increased from default 10
                allow_parallel_tool_calls=False  # Sequential for better reasoning
            )

            # Create agent runner with enhanced configuration
            self.agent = AgentRunner(
                agent_worker,
                memory=None,  # Fresh context for each query
                verbose=True
            )

            print("✅ Enhanced AI agent created successfully!")
            print("   🔧 Max iterations: 15 (increased for complex scenarios)")
            print("   🧠 Sequential reasoning: Enabled for better tool selection")
            print("   📝 Fresh context: Each query starts clean")
            print("   ⚡ Parallel processing: Available for expert scenarios")
            return True

        except Exception as e:
            print(f"❌ Agent creation failed: {e}")
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
                        cont = input(f"\nPress Enter to continue to next scenario (or 'stop' to finish): ").strip().lower()
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
                cont = input(f"\nPress Enter to continue to next {difficulty.lower()} scenario (or 'stop' to finish): ").strip().lower()
                if cont == 'stop':
                    break

    def run_custom_query(self):
        """Run a custom user query"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        print("\n💬 Custom Query Mode")
        print("=" * 30)
        print("You can ask questions about:")
        print("- Order details (e.g., 'What items are in order 1001?')")
        print("- Delivery dates (e.g., 'When will order 1002 be delivered?')")
        print("- Return policies (e.g., 'What's the return policy for laptops?')")
        print("- Customer support information")
        print("- Multi-order analysis (e.g., 'Compare orders 1007, 1017, 1023')")
        print("- Geographic analysis (e.g., 'Orders in Auburn, AL')")
        print("- Predictive analysis (e.g., 'Which customers might have issues?')")
        print("- Any combination of the above")
        print("\nType 'back' to return to main menu.")
        print()

        while True:
            query = input("🤔 Your question: ").strip()

            if query.lower() == 'back':
                break
            elif query:
                self.run_query_with_options(query)
                print("\n" + "-" * 50)
            else:
                print("❌ Please enter a valid question.")

    @performance_monitor
    def run_query(self, query: str):
        """Execute a query using the agent with enhanced error handling"""
        if not self.agent:
            print("❌ Agent not initialized!")
            return

        try:
            print(f"🤖 Processing query: '{query}'")
            print("-" * 50)

            # Add timeout and retry logic
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Query execution timeout")

            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                response = self.agent.query(query)
                signal.alarm(0)  # Cancel timeout

                print("\n✅ Agent Response:")
                print("=" * 30)
                print(response)
                print()

            except TimeoutError:
                signal.alarm(0)
                print("\n⏰ Query timeout! The query took too long to process.")
                print("💡 Try simplifying your question or breaking it into parts.")

            except Exception as query_error:
                signal.alarm(0)
                print(f"\n❌ Query execution error: {query_error}")

                # Try to provide helpful fallback
                if "max iterations" in str(query_error).lower():
                    print("\n🔄 The query was too complex and reached iteration limit.")
                    print("💡 Suggestions:")
                    print("   • Break down your question into simpler parts")
                    print("   • Ask about specific order IDs or customers")
                    print("   • Try using more specific keywords")
                elif "tool" in str(query_error).lower():
                    print("\n🛠️  Tool execution issue detected.")
                    print("💡 The system might need specific data to answer your question.")
                    print("   • Ensure order IDs, emails, or product names are correct")
                    print("   • Try asking about existing orders (1001-1030)")

        except Exception as e:
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
                                        ORDER BY order_date DESC
                                            LIMIT 10
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

    def main_menu(self):
        """Main interactive menu"""
        if not self.initialize():
            print("❌ Initialization failed. Exiting.")
            return

        while True:
            print("\n" + "=" * 60)
            print("🤖 CUSTOMER SERVICE AI AGENT")
            print("=" * 60)
            print("1. 🎯 Run Predefined Scenarios")
            print("2. 💬 Custom Query")
            print("3. 📊 Database Statistics")
            print("4. 🚪 Exit")
            print("=" * 60)

            choice = input("Select an option (1-4): ").strip()

            if choice == '1':
                self.run_predefined_scenarios()
            elif choice == '2':
                self.run_custom_query()
            elif choice == '3':
                self.show_database_stats()
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")

        # Cleanup
        print("🧹 Cleaning up resources...")
        self.db_manager.disconnect()
        if hasattr(self, 'async_tools') and self.async_tools:
            self.async_tools.close()
            print("✅ Async resources cleaned up")

    def setup_query_decomposer(self):
        """Setup intelligent query decomposition"""
        if self.agent:
            self.query_decomposer = QueryDecomposer(self.agent)
            return True
        return False

    def run_intelligent_query(self, query: str):
        """Enhanced query execution with automatic decomposition"""
        if not hasattr(self, 'query_decomposer') or self.query_decomposer is None:
            print("⚠️  Query decomposer not initialized, using standard execution...")
            return self.run_query(query)

        try:
            print(f"🤖 Processing query with intelligent decomposition: '{query}'")
            print("=" * 70)

            result = self.query_decomposer.execute_decomposed_query(query)

            print("\n✅ Complete Response:")
            print("=" * 50)
            print(result)
            print()

        except Exception as e:
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

    def get_performance_metrics(self):
        """Get performance metrics for the current session"""
        metrics = {
            'database_type': 'Unknown',
            'embedding_device': 'Unknown',
            'tools_count': len(self.tools) if self.tools else 0,
            'parallel_processing': hasattr(self, 'async_tools') and self.async_tools is not None
        }

        # Database metrics
        if hasattr(self, 'db_manager'):
            conn_info = self.db_manager.get_connection_info()
            metrics['database_type'] = conn_info.get('type', 'Unknown')

        # LLM metrics
        if hasattr(Settings, 'embed_model'):
            try:
                device = getattr(Settings.embed_model, 'device', 'Unknown')
                metrics['embedding_device'] = str(device)
            except:
                metrics['embedding_device'] = 'CPU (fallback)'

        return metrics

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
    """Main function"""
    print("🚀 Customer Service AI Agent")
    print("Connecting to MySQL database and setting up AI tools...")

    try:
        agent = CustomerServiceAgent()
        agent.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()