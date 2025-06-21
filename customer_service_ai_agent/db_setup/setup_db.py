#!/usr/bin/env python3
"""
Database Setup Script for Customer Service AI Agent
This script executes the SQL file to create the MySQL database and tables with sample data
"""

import mysql.connector
from mysql.connector import Error
import os
import sys

def create_database_connection():
    """Create connection to MySQL server"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='auburn'  # Your MySQL password
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def drop_existing_database(connection):
    """Drop the existing database if it exists"""
    try:
        cursor = connection.cursor()
        print("🗑️  Dropping existing database (if exists)...")

        # Drop database if it exists
        cursor.execute("DROP DATABASE IF EXISTS customer_service_db")
        connection.commit()

        print("✅ Existing database dropped successfully!")

    except Error as e:
        print(f"Warning: Could not drop existing database: {e}")
    finally:
        if cursor:
            cursor.close()

def execute_sql_file(connection, sql_file_path):
    """Execute SQL commands from file"""
    try:
        cursor = connection.cursor()

        # First, drop the existing database to ensure clean setup
        drop_existing_database(connection)

        # Read the SQL file
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_script = file.read()

        # Split the script into individual statements
        # Handle multi-line statements properly
        sql_commands = []
        current_command = ""

        for line in sql_script.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('--') or not line:
                continue

            current_command += line + ' '

            # If line ends with semicolon, it's the end of a command
            if line.endswith(';'):
                sql_commands.append(current_command.strip())
                current_command = ""

        # Execute each command
        print("🔧 Executing SQL commands...")
        command_count = 0

        for command in sql_commands:
            if command:  # Skip empty commands
                try:
                    # Execute the command
                    cursor.execute(command)
                    command_count += 1

                    # Fetch results if any (for SELECT statements)
                    if cursor.description:
                        results = cursor.fetchall()
                        if results:
                            for result in results:
                                print(f"  {result[0]}")

                    # Commit after each successful command
                    connection.commit()

                except Error as e:
                    print(f"Error executing command {command_count + 1}: {e}")
                    print(f"Command: {command[:100]}...")
                    # Continue with next command instead of stopping

        print(f"Successfully executed {command_count} SQL commands!")

    except Error as e:
        print(f"Error executing SQL file: {e}")
        return False
    except FileNotFoundError:
        print(f"SQL file not found: {sql_file_path}")
        print("Make sure 'customer_service_db.sql' is in the same directory as this script.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        if cursor:
            cursor.close()

    return True

def test_database_connection():
    """Test the database connection and show sample data"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='auburn',
            database='customer_service_db'
        )

        cursor = connection.cursor()

        # Test queries matching the original notebook functions
        print("\n=== Testing Database Queries ===")

        # Test get_order_items equivalent
        cursor.execute("""
                       SELECT p.product_name
                       FROM order_items oi
                                JOIN products p ON oi.product_id = p.product_id
                       WHERE oi.order_id = 1001
                       """)
        items = cursor.fetchall()
        print(f"Order 1001 items: {[item[0] for item in items]}")

        # Test get_delivery_date equivalent
        cursor.execute("SELECT DATE_FORMAT(delivery_date, '%d-%b') FROM orders WHERE order_id = 1001")
        delivery = cursor.fetchone()
        print(f"Order 1001 delivery date: {delivery[0] if delivery else 'Not found'}")

        # Test get_item_return_days equivalent
        cursor.execute("SELECT return_days FROM products WHERE product_name LIKE '%Laptop%' LIMIT 1")
        return_days = cursor.fetchone()
        print(f"Laptop return days: {return_days[0] if return_days else 'Not found'}")

        print("\n=== Database Summary ===")
        cursor.execute("SELECT COUNT(*) FROM categories")
        print(f"Total Categories: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM products")
        print(f"Total Products: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM customers")
        print(f"Total Customers: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM orders")
        print(f"Total Orders: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM order_items")
        print(f"Total Order Items: {cursor.fetchone()[0]}")

        # Show some sample data from key tables
        print("\n=== Sample Data Verification ===")

        # Check original notebook orders exist
        cursor.execute("SELECT order_id, customer_id, status FROM orders WHERE order_id IN (1001, 1002, 1003)")
        original_orders = cursor.fetchall()
        for order in original_orders:
            print(f"Order {order[0]}: Customer {order[1]}, Status: {order[2]}")

        cursor.close()
        connection.close()

        return True

    except Error as e:
        print(f"Database test failed: {e}")
        return False

def find_sql_file():
    """Find the SQL file in the current directory or common locations"""
    possible_paths = [
        'customer_service_db.sql',
        './customer_service_db.sql',
        '../customer_service_db.sql',
        'sql/customer_service_db.sql'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def main():
    """Main function to set up the database"""
    print("🚀 Setting up Customer Service Database using SQL file...")
    print("⚠️  This will DROP and RECREATE the entire database!")

    # Ask for confirmation
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Database setup cancelled.")
        return

    # Find the SQL file
    sql_file_path = find_sql_file()
    if not sql_file_path:
        print("❌ Error: customer_service_db.sql file not found!")
        print("Please make sure the SQL file is in the same directory as this script.")
        print("\nCurrent directory contents:")
        for file in os.listdir('.'):
            if file.endswith('.sql'):
                print(f"  Found SQL file: {file}")
        return

    print(f"📁 Found SQL file: {sql_file_path}")

    # Create connection
    connection = create_database_connection()
    if not connection:
        print("❌ Failed to connect to MySQL server. Please check your credentials.")
        print("Make sure MySQL is running and your password is 'auburn'")
        return

    try:
        # Execute the SQL file (which includes dropping existing database)
        print(f"\n🔄 Executing SQL file: {sql_file_path}")
        if execute_sql_file(connection, sql_file_path):
            print("✅ SQL file executed successfully!")

            # Test the database
            print("\n🔍 Testing database setup...")
            if test_database_connection():
                print("\n🎉 Database setup completed successfully!")
                print("📊 Fresh database created with all sample data!")
                print("You can now run the customer service AI agent script.")
            else:
                print("\n❌ Database setup failed during testing.")
        else:
            print("❌ Failed to execute SQL file.")

    except Exception as e:
        print(f"Error during setup: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    main()