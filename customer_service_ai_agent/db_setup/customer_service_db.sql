-- Customer Service Database Creation Script
-- Run this script in MySQL to create the database and tables

-- Create database
CREATE DATABASE IF NOT EXISTS customer_service_db;
USE customer_service_db;

-- Drop tables if they exist (to allow re-running the script)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;

-- Create categories table
CREATE TABLE categories (
                            category_id INT PRIMARY KEY AUTO_INCREMENT,
                            category_name VARCHAR(100) NOT NULL,
                            description TEXT
);

-- Create products table
CREATE TABLE products (
                          product_id INT PRIMARY KEY AUTO_INCREMENT,
                          product_name VARCHAR(255) NOT NULL,
                          category_id INT,
                          price DECIMAL(10, 2) NOT NULL,
                          return_days INT NOT NULL DEFAULT 30,
                          stock_quantity INT DEFAULT 0,
                          FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

-- Create customers table
CREATE TABLE customers (
                           customer_id INT PRIMARY KEY AUTO_INCREMENT,
                           first_name VARCHAR(100) NOT NULL,
                           last_name VARCHAR(100) NOT NULL,
                           email VARCHAR(255) UNIQUE NOT NULL,
                           phone VARCHAR(20),
                           address TEXT,
                           city VARCHAR(100),
                           state VARCHAR(50),
                           zip_code VARCHAR(10),
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
                        order_id INT PRIMARY KEY AUTO_INCREMENT,
                        customer_id INT NOT NULL,
                        order_date DATE NOT NULL,
                        delivery_date DATE,
                        status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
                        total_amount DECIMAL(10, 2),
                        shipping_address TEXT,
                        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create order_items table (junction table for orders and products)
CREATE TABLE order_items (
                             order_item_id INT PRIMARY KEY AUTO_INCREMENT,
                             order_id INT NOT NULL,
                             product_id INT NOT NULL,
                             quantity INT NOT NULL DEFAULT 1,
                             unit_price DECIMAL(10, 2) NOT NULL,
                             FOREIGN KEY (order_id) REFERENCES orders(order_id),
                             FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert sample data into categories
INSERT INTO categories (category_name, description) VALUES
                                                        ('Electronics', 'Electronic devices and accessories'),
                                                        ('Computers', 'Computer hardware and peripherals'),
                                                        ('Accessories', 'Computer and electronic accessories'),
                                                        ('Audio', 'Audio equipment and accessories'),
                                                        ('Gaming', 'Gaming equipment and accessories');

-- Insert sample data into products
INSERT INTO products (product_name, category_id, price, return_days, stock_quantity) VALUES
                                                                                         ('Dell Laptop XPS 13', 2, 1299.99, 30, 50),
                                                                                         ('MacBook Pro 14"', 2, 1999.99, 30, 30),
                                                                                         ('HP Pavilion Laptop', 2, 899.99, 30, 40),
                                                                                         ('Gaming Laptop ASUS ROG', 2, 1599.99, 30, 25),
                                                                                         ('Lenovo ThinkPad', 2, 1199.99, 30, 35),
                                                                                         ('Wireless Mouse Logitech', 3, 79.99, 15, 100),
                                                                                         ('Gaming Mouse Razer', 3, 129.99, 15, 80),
                                                                                         ('Optical Mouse HP', 3, 29.99, 15, 150),
                                                                                         ('Bluetooth Mouse Apple', 3, 99.99, 15, 60),
                                                                                         ('Ergonomic Mouse Microsoft', 3, 69.99, 15, 90),
                                                                                         ('Mechanical Keyboard Corsair', 3, 149.99, 15, 70),
                                                                                         ('Wireless Keyboard Logitech', 3, 89.99, 15, 85),
                                                                                         ('Gaming Keyboard Razer', 3, 179.99, 15, 55),
                                                                                         ('Compact Keyboard Apple', 3, 129.99, 15, 45),
                                                                                         ('Bluetooth Keyboard Microsoft', 3, 99.99, 15, 65),
                                                                                         ('HDMI Cable 6ft', 3, 19.99, 5, 200),
                                                                                         ('USB-C Cable', 3, 24.99, 5, 180),
                                                                                         ('DisplayPort Cable', 3, 29.99, 5, 120),
                                                                                         ('Ethernet Cable Cat6', 3, 15.99, 5, 250),
                                                                                         ('Audio Cable 3.5mm', 3, 12.99, 5, 300),
                                                                                         ('Wireless Headphones Sony', 4, 199.99, 20, 75),
                                                                                         ('Gaming Headset SteelSeries', 4, 159.99, 20, 60),
                                                                                         ('Bluetooth Speaker JBL', 4, 129.99, 20, 90),
                                                                                         ('Webcam Logitech HD', 1, 89.99, 25, 110),
                                                                                         ('USB Flash Drive 64GB', 1, 29.99, 10, 200),
                                                                                         ('External Hard Drive 1TB', 1, 119.99, 30, 85),
                                                                                         ('SSD External 500GB', 1, 149.99, 30, 70),
                                                                                         ('Monitor 24" Dell', 1, 299.99, 30, 40),
                                                                                         ('Monitor 27" LG', 1, 399.99, 30, 35),
                                                                                         ('Gaming Monitor 32" ASUS', 1, 599.99, 30, 25);

-- Insert sample data into customers
INSERT INTO customers (first_name, last_name, email, phone, address, city, state, zip_code) VALUES
                                                                                                ('John', 'Smith', 'john.smith@email.com', '555-0101', '123 Main St', 'Auburn', 'AL', '36830'),
                                                                                                ('Sarah', 'Johnson', 'sarah.johnson@email.com', '555-0102', '456 Oak Ave', 'Montgomery', 'AL', '36104'),
                                                                                                ('Michael', 'Brown', 'michael.brown@email.com', '555-0103', '789 Pine Rd', 'Birmingham', 'AL', '35203'),
                                                                                                ('Emily', 'Davis', 'emily.davis@email.com', '555-0104', '321 Elm St', 'Mobile', 'AL', '36601'),
                                                                                                ('David', 'Wilson', 'david.wilson@email.com', '555-0105', '654 Maple Dr', 'Huntsville', 'AL', '35801'),
                                                                                                ('Jessica', 'Martinez', 'jessica.martinez@email.com', '555-0106', '987 Cedar Ln', 'Tuscaloosa', 'AL', '35401'),
                                                                                                ('Christopher', 'Anderson', 'chris.anderson@email.com', '555-0107', '147 Birch St', 'Auburn', 'AL', '36832'),
                                                                                                ('Amanda', 'Taylor', 'amanda.taylor@email.com', '555-0108', '258 Walnut Ave', 'Dothan', 'AL', '36301'),
                                                                                                ('James', 'Thomas', 'james.thomas@email.com', '555-0109', '369 Cherry Rd', 'Florence', 'AL', '35630'),
                                                                                                ('Lisa', 'Jackson', 'lisa.jackson@email.com', '555-0110', '741 Spruce Dr', 'Gadsden', 'AL', '35901'),
                                                                                                ('Robert', 'White', 'robert.white@email.com', '555-0111', '852 Ash St', 'Anniston', 'AL', '36201'),
                                                                                                ('Michelle', 'Harris', 'michelle.harris@email.com', '555-0112', '963 Poplar Ave', 'Decatur', 'AL', '35601'),
                                                                                                ('Kevin', 'Clark', 'kevin.clark@email.com', '555-0113', '159 Hickory Ln', 'Auburn', 'AL', '36849'),
                                                                                                ('Nicole', 'Lewis', 'nicole.lewis@email.com', '555-0114', '357 Magnolia Dr', 'Opelika', 'AL', '36801'),
                                                                                                ('Daniel', 'Lee', 'daniel.lee@email.com', '555-0115', '468 Dogwood St', 'Prattville', 'AL', '36066'),
                                                                                                ('Ashley', 'Walker', 'ashley.walker@email.com', '555-0116', '579 Pecan Ave', 'Vestavia Hills', 'AL', '35216'),
                                                                                                ('Matthew', 'Hall', 'matthew.hall@email.com', '555-0117', '680 Sycamore Rd', 'Hoover', 'AL', '35244'),
                                                                                                ('Stephanie', 'Allen', 'stephanie.allen@email.com', '555-0118', '791 Willow Dr', 'Madison', 'AL', '35758'),
                                                                                                ('Jason', 'Young', 'jason.young@email.com', '555-0119', '802 Redwood St', 'Pelham', 'AL', '35124'),
                                                                                                ('Megan', 'King', 'megan.king@email.com', '555-0120', '913 Cypress Ave', 'Homewood', 'AL', '35209'),
                                                                                                ('Ryan', 'Wright', 'ryan.wright@email.com', '555-0121', '124 Palm Ln', 'Enterprise', 'AL', '36330'),
                                                                                                ('Lauren', 'Lopez', 'lauren.lopez@email.com', '555-0122', '235 Bamboo Dr', 'Albertville', 'AL', '35950'),
                                                                                                ('Brandon', 'Hill', 'brandon.hill@email.com', '555-0123', '346 Fern St', 'Alexander City', 'AL', '35010'),
                                                                                                ('Kayla', 'Green', 'kayla.green@email.com', '555-0124', '457 Moss Ave', 'Athens', 'AL', '35611'),
                                                                                                ('Tyler', 'Adams', 'tyler.adams@email.com', '555-0125', '568 Ivy Rd', 'Auburn', 'AL', '36830'),
                                                                                                ('Rachel', 'Baker', 'rachel.baker@email.com', '555-0126', '679 Rose Dr', 'Cullman', 'AL', '35055'),
                                                                                                ('Jonathan', 'Gonzalez', 'jonathan.gonzalez@email.com', '555-0127', '780 Lily St', 'Fairhope', 'AL', '36532'),
                                                                                                ('Samantha', 'Nelson', 'samantha.nelson@email.com', '555-0128', '891 Daisy Ave', 'Gulf Shores', 'AL', '36542'),
                                                                                                ('Andrew', 'Carter', 'andrew.carter@email.com', '555-0129', '902 Tulip Ln', 'Jasper', 'AL', '35501'),
                                                                                                ('Brittany', 'Mitchell', 'brittany.mitchell@email.com', '555-0130', '013 Orchid Dr', 'Scottsboro', 'AL', '35768');

-- Insert sample data into orders (using the original order IDs from the notebook for compatibility)
INSERT INTO orders (order_id, customer_id, order_date, delivery_date, status, total_amount, shipping_address) VALUES
                                                                                                                  (1001, 1, '2024-06-05', '2024-06-10', 'delivered', 1379.98, '123 Main St, Auburn, AL 36830'),
                                                                                                                  (1002, 2, '2024-06-07', '2024-06-12', 'delivered', 239.98, '456 Oak Ave, Montgomery, AL 36104'),
                                                                                                                  (1003, 3, '2024-06-03', '2024-06-08', 'delivered', 1449.98, '789 Pine Rd, Birmingham, AL 35203'),
                                                                                                                  (1004, 4, '2024-06-08', '2024-06-13', 'shipped', 229.98, '321 Elm St, Mobile, AL 36601'),
                                                                                                                  (1005, 5, '2024-06-10', '2024-06-15', 'processing', 599.99, '654 Maple Dr, Huntsville, AL 35801'),
                                                                                                                  (1006, 6, '2024-06-12', '2024-06-17', 'pending', 319.97, '987 Cedar Ln, Tuscaloosa, AL 35401'),
                                                                                                                  (1007, 7, '2024-06-01', '2024-06-06', 'delivered', 2129.98, '147 Birch St, Auburn, AL 36832'),
                                                                                                                  (1008, 8, '2024-06-04', '2024-06-09', 'delivered', 179.98, '258 Walnut Ave, Dothan, AL 36301'),
                                                                                                                  (1009, 9, '2024-06-06', '2024-06-11', 'delivered', 449.98, '369 Cherry Rd, Florence, AL 35630'),
                                                                                                                  (1010, 10, '2024-06-09', '2024-06-14', 'shipped', 269.97, '741 Spruce Dr, Gadsden, AL 35901'),
                                                                                                                  (1011, 11, '2024-06-11', '2024-06-16', 'processing', 899.99, '852 Ash St, Anniston, AL 36201'),
                                                                                                                  (1012, 12, '2024-06-02', '2024-06-07', 'delivered', 359.97, '963 Poplar Ave, Decatur, AL 35601'),
                                                                                                                  (1013, 13, '2024-06-13', '2024-06-18', 'pending', 1599.99, '159 Hickory Ln, Auburn, AL 36849'),
                                                                                                                  (1014, 14, '2024-06-05', '2024-06-10', 'delivered', 129.99, '357 Magnolia Dr, Opelika, AL 36801'),
                                                                                                                  (1015, 15, '2024-06-07', '2024-06-12', 'delivered', 149.99, '468 Dogwood St, Prattville, AL 36066'),
                                                                                                                  (1016, 16, '2024-06-14', '2024-06-19', 'pending', 699.98, '579 Pecan Ave, Vestavia Hills, AL 35216'),
                                                                                                                  (1017, 17, '2024-06-08', '2024-06-13', 'shipped', 1319.98, '680 Sycamore Rd, Hoover, AL 35244'),
                                                                                                                  (1018, 18, '2024-06-10', '2024-06-15', 'processing', 89.99, '791 Willow Dr, Madison, AL 35758'),
                                                                                                                  (1019, 19, '2024-06-12', '2024-06-17', 'pending', 249.98, '802 Redwood St, Pelham, AL 35124'),
                                                                                                                  (1020, 20, '2024-06-03', '2024-06-08', 'delivered', 399.99, '913 Cypress Ave, Homewood, AL 35209'),
                                                                                                                  (1021, 21, '2024-06-15', '2024-06-20', 'pending', 179.98, '124 Palm Ln, Enterprise, AL 36330'),
                                                                                                                  (1022, 22, '2024-06-01', '2024-06-06', 'delivered', 529.97, '235 Bamboo Dr, Albertville, AL 35950'),
                                                                                                                  (1023, 23, '2024-06-04', '2024-06-09', 'delivered', 1229.98, '346 Fern St, Alexander City, AL 35010'),
                                                                                                                  (1024, 24, '2024-06-06', '2024-06-11', 'delivered', 89.99, '457 Moss Ave, Athens, AL 35611'),
                                                                                                                  (1025, 25, '2024-06-16', '2024-06-21', 'pending', 329.97, '568 Ivy Rd, Auburn, AL 36830'),
                                                                                                                  (1026, 26, '2024-06-09', '2024-06-14', 'shipped', 999.99, '679 Rose Dr, Cullman, AL 35055'),
                                                                                                                  (1027, 27, '2024-06-11', '2024-06-16', 'processing', 199.98, '780 Lily St, Fairhope, AL 36532'),
                                                                                                                  (1028, 28, '2024-06-13', '2024-06-18', 'pending', 149.99, '891 Daisy Ave, Gulf Shores, AL 36542'),
                                                                                                                  (1029, 29, '2024-06-17', '2024-06-22', 'pending', 719.98, '902 Tulip Ln, Jasper, AL 35501'),
                                                                                                                  (1030, 30, '2024-06-02', '2024-06-07', 'delivered', 79.99, '013 Orchid Dr, Scottsboro, AL 35768');

-- Insert sample data into order_items (matching the original notebook examples)
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Order 1001: Laptop + Mouse (matching notebook)
(1001, (SELECT product_id FROM products WHERE product_name = 'Dell Laptop XPS 13'), 1, 1299.99),
(1001, (SELECT product_id FROM products WHERE product_name = 'Wireless Mouse Logitech'), 1, 79.99),

-- Order 1002: Keyboard + HDMI Cable (matching notebook)
(1002, (SELECT product_id FROM products WHERE product_name = 'Mechanical Keyboard Corsair'), 1, 149.99),
(1002, (SELECT product_id FROM products WHERE product_name = 'HDMI Cable 6ft'), 1, 19.99),

-- Order 1003: Laptop + Keyboard (matching notebook)
(1003, (SELECT product_id FROM products WHERE product_name = 'Dell Laptop XPS 13'), 1, 1299.99),
(1003, (SELECT product_id FROM products WHERE product_name = 'Mechanical Keyboard Corsair'), 1, 149.99),

-- Additional order items for other orders
(1004, (SELECT product_id FROM products WHERE product_name = 'Wireless Headphones Sony'), 1, 199.99),
(1004, (SELECT product_id FROM products WHERE product_name = 'USB-C Cable'), 1, 24.99),

(1005, (SELECT product_id FROM products WHERE product_name = 'Gaming Monitor 32" ASUS'), 1, 599.99),

(1006, (SELECT product_id FROM products WHERE product_name = 'Webcam Logitech HD'), 1, 89.99),
(1006, (SELECT product_id FROM products WHERE product_name = 'USB Flash Drive 64GB'), 1, 29.99),
(1006, (SELECT product_id FROM products WHERE product_name = 'Bluetooth Speaker JBL'), 1, 129.99),

(1007, (SELECT product_id FROM products WHERE product_name = 'MacBook Pro 14"'), 1, 1999.99),
(1007, (SELECT product_id FROM products WHERE product_name = 'Bluetooth Mouse Apple'), 1, 99.99),

(1008, (SELECT product_id FROM products WHERE product_name = 'Gaming Headset SteelSeries'), 1, 159.99),
(1008, (SELECT product_id FROM products WHERE product_name = 'Ethernet Cable Cat6'), 1, 15.99),

(1009, (SELECT product_id FROM products WHERE product_name = 'Monitor 27" LG'), 1, 399.99),
(1009, (SELECT product_id FROM products WHERE product_name = 'Wireless Keyboard Logitech'), 1, 89.99),

(1010, (SELECT product_id FROM products WHERE product_name = 'Gaming Mouse Razer'), 1, 129.99),
(1010, (SELECT product_id FROM products WHERE product_name = 'Gaming Keyboard Razer'), 1, 179.99),
(1010, (SELECT product_id FROM products WHERE product_name = 'USB-C Cable'), 2, 24.99),

(1011, (SELECT product_id FROM products WHERE product_name = 'HP Pavilion Laptop'), 1, 899.99),

(1012, (SELECT product_id FROM products WHERE product_name = 'Monitor 24" Dell'), 1, 299.99),
(1012, (SELECT product_id FROM products WHERE product_name = 'DisplayPort Cable'), 1, 29.99),
(1012, (SELECT product_id FROM products WHERE product_name = 'USB-C Cable'), 1, 24.99),

(1013, (SELECT product_id FROM products WHERE product_name = 'Gaming Laptop ASUS ROG'), 1, 1599.99),

(1014, (SELECT product_id FROM products WHERE product_name = 'Bluetooth Speaker JBL'), 1, 129.99),

(1015, (SELECT product_id FROM products WHERE product_name = 'External Hard Drive 1TB'), 1, 119.99),

(1016, (SELECT product_id FROM products WHERE product_name = 'Gaming Monitor 32" ASUS'), 1, 599.99),
(1016, (SELECT product_id FROM products WHERE product_name = 'Optical Mouse HP'), 1, 29.99),

(1017, (SELECT product_id FROM products WHERE product_name = 'Lenovo ThinkPad'), 1, 1199.99),
(1017, (SELECT product_id FROM products WHERE product_name = 'Bluetooth Keyboard Microsoft'), 1, 99.99),

(1018, (SELECT product_id FROM products WHERE product_name = 'Wireless Keyboard Logitech'), 1, 89.99),

(1019, (SELECT product_id FROM products WHERE product_name = 'Wireless Headphones Sony'), 1, 199.99),
(1019, (SELECT product_id FROM products WHERE product_name = 'USB Flash Drive 64GB'), 2, 29.99),

(1020, (SELECT product_id FROM products WHERE product_name = 'Monitor 27" LG'), 1, 399.99),

(1021, (SELECT product_id FROM products WHERE product_name = 'Gaming Headset SteelSeries'), 1, 159.99),
(1021, (SELECT product_id FROM products WHERE product_name = 'Ethernet Cable Cat6'), 1, 15.99),

(1022, (SELECT product_id FROM products WHERE product_name = 'Gaming Laptop ASUS ROG'), 1, 1599.99),
(1022, (SELECT product_id FROM products WHERE product_name = 'Gaming Mouse Razer'), 3, 129.99),

(1023, (SELECT product_id FROM products WHERE product_name = 'MacBook Pro 14"'), 1, 1999.99),
(1023, (SELECT product_id FROM products WHERE product_name = 'Compact Keyboard Apple'), 2, 129.99),

(1024, (SELECT product_id FROM products WHERE product_name = 'Wireless Keyboard Logitech'), 1, 89.99),

(1025, (SELECT product_id FROM products WHERE product_name = 'Monitor 24" Dell'), 1, 299.99),
(1025, (SELECT product_id FROM products WHERE product_name = 'USB-C Cable'), 1, 24.99),

(1026, (SELECT product_id FROM products WHERE product_name = 'Dell Laptop XPS 13'), 1, 1299.99),

(1027, (SELECT product_id FROM products WHERE product_name = 'Wireless Mouse Logitech'), 2, 79.99),

(1028, (SELECT product_id FROM products WHERE product_name = 'SSD External 500GB'), 1, 149.99),

(1029, (SELECT product_id FROM products WHERE product_name = 'Gaming Monitor 32" ASUS'), 1, 599.99),
(1029, (SELECT product_id FROM products WHERE product_name = 'Bluetooth Keyboard Microsoft'), 1, 99.99),

(1030, (SELECT product_id FROM products WHERE product_name = 'Wireless Mouse Logitech'), 1, 79.99);

-- Display summary of created data
SELECT 'Database created successfully!' as Status;
SELECT COUNT(*) as 'Total Categories' FROM categories;
SELECT COUNT(*) as 'Total Products' FROM products;
SELECT COUNT(*) as 'Total Customers' FROM customers;
SELECT COUNT(*) as 'Total Orders' FROM orders;
SELECT COUNT(*) as 'Total Order Items' FROM order_items;