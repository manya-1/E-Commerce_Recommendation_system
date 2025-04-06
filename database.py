import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Database path
DB_PATH = "ecommerce.db"

def get_connection():
    """Create and return a database connection"""
    return sqlite3.connect(DB_PATH)

def initialize_database():
    """Create database tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        description TEXT,
        attributes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create customers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER,
        gender TEXT,
        location TEXT,
        segment TEXT,
        preferences TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create interactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        interaction_type TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        details TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    # Create recommendations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        score REAL,
        algorithm TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def add_sample_data_if_empty():
    """Add sample data if the database is empty"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if tables are empty
    cursor.execute("SELECT COUNT(*) FROM products")
    product_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM customers")
    customer_count = cursor.fetchone()[0]
    
    # Only add sample data if tables are empty
    if product_count == 0 and customer_count == 0:
        # Sample product categories
        categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty", "Food"]
        
        # Sample products
        products = []
        for i in range(1, 31):
            category = random.choice(categories)
            price = round(random.uniform(10.0, 500.0), 2)
            
            # Generate attributes based on category
            if category == "Electronics":
                attrs = {"brand": random.choice(["Apple", "Samsung", "Sony", "LG"]), 
                         "color": random.choice(["Black", "White", "Silver"]),
                         "rating": round(random.uniform(3.0, 5.0), 1)}
            elif category == "Clothing":
                attrs = {"size": random.choice(["S", "M", "L", "XL"]), 
                         "color": random.choice(["Red", "Blue", "Black", "White", "Green"]),
                         "material": random.choice(["Cotton", "Polyester", "Wool"])}
            else:
                attrs = {"rating": round(random.uniform(3.0, 5.0), 1)}
            
            products.append({
                "name": f"Product {i}",
                "category": category,
                "price": price,
                "description": f"This is a {category.lower()} product with id {i}",
                "attributes": str(attrs)
            })
        
        # Sample customers with segmentation
        segments = ["New Visitor", "Occasional Buyer", "Regular Customer", "Loyal Customer", "VIP"]
        customers = []
        for i in range(1, 21):
            age = random.randint(18, 70)
            gender = random.choice(["Male", "Female", "Other"])
            segment = random.choice(segments)
            
            # Generate preferences based on segment
            prefs = {}
            if segment in ["Regular Customer", "Loyal Customer", "VIP"]:
                prefs["favorite_categories"] = random.sample(categories, random.randint(1, 3))
                prefs["price_sensitivity"] = random.choice(["Low", "Medium", "High"])
            
            customers.append({
                "name": f"Customer {i}",
                "email": f"customer{i}@example.com",
                "age": age,
                "gender": gender,
                "location": random.choice(["New York", "London", "Tokyo", "Paris", "Berlin"]),
                "segment": segment,
                "preferences": str(prefs)
            })
        
        # Insert sample products
        cursor.executemany('''
        INSERT INTO products (name, category, price, description, attributes)
        VALUES (:name, :category, :price, :description, :attributes)
        ''', products)
        
        # Insert sample customers
        cursor.executemany('''
        INSERT INTO customers (name, email, age, gender, location, segment, preferences)
        VALUES (:name, :email, :age, :gender, :location, :segment, :preferences)
        ''', customers)
        
        # Generate sample interactions
        interaction_types = ["view", "add_to_cart", "purchase", "review"]
        
        # Get customer and product IDs
        product_ids = [row[0] for row in cursor.execute("SELECT id FROM products").fetchall()]
        customer_ids = [row[0] for row in cursor.execute("SELECT id FROM customers").fetchall()]
        
        # Generate random interactions over the last 30 days
        interactions = []
        
        for _ in range(200):  # Generate 200 random interactions
            customer_id = random.choice(customer_ids)
            product_id = random.choice(product_ids)
            interaction_type = random.choice(interaction_types)
            
            # Random timestamp within the last 30 days
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate details based on interaction type
            details = {}
            if interaction_type == "review":
                details["rating"] = random.randint(1, 5)
                details["comment"] = f"Sample review comment for product {product_id}"
            elif interaction_type == "purchase":
                details["quantity"] = random.randint(1, 3)
                details["total_price"] = round(random.uniform(10.0, 500.0), 2)
            
            interactions.append({
                "customer_id": customer_id,
                "product_id": product_id,
                "interaction_type": interaction_type,
                "timestamp": timestamp_str,
                "details": str(details) if details else None
            })
        
        # Insert sample interactions
        cursor.executemany('''
        INSERT INTO interactions (customer_id, product_id, interaction_type, timestamp, details)
        VALUES (:customer_id, :product_id, :interaction_type, :timestamp, :details)
        ''', interactions)
        
        conn.commit()
    
    conn.close()

def get_products():
    """Get all products from the database"""
    conn = get_connection()
    products = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    return products

def get_customers():
    """Get all customers from the database"""
    conn = get_connection()
    customers = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return customers

def get_interactions():
    """Get all interactions from the database"""
    conn = get_connection()
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    conn.close()
    return interactions

def add_interaction(customer_id, product_id, interaction_type, details=None):
    """Add a new interaction to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO interactions (customer_id, product_id, interaction_type, details)
    VALUES (?, ?, ?, ?)
    ''', (customer_id, product_id, interaction_type, details))
    
    conn.commit()
    interaction_id = cursor.lastrowid
    conn.close()
    
    return interaction_id

def save_recommendation(customer_id, product_id, score, algorithm):
    """Save a product recommendation to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO recommendations (customer_id, product_id, score, algorithm)
    VALUES (?, ?, ?, ?)
    ''', (customer_id, product_id, score, algorithm))
    
    conn.commit()
    recommendation_id = cursor.lastrowid
    conn.close()
    
    return recommendation_id

def get_customer_recommendations(customer_id):
    """Get recommendations for a specific customer"""
    conn = get_connection()
    
    recommendations = pd.read_sql("""
    SELECT r.id, r.customer_id, r.product_id, p.name as product_name, 
           p.category, p.price, r.score, r.algorithm, r.timestamp
    FROM recommendations r
    JOIN products p ON r.product_id = p.id
    WHERE r.customer_id = ?
    ORDER BY r.score DESC
    """, conn, params=(customer_id,))
    
    conn.close()
    return recommendations

def update_customer_segment(customer_id, segment):
    """Update a customer's segment"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE customers
    SET segment = ?
    WHERE id = ?
    ''', (segment, customer_id))
    
    conn.commit()
    conn.close()

def add_product(name, category, price, description, attributes):
    """Add a new product to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO products (name, category, price, description, attributes)
    VALUES (?, ?, ?, ?, ?)
    ''', (name, category, price, description, attributes))
    
    conn.commit()
    product_id = cursor.lastrowid
    conn.close()
    
    return product_id

def add_customer(name, email, age, gender, location, segment, preferences):
    """Add a new customer to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO customers (name, email, age, gender, location, segment, preferences)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, age, gender, location, segment, preferences))
    
    conn.commit()
    customer_id = cursor.lastrowid
    conn.close()
    
    return customer_id
