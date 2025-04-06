import pandas as pd
import json
from database import get_connection, add_customer, add_product, add_interaction

def import_customer_data(file_path, limit=None):
    """Import customer data from CSV file and save to database"""
    # Read customer data
    df = pd.read_csv(file_path)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
    
    # Process and import each customer
    imported_count = 0
    for _, row in df.iterrows():
        try:
            # Parse list columns (wrap in try-except as they might not be valid Python literals)
            try:
                browsing_history = eval(row['Browsing_History']) if isinstance(row['Browsing_History'], str) else row['Browsing_History']
            except:
                browsing_history = []
                
            try:
                purchase_history = eval(row['Purchase_History']) if isinstance(row['Purchase_History'], str) else row['Purchase_History']
            except:
                purchase_history = []
            
            # Create preferences dict
            preferences = {
                "browsing_history": browsing_history,
                "purchase_history": purchase_history,
                "avg_order_value": float(row['Avg_Order_Value']) if 'Avg_Order_Value' in row else 0,
                "season_preference": row['Season'] if 'Season' in row else None,
                "holiday_preference": row['Holiday'] if 'Holiday' in row else "No"
            }
            
            # Add customer to database
            customer_id = add_customer(
                name=f"Customer {row['Customer_ID']}",
                email=f"{row['Customer_ID'].lower()}@example.com",
                age=int(row['Age']) if 'Age' in row else None,
                gender=row['Gender'] if 'Gender' in row else None,
                location=row['Location'] if 'Location' in row else None,
                segment=row['Customer_Segment'] if 'Customer_Segment' in row else "New Visitor",
                preferences=str(preferences)
            )
            
            imported_count += 1
            
        except Exception as e:
            print(f"Error importing customer {row['Customer_ID']}: {str(e)}")
    
    return imported_count

def import_product_data(file_path, limit=None):
    """Import product data from CSV file and save to database"""
    # Read product data
    df = pd.read_csv(file_path)
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
    
    # Process and import each product
    imported_count = 0
    for _, row in df.iterrows():
        try:
            # Parse list columns
            try:
                similar_products = eval(row['Similar_Product_List']) if isinstance(row['Similar_Product_List'], str) else row['Similar_Product_List']
            except:
                similar_products = []
            
            # Create attributes dict
            attributes = {
                "brand": row['Brand'] if 'Brand' in row else None,
                "product_rating": float(row['Product_Rating']) if 'Product_Rating' in row else None,
                "avg_similar_rating": float(row['Average_Rating_of_Similar_Products']) if 'Average_Rating_of_Similar_Products' in row else None,
                "sentiment_score": float(row['Customer_Review_Sentiment_Score']) if 'Customer_Review_Sentiment_Score' in row else None,
                "similar_products": similar_products,
                "seasonal_relevance": row['Season'] if 'Season' in row else None,
                "holiday_relevance": row['Holiday'] if 'Holiday' in row else "No",
                "geographical_relevance": row['Geographical_Location'] if 'Geographical_Location' in row else None,
                "recommendation_probability": float(row['Probability_of_Recommendation']) if 'Probability_of_Recommendation' in row else 0.5
            }
            
            # Add product to database
            product_id = add_product(
                name=f"{row['Subcategory']} {row['Product_ID']}",
                category=row['Category'] if 'Category' in row else "Other",
                price=float(row['Price']) if 'Price' in row else 0,
                description=f"A {row['Subcategory']} product from {row['Brand']} with ID {row['Product_ID']}",
                attributes=str(attributes)
            )
            
            imported_count += 1
            
        except Exception as e:
            print(f"Error importing product {row['Product_ID'] if 'Product_ID' in row else 'unknown'}: {str(e)}")
    
    return imported_count

def generate_interactions_from_history():
    """Generate interactions based on customer browsing and purchase history"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all customers
    cursor.execute("SELECT id, preferences FROM customers")
    customers = cursor.fetchall()
    
    # Get products by category for matching
    products_by_category = {}
    cursor.execute("SELECT id, category FROM products")
    for product_id, category in cursor.fetchall():
        if category not in products_by_category:
            products_by_category[category] = []
        products_by_category[category].append(product_id)
    
    # Generate interactions count
    interaction_count = 0
    
    for customer_id, preferences_str in customers:
        try:
            # Parse preferences
            preferences = eval(preferences_str) if preferences_str else {}
            
            # Process browsing history as views
            if 'browsing_history' in preferences and preferences['browsing_history']:
                for category in preferences['browsing_history']:
                    if category in products_by_category and products_by_category[category]:
                        # Get random product from category
                        import random
                        product_id = random.choice(products_by_category[category])
                        
                        # Add view interaction
                        add_interaction(
                            customer_id=customer_id,
                            product_id=product_id,
                            interaction_type="view",
                            details=None
                        )
                        interaction_count += 1
            
            # Process purchase history as purchases
            if 'purchase_history' in preferences and preferences['purchase_history']:
                for subcategory in preferences['purchase_history']:
                    # Find matching product by name (which includes subcategory)
                    cursor.execute("SELECT id FROM products WHERE name LIKE ?", (f"%{subcategory}%",))
                    product_matches = cursor.fetchall()
                    
                    if product_matches:
                        import random
                        product_id = random.choice(product_matches)[0]
                        
                        # Add purchase interaction
                        add_interaction(
                            customer_id=customer_id,
                            product_id=product_id,
                            interaction_type="purchase",
                            details=str({"amount": 1})
                        )
                        interaction_count += 1
                        
                        # Also add view interaction (since they must have viewed it to purchase)
                        add_interaction(
                            customer_id=customer_id,
                            product_id=product_id,
                            interaction_type="view",
                            details=None
                        )
                        interaction_count += 1
                        
        except Exception as e:
            print(f"Error generating interactions for customer {customer_id}: {str(e)}")
    
    conn.close()
    return interaction_count

def import_all_data(customer_file, product_file, limit=100):
    """Import data from CSV files and generate interactions"""
    # Import customers and products
    customer_count = import_customer_data(customer_file, limit)
    product_count = import_product_data(product_file, limit)
    
    # Generate interactions based on history
    interaction_count = generate_interactions_from_history()
    
    return {
        "customer_count": customer_count,
        "product_count": product_count,
        "interaction_count": interaction_count
    }

# Example usage
if __name__ == "__main__":
    results = import_all_data(
        "attached_assets/customer_data_collection.csv",
        "attached_assets/product_recommendation_data.csv",
        limit=100  # Import only 100 records for testing
    )
    print(f"Imported {results['customer_count']} customers")
    print(f"Imported {results['product_count']} products")
    print(f"Generated {results['interaction_count']} interactions")