import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
from database import get_connection, save_recommendation
from agents import agent_system

def generate_all_recommendations():
    """Generate recommendations for all customers using all algorithms"""
    conn = get_connection()
    customers = pd.read_sql("SELECT id FROM customers", conn)
    conn.close()
    
    algorithms = ["popularity", "collaborative", "content", "hybrid"]
    results = {}
    
    for _, customer in customers.iterrows():
        customer_id = customer['id']
        customer_results = {}
        
        for algorithm in algorithms:
            recommendations = agent_system.get_recommendations(
                customer_id, 
                algorithm=algorithm,
                limit=10
            )
            customer_results[algorithm] = recommendations
        
        results[customer_id] = customer_results
    
    return results

def segment_customers():
    """Segment customers based on their behavior and preferences"""
    conn = get_connection()
    
    # Get all customers
    customers = pd.read_sql("SELECT * FROM customers", conn)
    
    # Get interaction data
    interactions = pd.read_sql("""
    SELECT customer_id, product_id, interaction_type, COUNT(*) as count
    FROM interactions
    GROUP BY customer_id, product_id, interaction_type
    """, conn)
    
    # Get products
    products = pd.read_sql("SELECT * FROM products", conn)
    
    # Create feature vectors for each customer
    customer_features = []
    customer_ids = []
    
    for _, customer in customers.iterrows():
        customer_id = customer['id']
        customer_ids.append(customer_id)
        
        # Base features
        features = {}
        
        # Add demographic features
        features['age'] = customer['age'] if pd.notna(customer['age']) else 0
        features['gender_male'] = 1 if customer['gender'] == 'Male' else 0
        features['gender_female'] = 1 if customer['gender'] == 'Female' else 0
        features['gender_other'] = 1 if customer['gender'] == 'Other' else 0
        
        # Add interaction-based features
        customer_interactions = interactions[interactions['customer_id'] == customer_id]
        
        # Total interactions count
        features['total_interactions'] = len(customer_interactions)
        
        # Interaction types count
        for itype in ['view', 'add_to_cart', 'purchase', 'review']:
            type_count = customer_interactions[customer_interactions['interaction_type'] == itype]['count'].sum()
            features[f'interaction_{itype}'] = type_count
        
        # Products interacted with by category
        for category in products['category'].unique():
            category_products = products[products['category'] == category]['id'].tolist()
            category_interactions = customer_interactions[customer_interactions['product_id'].isin(category_products)]
            features[f'category_{category}'] = len(category_interactions)
        
        customer_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(customer_features, index=customer_ids)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    
    # Determine optimal number of clusters
    # This is a simplified approach, in a real system you would use methods like elbow method
    # or silhouette score to determine the optimal number of clusters
    num_clusters = min(5, len(scaled_features))  # Maximum of 5 clusters
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Map cluster numbers to meaningful segment names
    segment_names = {
        0: "New Visitor",
        1: "Occasional Buyer",
        2: "Regular Customer",
        3: "Loyal Customer",
        4: "VIP"
    }
    
    # Update customer segments in database
    cursor = conn.cursor()
    
    for i, customer_id in enumerate(customer_ids):
        cluster = clusters[i]
        segment = segment_names.get(cluster, f"Segment {cluster}")
        
        cursor.execute("""
        UPDATE customers
        SET segment = ?
        WHERE id = ?
        """, (segment, customer_id))
    
    conn.commit()
    conn.close()
    
    return {"success": True, "clusters": num_clusters}

def evaluate_recommendations(customer_id):
    """Evaluate different recommendation algorithms for a specific customer"""
    conn = get_connection()
    
    # Get customer information
    customer = pd.read_sql("SELECT * FROM customers WHERE id = ?", conn, params=(customer_id,))
    
    if customer.empty:
        conn.close()
        return {"error": "Customer not found"}
    
    # Get customer's interaction history
    interactions = pd.read_sql("""
    SELECT i.*, p.name as product_name, p.category
    FROM interactions i
    JOIN products p ON i.product_id = p.id
    WHERE i.customer_id = ?
    ORDER BY i.timestamp DESC
    """, conn, params=(customer_id,))
    
    # Generate recommendations using different algorithms
    algorithms = ["popularity", "collaborative", "content", "hybrid"]
    all_recommendations = {}
    
    for algorithm in algorithms:
        recommendations = agent_system.get_recommendations(
            customer_id, 
            algorithm=algorithm,
            limit=10
        )
        all_recommendations[algorithm] = recommendations
    
    # Evaluate recommendations based on customer's interaction history
    evaluation = {}
    
    if not interactions.empty:
        # Create sets of products the customer has interacted with by interaction type
        viewed_products = set(interactions[interactions['interaction_type'] == 'view']['product_id'])
        cart_products = set(interactions[interactions['interaction_type'] == 'add_to_cart']['product_id'])
        purchased_products = set(interactions[interactions['interaction_type'] == 'purchase']['product_id'])
        
        # Preferred categories based on interactions
        category_counts = interactions['category'].value_counts()
        preferred_categories = category_counts[category_counts > 1].index.tolist()
        
        for algorithm, recommendations in all_recommendations.items():
            recommended_products = [rec['product_id'] for rec in recommendations]
            
            # Get categories of recommended products
            recommended_product_ids = tuple(recommended_products) if recommended_products else (-1,)
            recommended_categories = pd.read_sql(
                f"SELECT id, category FROM products WHERE id IN {recommended_product_ids}",
                conn
            )
            
            # Calculate evaluation metrics
            # 1. Overlap with viewed products
            viewed_overlap = len(viewed_products.intersection(recommended_products))
            
            # 2. Overlap with cart products
            cart_overlap = len(cart_products.intersection(recommended_products))
            
            # 3. Overlap with purchased products
            purchased_overlap = len(purchased_products.intersection(recommended_products))
            
            # 4. Category relevance
            if preferred_categories and not recommended_categories.empty:
                category_matches = recommended_categories[recommended_categories['category'].isin(preferred_categories)]
                category_relevance = len(category_matches) / len(recommended_products) if recommended_products else 0
            else:
                category_relevance = 0
            
            # Calculate overall score
            # Using different weights for each metric based on importance
            overall_score = (
                0.1 * viewed_overlap +
                0.3 * cart_overlap +
                0.4 * purchased_overlap +
                0.2 * category_relevance
            )
            
            evaluation[algorithm] = {
                "viewed_overlap": viewed_overlap,
                "cart_overlap": cart_overlap,
                "purchased_overlap": purchased_overlap,
                "category_relevance": category_relevance,
                "overall_score": overall_score
            }
    else:
        # If no interaction history, give basic scores
        for algorithm in algorithms:
            evaluation[algorithm] = {
                "viewed_overlap": 0,
                "cart_overlap": 0,
                "purchased_overlap": 0,
                "category_relevance": 0,
                "overall_score": 0
            }
    
    conn.close()
    
    return {
        "customer_id": customer_id,
        "customer_name": customer.iloc[0]['name'] if not customer.empty else "Unknown",
        "segment": customer.iloc[0]['segment'] if not customer.empty else "None",
        "interaction_count": len(interactions),
        "recommendations": all_recommendations,
        "evaluation": evaluation
    }

def get_recommendation_metrics():
    """Get metrics on recommendations generated by the system"""
    conn = get_connection()
    
    # Count recommendations by algorithm
    algorithm_counts = pd.read_sql("""
    SELECT algorithm, COUNT(*) as count
    FROM recommendations
    GROUP BY algorithm
    """, conn)
    
    # Get latest recommendations
    latest_recommendations = pd.read_sql("""
    SELECT r.*, c.name as customer_name, p.name as product_name
    FROM recommendations r
    JOIN customers c ON r.customer_id = c.id
    JOIN products p ON r.product_id = p.id
    ORDER BY r.timestamp DESC
    LIMIT 100
    """, conn)
    
    # Get average recommendation score by algorithm
    avg_scores = pd.read_sql("""
    SELECT algorithm, AVG(score) as avg_score
    FROM recommendations
    GROUP BY algorithm
    """, conn)
    
    # Get recommendations by customer segment
    segment_recommendations = pd.read_sql("""
    SELECT c.segment, r.algorithm, COUNT(*) as count
    FROM recommendations r
    JOIN customers c ON r.customer_id = c.id
    GROUP BY c.segment, r.algorithm
    """, conn)
    
    conn.close()
    
    return {
        "algorithm_counts": algorithm_counts.to_dict('records'),
        "latest_recommendations": latest_recommendations.to_dict('records'),
        "avg_scores": avg_scores.to_dict('records'),
        "segment_recommendations": segment_recommendations.to_dict('records')
    }
