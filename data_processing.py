import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from database import get_connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_customer_behavior(customer_id=None):
    """
    Analyze customer behavior patterns
    If customer_id is provided, analyze only that customer, otherwise analyze all customers
    """
    conn = get_connection()
    
    # Query to get interactions with customer and product info
    if customer_id:
        interactions = pd.read_sql("""
        SELECT i.*, c.name as customer_name, c.segment, p.name as product_name, p.category, p.price
        FROM interactions i
        JOIN customers c ON i.customer_id = c.id
        JOIN products p ON i.product_id = p.id
        WHERE i.customer_id = ?
        ORDER BY i.timestamp
        """, conn, params=(customer_id,))
    else:
        interactions = pd.read_sql("""
        SELECT i.*, c.name as customer_name, c.segment, p.name as product_name, p.category, p.price
        FROM interactions i
        JOIN customers c ON i.customer_id = c.id
        JOIN products p ON i.product_id = p.id
        ORDER BY i.timestamp
        """, conn)
    
    # If no interactions found, return empty results
    if interactions.empty:
        conn.close()
        return {
            "interaction_count": 0,
            "customer_count": 0,
            "product_count": 0,
            "category_distribution": {},
            "interaction_types": {},
            "customer_segments": {},
            "time_analysis": {},
            "conversion_rate": 0
        }
    
    # Basic statistics
    interaction_count = len(interactions)
    customer_count = interactions['customer_id'].nunique()
    product_count = interactions['product_id'].nunique()
    
    # Analyze product categories
    category_distribution = interactions['category'].value_counts().to_dict()
    
    # Analyze interaction types
    interaction_types = interactions['interaction_type'].value_counts().to_dict()
    
    # Analyze customer segments
    customer_segments = interactions['segment'].value_counts().to_dict()
    
    # Time analysis
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
    interactions['hour'] = interactions['timestamp'].dt.hour
    interactions['day'] = interactions['timestamp'].dt.day_name()
    
    hourly_distribution = interactions['hour'].value_counts().to_dict()
    daily_distribution = interactions['day'].value_counts().to_dict()
    
    # Calculate conversion rates
    customer_view_counts = interactions[interactions['interaction_type'] == 'view']['customer_id'].value_counts()
    customer_purchase_counts = interactions[interactions['interaction_type'] == 'purchase']['customer_id'].value_counts()
    
    # Merge the series for calculating conversion rate
    conversion_df = pd.DataFrame({
        'views': customer_view_counts,
        'purchases': customer_purchase_counts
    }).fillna(0)
    
    # Calculate conversion rate as purchases / views
    if 'views' in conversion_df.columns and 'purchases' in conversion_df.columns:
        conversion_df['conversion_rate'] = conversion_df['purchases'] / conversion_df['views']
        avg_conversion_rate = conversion_df['conversion_rate'].mean()
    else:
        avg_conversion_rate = 0
    
    # Customer journey analysis
    journey_analysis = {}
    
    if customer_id:
        # For a specific customer, analyze their journey over time
        journey = interactions.sort_values('timestamp')
        
        # Convert journey to a list of events
        journey_events = []
        for _, row in journey.iterrows():
            journey_events.append({
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'interaction_type': row['interaction_type'],
                'product_name': row['product_name'],
                'category': row['category'],
                'price': float(row['price'])
            })
        
        # Calculate time between interactions
        if len(journey) > 1:
            journey['time_diff'] = journey['timestamp'].diff()
            avg_time_between_interactions = journey['time_diff'].mean().total_seconds() / 60  # in minutes
        else:
            avg_time_between_interactions = None
        
        # Calculate category preferences
        category_preferences = journey['category'].value_counts().to_dict()
        
        # Calculate price range preferences
        price_min = journey['price'].min()
        price_max = journey['price'].max()
        price_avg = journey['price'].mean()
        
        journey_analysis = {
            'journey_events': journey_events,
            'avg_time_between_interactions': avg_time_between_interactions,
            'category_preferences': category_preferences,
            'price_range': {
                'min': float(price_min),
                'max': float(price_max),
                'avg': float(price_avg)
            }
        }
    
    conn.close()
    
    # Build the analysis result
    result = {
        "interaction_count": interaction_count,
        "customer_count": customer_count,
        "product_count": product_count,
        "category_distribution": category_distribution,
        "interaction_types": interaction_types,
        "customer_segments": customer_segments,
        "time_analysis": {
            "hourly": hourly_distribution,
            "daily": daily_distribution
        },
        "conversion_rate": float(avg_conversion_rate)
    }
    
    if customer_id:
        result["customer_journey"] = journey_analysis
    
    return result

def analyze_product_relationships():
    """Analyze relationships between products based on customer interactions"""
    conn = get_connection()
    
    # Get all interactions
    interactions = pd.read_sql("""
    SELECT customer_id, product_id, interaction_type 
    FROM interactions
    """, conn)
    
    # Get all products
    products = pd.read_sql("SELECT * FROM products", conn)
    
    # If no data, return empty results
    if interactions.empty or products.empty:
        conn.close()
        return {
            "co_viewed_products": [],
            "co_purchased_products": [],
            "category_relationships": {},
            "product_clusters": []
        }
    
    # Find products that are frequently viewed together
    co_viewed = {}
    
    # Group interactions by customer
    customer_views = interactions[interactions['interaction_type'] == 'view'].groupby('customer_id')['product_id'].apply(list)
    
    # For each customer, find product pairs that were viewed
    for customer_id, viewed_products in customer_views.items():
        for i, product1 in enumerate(viewed_products):
            for product2 in viewed_products[i+1:]:
                # Create a key for the product pair
                pair_key = tuple(sorted([product1, product2]))
                
                if pair_key in co_viewed:
                    co_viewed[pair_key] += 1
                else:
                    co_viewed[pair_key] = 1
    
    # Sort co-viewed products by frequency
    co_viewed_sorted = sorted(co_viewed.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare results for co-viewed products
    co_viewed_results = []
    for (product1, product2), count in co_viewed_sorted[:20]:  # Top 20 co-viewed pairs
        product1_info = products[products['id'] == product1].iloc[0]
        product2_info = products[products['id'] == product2].iloc[0]
        
        co_viewed_results.append({
            "product1": {
                "id": int(product1),
                "name": product1_info['name'],
                "category": product1_info['category']
            },
            "product2": {
                "id": int(product2),
                "name": product2_info['name'],
                "category": product2_info['category']
            },
            "count": count
        })
    
    # Similar analysis for co-purchased products
    co_purchased = {}
    
    # Group purchases by customer
    customer_purchases = interactions[interactions['interaction_type'] == 'purchase'].groupby('customer_id')['product_id'].apply(list)
    
    # For each customer, find product pairs that were purchased
    for customer_id, purchased_products in customer_purchases.items():
        for i, product1 in enumerate(purchased_products):
            for product2 in purchased_products[i+1:]:
                # Create a key for the product pair
                pair_key = tuple(sorted([product1, product2]))
                
                if pair_key in co_purchased:
                    co_purchased[pair_key] += 1
                else:
                    co_purchased[pair_key] = 1
    
    # Sort co-purchased products by frequency
    co_purchased_sorted = sorted(co_purchased.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare results for co-purchased products
    co_purchased_results = []
    for (product1, product2), count in co_purchased_sorted[:20]:  # Top 20 co-purchased pairs
        try:
            product1_info = products[products['id'] == product1].iloc[0]
            product2_info = products[products['id'] == product2].iloc[0]
            
            co_purchased_results.append({
                "product1": {
                    "id": int(product1),
                    "name": product1_info['name'],
                    "category": product1_info['category']
                },
                "product2": {
                    "id": int(product2),
                    "name": product2_info['name'],
                    "category": product2_info['category']
                },
                "count": count
            })
        except IndexError:
            # Skip if product info not found
            continue
    
    # Analyze category relationships
    category_relationships = {}
    
    # For each co-viewed product pair, analyze category relationships
    for (product1, product2), count in co_viewed.items():
        try:
            category1 = products[products['id'] == product1]['category'].iloc[0]
            category2 = products[products['id'] == product2]['category'].iloc[0]
            
            if category1 != category2:  # Only consider different categories
                # Create a key for the category pair
                cat_pair_key = tuple(sorted([category1, category2]))
                
                if cat_pair_key in category_relationships:
                    category_relationships[cat_pair_key] += count
                else:
                    category_relationships[cat_pair_key] = count
        except IndexError:
            # Skip if product info not found
            continue
    
    # Sort category relationships by frequency
    category_relationships_sorted = sorted(category_relationships.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare results for category relationships
    category_results = {}
    for (category1, category2), count in category_relationships_sorted:
        key = f"{category1} - {category2}"
        category_results[key] = count
    
    conn.close()
    
    return {
        "co_viewed_products": co_viewed_results,
        "co_purchased_products": co_purchased_results,
        "category_relationships": category_results
    }

def generate_customer_insights():
    """Generate insights about customers based on their interactions"""
    conn = get_connection()
    
    # Get customer data
    customers = pd.read_sql("SELECT * FROM customers", conn)
    
    # Get interaction data
    interactions = pd.read_sql("""
    SELECT i.*, p.category, p.price
    FROM interactions i
    JOIN products p ON i.product_id = p.id
    """, conn)
    
    # If no data, return empty results
    if customers.empty or interactions.empty:
        conn.close()
        return {
            "segment_insights": [],
            "demographic_insights": [],
            "value_insights": []
        }
    
    # Merge customer and interaction data
    merged_data = pd.merge(
        interactions,
        customers,
        left_on='customer_id',
        right_on='id',
        suffixes=('_interaction', '_customer')
    )
    
    # Segment insights
    segment_insights = []
    
    for segment in customers['segment'].unique():
        segment_data = merged_data[merged_data['segment'] == segment]
        
        if not segment_data.empty:
            # Calculate interaction counts by type
            interaction_counts = segment_data['interaction_type'].value_counts()
            
            # Calculate preferred categories
            category_counts = segment_data['category'].value_counts()
            top_categories = category_counts.head(3).index.tolist()
            
            # Calculate average price of products interacted with
            avg_price = segment_data['price'].mean()
            
            # Calculate conversion rate (purchases / views)
            views = len(segment_data[segment_data['interaction_type'] == 'view'])
            purchases = len(segment_data[segment_data['interaction_type'] == 'purchase'])
            conversion_rate = purchases / views if views > 0 else 0
            
            segment_insights.append({
                "segment": segment,
                "customer_count": len(segment_data['id_customer'].unique()),
                "interaction_counts": interaction_counts.to_dict(),
                "top_categories": top_categories,
                "avg_price": float(avg_price),
                "conversion_rate": float(conversion_rate)
            })
    
    # Demographic insights
    demographic_insights = []
    
    # Age group analysis
    if 'age' in customers.columns:
        # Create age groups
        customers['age_group'] = pd.cut(customers['age'], 
                                        bins=[0, 18, 30, 45, 60, 100],
                                        labels=['Under 18', '18-30', '31-45', '46-60', 'Over 60'])
        
        for age_group in customers['age_group'].dropna().unique():
            age_customers = customers[customers['age_group'] == age_group]['id'].tolist()
            age_interactions = merged_data[merged_data['id_customer'].isin(age_customers)]
            
            if not age_interactions.empty:
                # Calculate preferred categories
                category_counts = age_interactions['category'].value_counts()
                top_categories = category_counts.head(3).index.tolist()
                
                # Calculate average price
                avg_price = age_interactions['price'].mean()
                
                demographic_insights.append({
                    "demographic": f"Age: {age_group}",
                    "customer_count": len(age_customers),
                    "top_categories": top_categories,
                    "avg_price": float(avg_price)
                })
    
    # Gender analysis
    if 'gender' in customers.columns:
        for gender in customers['gender'].dropna().unique():
            gender_customers = customers[customers['gender'] == gender]['id'].tolist()
            gender_interactions = merged_data[merged_data['id_customer'].isin(gender_customers)]
            
            if not gender_interactions.empty:
                # Calculate preferred categories
                category_counts = gender_interactions['category'].value_counts()
                top_categories = category_counts.head(3).index.tolist()
                
                # Calculate average price
                avg_price = gender_interactions['price'].mean()
                
                demographic_insights.append({
                    "demographic": f"Gender: {gender}",
                    "customer_count": len(gender_customers),
                    "top_categories": top_categories,
                    "avg_price": float(avg_price)
                })
    
    # Location analysis
    if 'location' in customers.columns:
        for location in customers['location'].dropna().unique():
            location_customers = customers[customers['location'] == location]['id'].tolist()
            location_interactions = merged_data[merged_data['id_customer'].isin(location_customers)]
            
            if not location_interactions.empty:
                # Calculate preferred categories
                category_counts = location_interactions['category'].value_counts()
                top_categories = category_counts.head(3).index.tolist()
                
                # Calculate average price
                avg_price = location_interactions['price'].mean()
                
                demographic_insights.append({
                    "demographic": f"Location: {location}",
                    "customer_count": len(location_customers),
                    "top_categories": top_categories,
                    "avg_price": float(avg_price)
                })
    
    # Customer value insights
    value_insights = []
    
    # Calculate purchase value for each customer
    purchase_data = merged_data[merged_data['interaction_type'] == 'purchase']
    
    if not purchase_data.empty:
        customer_purchases = purchase_data.groupby('id_customer')
        
        purchase_values = {}
        for customer_id, group in customer_purchases:
            purchase_values[customer_id] = group['price'].sum()
        
        # Sort customers by purchase value
        sorted_customers = sorted(purchase_values.items(), key=lambda x: x[1], reverse=True)
        
        # Top 20% of customers (high value)
        top_20_percent = sorted_customers[:int(len(sorted_customers) * 0.2)]
        top_20_ids = [customer_id for customer_id, _ in top_20_percent]
        
        # Rest of customers (medium to low value)
        rest_ids = [customer_id for customer_id, _ in sorted_customers[int(len(sorted_customers) * 0.2):]]
        
        # Analyze high value customers
        if top_20_ids:
            high_value_data = merged_data[merged_data['id_customer'].isin(top_20_ids)]
            
            # Calculate preferred categories
            category_counts = high_value_data['category'].value_counts()
            top_categories = category_counts.head(3).index.tolist()
            
            # Calculate average price
            avg_price = high_value_data['price'].mean()
            
            # Calculate segments distribution
            segment_counts = high_value_data['segment'].value_counts().to_dict()
            
            value_insights.append({
                "value_group": "High Value (Top 20%)",
                "customer_count": len(top_20_ids),
                "avg_purchase_value": float(sum(v for _, v in top_20_percent) / len(top_20_percent)),
                "top_categories": top_categories,
                "avg_price": float(avg_price),
                "segment_distribution": segment_counts
            })
        
        # Analyze rest of customers
        if rest_ids:
            rest_data = merged_data[merged_data['id_customer'].isin(rest_ids)]
            
            # Calculate preferred categories
            category_counts = rest_data['category'].value_counts()
            top_categories = category_counts.head(3).index.tolist()
            
            # Calculate average price
            avg_price = rest_data['price'].mean()
            
            # Calculate segments distribution
            segment_counts = rest_data['segment'].value_counts().to_dict()
            
            rest_purchases = [v for cid, v in sorted_customers[int(len(sorted_customers) * 0.2):]]
            
            value_insights.append({
                "value_group": "Medium to Low Value (Bottom 80%)",
                "customer_count": len(rest_ids),
                "avg_purchase_value": float(sum(rest_purchases) / len(rest_purchases)) if rest_purchases else 0,
                "top_categories": top_categories,
                "avg_price": float(avg_price),
                "segment_distribution": segment_counts
            })
    
    conn.close()
    
    return {
        "segment_insights": segment_insights,
        "demographic_insights": demographic_insights,
        "value_insights": value_insights
    }

def calculate_product_similarities():
    """Calculate similarities between products based on attributes and customer interactions"""
    conn = get_connection()
    
    # Get all products
    products = pd.read_sql("SELECT * FROM products", conn)
    
    # Get all interactions
    interactions = pd.read_sql("SELECT * FROM interactions", conn)
    
    if products.empty:
        conn.close()
        return {"similarities": []}
    
    # Create product feature vectors
    product_features = []
    
    for _, product in products.iterrows():
        features = {
            'id': product['id'],
            'name': product['name'],
            'category': product['category'],
            'price': product['price']
        }
        
        # Extract attributes if available
        if 'attributes' in product and product['attributes']:
            try:
                attributes = json.loads(product['attributes'].replace("'", "\""))
            except:
                try:
                    attributes = eval(product['attributes'])
                except:
                    attributes = {}
            
            for key, value in attributes.items():
                features[f'attr_{key}'] = value
        
        product_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(product_features)
    
    # Calculate feature-based similarities
    # Use basic features for similarity calculations
    basic_features = features_df[['category', 'price']].copy()
    
    # Convert category to one-hot encoding
    category_dummies = pd.get_dummies(basic_features['category'], prefix='category')
    basic_features = pd.concat([basic_features.drop('category', axis=1), category_dummies], axis=1)
    
    # Normalize price
    price_max = basic_features['price'].max()
    price_min = basic_features['price'].min()
    if price_max > price_min:
        basic_features['price'] = (basic_features['price'] - price_min) / (price_max - price_min)
    
    # Calculate pairwise similarity
    feature_similarity = cosine_similarity(basic_features)
    
    # Create interaction-based similarity
    interaction_similarity = np.zeros((len(products), len(products)))
    
    if not interactions.empty:
        # Create user-item matrix
        user_item = pd.pivot_table(
            interactions,
            values='id',
            index='customer_id',
            columns='product_id',
            aggfunc='count',
            fill_value=0
        )
        
        # Calculate item-item collaborative similarity if there are enough interactions
        if user_item.shape[0] > 1:  # If there are at least 2 users
            item_similarity = cosine_similarity(user_item.T)
            
            # Map product IDs to indices
            id_to_index = {id: i for i, id in enumerate(products['id'])}
            
            # Fill interaction_similarity with calculated values
            for i, item1 in enumerate(user_item.columns):
                for j, item2 in enumerate(user_item.columns):
                    if item1 in id_to_index and item2 in id_to_index:
                        idx1 = id_to_index[item1]
                        idx2 = id_to_index[item2]
                        interaction_similarity[idx1, idx2] = item_similarity[i, j]
    
    # Combine feature and interaction similarities (with weights)
    feature_weight = 0.7
    interaction_weight = 0.3
    
    combined_similarity = (feature_weight * feature_similarity + 
                          interaction_weight * interaction_similarity)
    
    # Prepare results
    similarity_results = []
    
    for i, product1 in features_df.iterrows():
        product_id = product1['id']
        
        # Get top 5 similar products (excluding self)
        similar_indices = np.argsort(combined_similarity[i])[::-1][1:6]
        
        for idx in similar_indices:
            product2 = features_df.iloc[idx]
            
            similarity_results.append({
                "product1_id": int(product_id),
                "product1_name": product1['name'],
                "product2_id": int(product2['id']),
                "product2_name": product2['name'],
                "similarity_score": float(combined_similarity[i, idx])
            })
    
    conn.close()
    
    return {"similarities": similarity_results}
