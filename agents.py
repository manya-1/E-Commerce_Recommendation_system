import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from database import get_connection, add_interaction, save_recommendation
from models import Customer, Product, Interaction
import json
import random

class Agent:
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id, agent_type):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = {}
    
    def update_state(self, new_state):
        """Update the agent's state"""
        self.state.update(new_state)
    
    def act(self, context):
        """Perform an action based on the current state and context"""
        raise NotImplementedError("Subclasses must implement this method")


class CustomerAgent(Agent):
    """Agent representing a customer"""
    
    def __init__(self, customer_id, customer_data=None):
        super().__init__(customer_id, "customer")
        self.customer_id = customer_id
        self.preferences = {}
        self.interaction_history = []
        self.segment = None
        
        if customer_data:
            if isinstance(customer_data, Customer):
                self.update_from_customer_object(customer_data)
            elif isinstance(customer_data, dict):
                self.update_from_dict(customer_data)
    
    def update_from_customer_object(self, customer):
        """Update agent state from Customer object"""
        self.update_state({
            "name": customer.name,
            "email": customer.email,
            "age": customer.age,
            "gender": customer.gender,
            "location": customer.location,
            "segment": customer.segment,
            "preferences": customer.preferences
        })
        self.preferences = customer.preferences
        self.segment = customer.segment
    
    def update_from_dict(self, data):
        """Update agent state from dictionary"""
        self.update_state(data)
        self.preferences = data.get("preferences", {})
        self.segment = data.get("segment")
    
    def load_interaction_history(self):
        """Load customer interaction history from database"""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM interactions 
        WHERE customer_id = ? 
        ORDER BY timestamp DESC
        """, (self.customer_id,))
        
        interactions = cursor.fetchall()
        conn.close()
        
        self.interaction_history = [Interaction.from_db_row(row) for row in interactions]
        return self.interaction_history
    
    def predict_interest(self, product):
        """Predict customer's interest in a product based on preferences and history"""
        # Initialize base interest score
        interest_score = 0.5  # Neutral starting point
        
        # Check if product category is in customer's favorite categories
        if self.preferences and 'favorite_categories' in self.preferences:
            if product.category in self.preferences['favorite_categories']:
                interest_score += 0.2
        
        # Adjust based on previous interactions with similar products
        if self.interaction_history:
            for interaction in self.interaction_history:
                if interaction.product_id == product.id:
                    # Direct interaction with this product
                    if interaction.interaction_type == "view":
                        interest_score += 0.1
                    elif interaction.interaction_type == "add_to_cart":
                        interest_score += 0.3
                    elif interaction.interaction_type == "purchase":
                        interest_score += 0.4
                    elif interaction.interaction_type == "review":
                        # Check rating if available
                        if interaction.details and 'rating' in interaction.details:
                            rating = interaction.details['rating']
                            interest_score += (rating / 5.0) * 0.3
                        else:
                            interest_score += 0.2
        
        # Cap the interest score between 0 and 1
        return max(0, min(1, interest_score))
    
    def act(self, context):
        """Perform customer agent action based on context"""
        if context.get("action") == "predict_interest":
            product = context.get("product")
            return self.predict_interest(product)
        
        elif context.get("action") == "get_preferences":
            return self.preferences
        
        elif context.get("action") == "record_interaction":
            product_id = context.get("product_id")
            interaction_type = context.get("interaction_type")
            details = context.get("details", {})
            
            # Record the interaction in the database
            interaction_id = add_interaction(
                self.customer_id, 
                product_id, 
                interaction_type,
                str(details) if details else None
            )
            
            # Update local interaction history
            self.load_interaction_history()
            
            return {"success": True, "interaction_id": interaction_id}
        
        return {"success": False, "error": "Invalid action"}


class ProductAgent(Agent):
    """Agent representing a product"""
    
    def __init__(self, product_id, product_data=None):
        super().__init__(product_id, "product")
        self.product_id = product_id
        self.category = None
        self.attributes = {}
        self.popularity = 0
        self.related_products = []
        
        if product_data:
            if isinstance(product_data, Product):
                self.update_from_product_object(product_data)
            elif isinstance(product_data, dict):
                self.update_from_dict(product_data)
    
    def update_from_product_object(self, product):
        """Update agent state from Product object"""
        self.update_state({
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "description": product.description,
            "attributes": product.attributes
        })
        self.category = product.category
        self.attributes = product.attributes
    
    def update_from_dict(self, data):
        """Update agent state from dictionary"""
        self.update_state(data)
        self.category = data.get("category")
        self.attributes = data.get("attributes", {})
    
    def calculate_popularity(self):
        """Calculate product popularity based on interaction data"""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Count different types of interactions with weights
        cursor.execute("""
        SELECT interaction_type, COUNT(*) as count
        FROM interactions
        WHERE product_id = ?
        GROUP BY interaction_type
        """, (self.product_id,))
        
        interaction_counts = cursor.fetchall()
        conn.close()
        
        # Apply weights to different interaction types
        weights = {
            "view": 1,
            "add_to_cart": 3,
            "purchase": 5,
            "review": 2
        }
        
        popularity = 0
        for interaction_type, count in interaction_counts:
            popularity += count * weights.get(interaction_type, 1)
        
        self.popularity = popularity
        return popularity
    
    def find_related_products(self, limit=5):
        """Find products related to this one based on category and attributes"""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find products in the same category
        cursor.execute("""
        SELECT id, name, category, price, description, attributes 
        FROM products
        WHERE category = ? AND id != ?
        LIMIT ?
        """, (self.category, self.product_id, limit))
        
        related = cursor.fetchall()
        conn.close()
        
        self.related_products = [Product.from_db_row([p[0], p[1], p[2], p[3], p[4], p[5], None]) for p in related]
        return self.related_products
    
    def get_feature_vector(self):
        """Create a feature vector representing the product for similarity calculations"""
        # Enhanced feature vector with more product attributes
        features = {
            "price": self.state.get("price", 0),
        }
        
        # Add category as one-hot encoding
        categories = ["Electronics", "Clothing", "Fashion", "Home", "Home Decor", "Books", "Sports", "Beauty", "Food", "Fitness"]
        for category in categories:
            features[f"category_{category}"] = 1 if self.category == category else 0
        
        # Add some common attributes if available
        attrs = self.attributes or {}
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs.replace("'", "\""))
            except:
                try:
                    attrs = eval(attrs)
                except:
                    attrs = {}
        
        # Add basic attributes to feature vector
        for key, value in attrs.items():
            if isinstance(value, (int, float)):
                features[f"attr_{key}"] = value
            elif isinstance(value, str):
                features[f"attr_{key}_{value}"] = 1
        
        # Add specific product attributes that are especially valuable for recommendations
        # Rating data
        features["rating"] = attrs.get("product_rating", 3.0)
        features["avg_similar_rating"] = attrs.get("avg_similar_rating", 3.0)
        features["sentiment_score"] = attrs.get("sentiment_score", 0.5)
        
        # Seasonal and contextual attributes
        if "seasonal_relevance" in attrs:
            seasons = ["Spring", "Summer", "Autumn", "Winter"]
            for season in seasons:
                features[f"season_{season}"] = 1 if attrs["seasonal_relevance"] == season else 0
                
        features["holiday_relevant"] = 1 if attrs.get("holiday_relevance") == "Yes" else 0
        
        # Geographical relevance
        if "geographical_relevance" in attrs:
            locations = ["USA", "UK", "Canada", "India", "Germany"]
            for location in locations:
                features[f"geo_{location}"] = 1 if attrs["geographical_relevance"] == location else 0
        
        # Recommendation probability (if available from dataset)
        features["recommendation_probability"] = attrs.get("recommendation_probability", 0.5)
        
        return features
    
    def act(self, context):
        """Perform product agent action based on context"""
        if context.get("action") == "get_popularity":
            return self.calculate_popularity()
        
        elif context.get("action") == "get_related_products":
            limit = context.get("limit", 5)
            return self.find_related_products(limit)
        
        elif context.get("action") == "get_features":
            return self.get_feature_vector()
        
        return {"success": False, "error": "Invalid action"}


class RecommendationAgent(Agent):
    """Agent responsible for generating product recommendations"""
    
    def __init__(self, agent_id="recommendation_agent"):
        super().__init__(agent_id, "recommendation")
        self.algorithms = {
            "popularity": self._popularity_based,
            "collaborative": self._collaborative_filtering,
            "content": self._content_based,
            "hybrid": self._hybrid_approach
        }
    
    def _popularity_based(self, customer_id, limit=5):
        """Generate recommendations based on product popularity"""
        conn = get_connection()
        
        # Get most popular products based on interaction counts
        popular_products = pd.read_sql("""
        SELECT p.id, p.name, p.category, p.price, COUNT(i.id) as interaction_count
        FROM products p
        JOIN interactions i ON p.id = i.product_id
        GROUP BY p.id
        ORDER BY interaction_count DESC
        LIMIT ?
        """, conn, params=(limit,))
        
        conn.close()
        
        # Create recommendation entries
        recommendations = []
        for _, row in popular_products.iterrows():
            score = min(1.0, row['interaction_count'] / 100)  # Normalize score
            
            recommendation = {
                "product_id": row['id'],
                "product_name": row['name'],
                "score": score,
                "algorithm": "popularity"
            }
            recommendations.append(recommendation)
            
            # Save to database
            save_recommendation(
                customer_id=customer_id,
                product_id=row['id'],
                score=score,
                algorithm="popularity"
            )
        
        return recommendations
    
    def _collaborative_filtering(self, customer_id, limit=5):
        """Generate recommendations using collaborative filtering"""
        conn = get_connection()
        
        # Get customer's interaction history
        customer_interactions = pd.read_sql("""
        SELECT product_id, interaction_type 
        FROM interactions 
        WHERE customer_id = ?
        """, conn, params=(customer_id,))
        
        # If customer has no interactions, fall back to popularity-based
        if customer_interactions.empty:
            conn.close()
            return self._popularity_based(customer_id, limit)
        
        # Get all interactions to find similar customers
        all_interactions = pd.read_sql("""
        SELECT customer_id, product_id, interaction_type
        FROM interactions
        """, conn)
        
        # Create a utility matrix (customer-product interactions)
        # Convert interaction types to numerical values
        interaction_values = {
            "view": 1,
            "add_to_cart": 3,
            "purchase": 5,
            "review": 4
        }
        
        all_interactions['value'] = all_interactions['interaction_type'].map(interaction_values)
        
        # Create utility matrix
        utility_matrix = all_interactions.pivot_table(
            index='customer_id',
            columns='product_id',
            values='value',
            aggfunc='mean',
            fill_value=0
        )
        
        # Find similar customers
        # Convert utility matrix to numpy array
        matrix = utility_matrix.values
        customer_idx = utility_matrix.index.get_loc(customer_id) if customer_id in utility_matrix.index else -1
        
        if customer_idx == -1:
            conn.close()
            return self._popularity_based(customer_id, limit)
        
        # Calculate similarity between the target customer and all other customers
        customer_vector = matrix[customer_idx:customer_idx+1]
        similarities = cosine_similarity(customer_vector, matrix)[0]
        
        # Get indices of most similar customers (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:6]  # top 5 similar customers
        
        # Get products that similar customers interacted with
        similar_customer_ids = [utility_matrix.index[idx] for idx in similar_indices]
        
        similar_interactions = all_interactions[
            (all_interactions['customer_id'].isin(similar_customer_ids)) &
            (~all_interactions['product_id'].isin(customer_interactions['product_id']))
        ]
        
        # Group by product and calculate recommendation score
        product_scores = similar_interactions.groupby('product_id')['value'].agg(['mean', 'count'])
        product_scores['score'] = product_scores['mean'] * (product_scores['count'] / product_scores['count'].max())
        
        # Get top products
        top_products = product_scores.nlargest(limit, 'score')
        
        # Get product details
        products = pd.read_sql("""
        SELECT id, name, category, price
        FROM products
        WHERE id IN ({})
        """.format(','.join(['?'] * len(top_products.index))), 
        conn, params=top_products.index.tolist())
        
        conn.close()
        
        # Create recommendation entries
        recommendations = []
        for product_id, row in top_products.iterrows():
            product_info = products[products['id'] == product_id]
            if not product_info.empty:
                score = min(1.0, row['score'] / 5)  # Normalize score
                
                recommendation = {
                    "product_id": product_id,
                    "product_name": product_info.iloc[0]['name'],
                    "score": score,
                    "algorithm": "collaborative"
                }
                recommendations.append(recommendation)
                
                # Save to database
                save_recommendation(
                    customer_id=customer_id,
                    product_id=product_id,
                    score=score,
                    algorithm="collaborative"
                )
        
        return recommendations
    
    def _content_based(self, customer_id, limit=5):
        """Generate recommendations using content-based filtering"""
        conn = get_connection()
        
        # Get customer preferences
        customer_data = pd.read_sql("""
        SELECT * FROM customers WHERE id = ?
        """, conn, params=(customer_id,))
        
        if customer_data.empty:
            conn.close()
            return self._popularity_based(customer_id, limit)
        
        # Create customer agent
        customer = Customer.from_dict(customer_data.iloc[0].to_dict())
        customer_agent = CustomerAgent(customer_id, customer)
        
        # Get customer's interaction history
        customer_interactions = pd.read_sql("""
        SELECT product_id, interaction_type 
        FROM interactions 
        WHERE customer_id = ?
        """, conn, params=(customer_id,))
        
        # Get all products
        all_products = pd.read_sql("SELECT * FROM products", conn)
        
        # If customer has no interactions, get recommendations based on preferences
        if customer_interactions.empty:
            # Check if customer has category preferences
            preferences = customer_agent.preferences
            preferred_categories = preferences.get('favorite_categories', []) if preferences else []
            
            if preferred_categories:
                # Filter products by preferred categories
                filtered_products = all_products[all_products['category'].isin(preferred_categories)]
                
                if len(filtered_products) > 0:
                    # Randomly select products from preferred categories
                    if len(filtered_products) > limit:
                        recommended_products = filtered_products.sample(limit)
                    else:
                        recommended_products = filtered_products
                else:
                    # Fall back to popularity-based if no products in preferred categories
                    conn.close()
                    return self._popularity_based(customer_id, limit)
            else:
                # Fall back to popularity-based if no preferences
                conn.close()
                return self._popularity_based(customer_id, limit)
        else:
            # Get products that customer has interacted with
            interacted_products = pd.read_sql("""
            SELECT p.* 
            FROM products p
            JOIN interactions i ON p.id = i.product_id
            WHERE i.customer_id = ?
            """, conn, params=(customer_id,))
            
            # Find products similar to what customer has interacted with
            # Collect product features
            product_features = []
            for _, product_row in interacted_products.iterrows():
                product = Product.from_dict(product_row.to_dict())
                product_agent = ProductAgent(product.id, product)
                features = product_agent.get_feature_vector()
                product_features.append(features)
            
            # For each product, calculate similarity to interacted products
            # This is a simplified approach - in a real system, you would use a more sophisticated similarity calculation
            recommended_product_ids = []
            for _, candidate_row in all_products.iterrows():
                # Skip products customer already interacted with
                if candidate_row['id'] in interacted_products['id'].values:
                    continue
                
                candidate = Product.from_dict(candidate_row.to_dict())
                candidate_agent = ProductAgent(candidate.id, candidate)
                candidate_features = candidate_agent.get_feature_vector()
                
                # Simple similarity: +1 for each matching category
                similarity = 0
                for product_feature in product_features:
                    if candidate_features.get('category_' + candidate.category, 0) == 1:
                        for k in product_feature.keys():
                            if k.startswith('category_') and product_feature[k] == 1 and candidate_features.get(k, 0) == 1:
                                similarity += 1
                                break
                
                if similarity > 0:
                    recommended_product_ids.append((candidate.id, similarity))
            
            # Sort by similarity and take the top limit
            recommended_product_ids.sort(key=lambda x: x[1], reverse=True)
            recommended_product_ids = recommended_product_ids[:limit]
            
            if recommended_product_ids:
                # Get full product details
                recommended_products = all_products[all_products['id'].isin([p_id for p_id, _ in recommended_product_ids])]
            else:
                # Fall back to popularity-based if no similar products found
                conn.close()
                return self._popularity_based(customer_id, limit)
        
        # Create recommendation entries
        recommendations = []
        for _, product_row in recommended_products.iterrows():
            # Calculate score based on product features matching customer preferences
            customer_agent = CustomerAgent(customer_id, customer)
            product = Product.from_dict(product_row.to_dict())
            product_agent = ProductAgent(product.id, product)
            
            # Predict customer's interest in this product
            interest_score = customer_agent.predict_interest(product)
            
            recommendation = {
                "product_id": product.id,
                "product_name": product.name,
                "score": interest_score,
                "algorithm": "content"
            }
            recommendations.append(recommendation)
            
            # Save to database
            save_recommendation(
                customer_id=customer_id,
                product_id=product.id,
                score=interest_score,
                algorithm="content"
            )
        
        conn.close()
        return recommendations
    
    def _hybrid_approach(self, customer_id, limit=5):
        """Generate recommendations using a hybrid approach combining multiple methods"""
        # Get customer data for contextual factors
        conn = get_connection()
        customer_data = pd.read_sql("SELECT * FROM customers WHERE id = ?", conn, params=(customer_id,))
        conn.close()
        
        if not customer_data.empty:
            customer = Customer.from_dict(customer_data.iloc[0].to_dict())
            customer_agent = CustomerAgent(customer_id, customer)
            preferences = customer_agent.preferences
        else:
            preferences = {}
            
        # Get recommendations from each algorithm
        collab_recs = self._collaborative_filtering(customer_id, limit=4)
        content_recs = self._content_based(customer_id, limit=4)
        popular_recs = self._popularity_based(customer_id, limit=3)
        
        # Combine recommendations and adjust scores
        all_recs = {}
        
        # Process collaborative filtering recommendations
        for rec in collab_recs:
            product_id = rec["product_id"]
            all_recs[product_id] = {
                "product_id": product_id,
                "product_name": rec["product_name"],
                "score": rec["score"] * 0.35,  # Weight: 35%
                "algorithms": ["collaborative"]
            }
        
        # Process content-based recommendations
        for rec in content_recs:
            product_id = rec["product_id"]
            if product_id in all_recs:
                all_recs[product_id]["score"] += rec["score"] * 0.35  # Weight: 35%
                all_recs[product_id]["algorithms"].append("content")
            else:
                all_recs[product_id] = {
                    "product_id": product_id,
                    "product_name": rec["product_name"],
                    "score": rec["score"] * 0.35,
                    "algorithms": ["content"]
                }
        
        # Process popularity-based recommendations
        for rec in popular_recs:
            product_id = rec["product_id"]
            if product_id in all_recs:
                all_recs[product_id]["score"] += rec["score"] * 0.15  # Weight: 15%
                all_recs[product_id]["algorithms"].append("popularity")
            else:
                all_recs[product_id] = {
                    "product_id": product_id,
                    "product_name": rec["product_name"],
                    "score": rec["score"] * 0.15,
                    "algorithms": ["popularity"]
                }
        
        # Apply contextual boosting based on product attributes and customer preferences
        conn = get_connection()
        for product_id in all_recs:
            product_data = pd.read_sql("SELECT * FROM products WHERE id = ?", conn, params=(product_id,))
            
            if not product_data.empty:
                product = Product.from_dict(product_data.iloc[0].to_dict())
                product_agent = ProductAgent(product_id, product)
                
                # Get product attributes
                attrs = product_agent.attributes
                if isinstance(attrs, str):
                    try:
                        attrs = eval(attrs)
                    except:
                        attrs = {}
                
                # Apply seasonal boosting
                customer_season_pref = preferences.get('season_preference', None)
                product_season = attrs.get('seasonal_relevance', None)
                if customer_season_pref and product_season and customer_season_pref == product_season:
                    all_recs[product_id]["score"] += 0.05
                    
                # Apply holiday boosting
                customer_holiday_pref = preferences.get('holiday_preference', 'No')
                product_holiday = attrs.get('holiday_relevance', 'No')
                if customer_holiday_pref == 'Yes' and product_holiday == 'Yes':
                    all_recs[product_id]["score"] += 0.05
                
                # Apply geographical relevance boosting
                customer_location = customer.location if customer else None
                product_geo = attrs.get('geographical_relevance', None)
                
                # Map Indian cities to 'India' for matching
                if customer_location in ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Kolkata'] and product_geo == 'India':
                    all_recs[product_id]["score"] += 0.05
                
                # Apply sentiment score boosting
                sentiment_score = attrs.get('sentiment_score', 0.5)
                if sentiment_score > 0.7:  # High positive sentiment
                    all_recs[product_id]["score"] += 0.05
                
                # Apply recommendation probability boosting
                rec_probability = attrs.get('recommendation_probability', 0.5)
                all_recs[product_id]["score"] += rec_probability * 0.1  # Use recommendation probability from dataset
        
        conn.close()
        
        # Sort recommendations by score and take top 'limit'
        sorted_recs = sorted(all_recs.values(), key=lambda x: x["score"], reverse=True)[:limit]
        
        # Prepare final recommendations
        recommendations = []
        for rec in sorted_recs:
            recommendation = {
                "product_id": rec["product_id"],
                "product_name": rec["product_name"],
                "score": min(1.0, rec["score"]),  # Cap at 1.0
                "algorithm": "hybrid"
            }
            recommendations.append(recommendation)
            
            # Save to database
            save_recommendation(
                customer_id=customer_id,
                product_id=rec["product_id"],
                score=min(1.0, rec["score"]), 
                algorithm="hybrid"
            )
        
        return recommendations
    
    def generate_recommendations(self, customer_id, algorithm="hybrid", limit=5):
        """Generate product recommendations for a customer"""
        if algorithm in self.algorithms:
            return self.algorithms[algorithm](customer_id, limit)
        else:
            # Default to hybrid approach if algorithm not found
            return self._hybrid_approach(customer_id, limit)
    
    def act(self, context):
        """Perform recommendation agent action based on context"""
        if context.get("action") == "recommend":
            customer_id = context.get("customer_id")
            algorithm = context.get("algorithm", "hybrid")
            limit = context.get("limit", 5)
            
            return self.generate_recommendations(customer_id, algorithm, limit)
        
        return {"success": False, "error": "Invalid action"}


class AgentSystem:
    """System managing all agents and their interactions"""
    
    def __init__(self):
        self.customer_agents = {}
        self.product_agents = {}
        self.recommendation_agent = RecommendationAgent()
    
    def get_customer_agent(self, customer_id):
        """Get or create a customer agent for a given customer ID"""
        if customer_id not in self.customer_agents:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM customers WHERE id = ?", (customer_id,))
            customer_data = cursor.fetchone()
            
            conn.close()
            
            if customer_data:
                customer = Customer.from_db_row(customer_data)
                self.customer_agents[customer_id] = CustomerAgent(customer_id, customer)
            else:
                # Create a new agent with default values if customer not found
                self.customer_agents[customer_id] = CustomerAgent(customer_id)
        
        return self.customer_agents[customer_id]
    
    def get_product_agent(self, product_id):
        """Get or create a product agent for a given product ID"""
        if product_id not in self.product_agents:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
            product_data = cursor.fetchone()
            
            conn.close()
            
            if product_data:
                product = Product.from_db_row(product_data)
                self.product_agents[product_id] = ProductAgent(product_id, product)
            else:
                # Create a new agent with default values if product not found
                self.product_agents[product_id] = ProductAgent(product_id)
        
        return self.product_agents[product_id]
    
    def record_interaction(self, customer_id, product_id, interaction_type, details=None):
        """Record a customer-product interaction"""
        customer_agent = self.get_customer_agent(customer_id)
        
        context = {
            "action": "record_interaction",
            "product_id": product_id,
            "interaction_type": interaction_type,
            "details": details
        }
        
        return customer_agent.act(context)
    
    def get_recommendations(self, customer_id, algorithm="hybrid", limit=5):
        """Get product recommendations for a customer"""
        context = {
            "action": "recommend",
            "customer_id": customer_id,
            "algorithm": algorithm,
            "limit": limit
        }
        
        return self.recommendation_agent.act(context)
    
    def get_product_popularity(self, product_id):
        """Get the popularity of a product"""
        product_agent = self.get_product_agent(product_id)
        
        context = {
            "action": "get_popularity"
        }
        
        return product_agent.act(context)
    
    def get_related_products(self, product_id, limit=5):
        """Get products related to a given product"""
        product_agent = self.get_product_agent(product_id)
        
        context = {
            "action": "get_related_products",
            "limit": limit
        }
        
        return product_agent.act(context)
    
    def predict_customer_interest(self, customer_id, product_id):
        """Predict a customer's interest in a product"""
        customer_agent = self.get_customer_agent(customer_id)
        product_agent = self.get_product_agent(product_id)
        
        # Get product data
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        product_data = cursor.fetchone()
        
        conn.close()
        
        if product_data:
            product = Product.from_db_row(product_data)
            
            context = {
                "action": "predict_interest",
                "product": product
            }
            
            return customer_agent.act(context)
        
        return 0.0  # Default interest score if product not found

# Global agent system instance for easy access
agent_system = AgentSystem()
