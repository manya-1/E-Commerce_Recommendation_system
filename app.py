import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from database import initialize_database, get_connection, add_sample_data_if_empty
from models import Product, Customer, Interaction

# Set page configuration
st.set_page_config(
    page_title="AI-Powered E-commerce System",
    page_icon="🛒",
    layout="wide"
)

# Initialize database
initialize_database()
add_sample_data_if_empty()

# Main app title and description
st.title("🛒 AI-Powered E-commerce Recommendation System")

st.markdown("""
This system utilizes multi-agent AI to provide personalized product recommendations 
for an e-commerce platform. The system analyzes customer browsing behavior, purchase history, 
and preferences to deliver tailored product suggestions.
""")

# Overview statistics
conn = get_connection()
customers_df = pd.read_sql("SELECT * FROM customers", conn)
products_df = pd.read_sql("SELECT * FROM products", conn)
interactions_df = pd.read_sql("SELECT * FROM interactions", conn)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", len(customers_df))
with col2:
    st.metric("Total Products", len(products_df))
with col3:
    st.metric("Total Interactions", len(interactions_df))

# Recent interactions
st.subheader("Recent Customer Interactions")
recent_interactions = pd.read_sql("""
    SELECT i.id, c.name as customer, p.name as product, i.interaction_type, i.timestamp 
    FROM interactions i
    JOIN customers c ON i.customer_id = c.id
    JOIN products p ON i.product_id = p.id
    ORDER BY i.timestamp DESC
    LIMIT 10
""", conn)

if not recent_interactions.empty:
    st.dataframe(recent_interactions, use_container_width=True)
else:
    st.info("No customer interactions recorded yet.")

# Top products
st.subheader("Top Products by Interaction")
top_products = pd.read_sql("""
    SELECT p.name as product, COUNT(i.id) as interaction_count
    FROM products p
    JOIN interactions i ON p.id = i.product_id
    GROUP BY p.id
    ORDER BY interaction_count DESC
    LIMIT 5
""", conn)

if not top_products.empty:
    fig = px.bar(top_products, x='product', y='interaction_count', 
                title="Top Products by Customer Interaction",
                labels={'product': 'Product', 'interaction_count': 'Interaction Count'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No product interaction data available yet.")

# Customer segmentation preview
st.subheader("Customer Segmentation Preview")
segments = pd.read_sql("""
    SELECT segment, COUNT(*) as count
    FROM customers
    GROUP BY segment
""", conn)

if not segments.empty:
    fig = px.pie(segments, values='count', names='segment', 
                title='Customer Segments',
                hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No customer segmentation data available yet.")

# System information
st.sidebar.title("System Information")
st.sidebar.info("""
### Multi-agent AI System
This e-commerce platform utilizes a multi-agent AI framework where:
- **Customer Agents**: Represent user preferences and behaviors
- **Product Agents**: Represent product attributes and relationships
- **Recommendation Agents**: Analyze and match customers with products

The system continuously learns from customer interactions to improve recommendations.
""")

# Quick links
st.sidebar.title("Quick Navigation")
st.sidebar.markdown("""
- [Product Catalog](/Product_Catalog)
- [Customer Profiles](/Customer_Profiles)
- [Recommendations](/Recommendations)
- [Analytics Dashboard](/Analytics)
- [Data Import](/Data_Import)
""")

# Information about the imported datasets
st.subheader("Available Datasets for Import")
col1, col2 = st.columns(2)

with col1:
    st.write("**Customer Dataset**")
    try:
        customer_df = pd.read_csv("attached_assets/customer_data_collection.csv")
        st.write(f"Records available: {len(customer_df):,}")
        st.write("Contains customer demographics, browsing history, purchase history, and segmentation information.")
        
        # Show preview of demographics
        age_stats = customer_df["Age"].describe().to_dict()
        st.write(f"Age range: {age_stats['min']:.0f} to {age_stats['max']:.0f} years (avg: {age_stats['mean']:.1f})")
        
        gender_counts = customer_df["Gender"].value_counts()
        st.write(f"Gender distribution: {gender_counts.get('Male', 0):,} male, {gender_counts.get('Female', 0):,} female, {gender_counts.get('Other', 0):,} other")
    except Exception as e:
        st.info("Customer dataset preview not available.")

with col2:
    st.write("**Product Dataset**")
    try:
        product_df = pd.read_csv("attached_assets/product_recommendation_data.csv")
        st.write(f"Records available: {len(product_df):,}")
        st.write("Contains product information, ratings, prices, and recommendation probabilities.")
        
        # Show preview of categories
        categories = product_df["Category"].value_counts().head(5)
        st.write("Top categories:")
        for category, count in categories.items():
            st.write(f"- {category}: {count:,} products")
            
        # Show price range
        price_stats = product_df["Price"].describe().to_dict()
        st.write(f"Price range: ${price_stats['min']:.2f} to ${price_stats['max']:.2f} (avg: ${price_stats['mean']:.2f})")
    except Exception as e:
        st.info("Product dataset preview not available.")

st.info("Visit the [Data Import](/Data_Import) page to import these datasets and enhance the recommendation system.")

# Connection cleanup
conn.close()
