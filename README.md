# LeadTap-AI
This repository consists the delivarables for the Vector Based Property Search System

# Docker Setup 
Step 1: Install Docker Desktop

Step 2: Pull Required Docker Images

# Pull Qdrant vector database
docker pull qdrant/qdrant:latest

# Access the qdrant UI 
[Qdrant localhost](http://localhost:6333/dashboard)

# Pull Ollama for embeddings
ollama pull nomic-embed-text



1. Semantic Embeddings
Converts text into mathematical vectors that capture meaning
Uses nomic-embed-text model to understand context and relationships
Enables searching based on conceptual similarity, not just keyword matching

2. Natural Language Query Parsing
Regex-based extraction of structured information from free text
Extracts: location, price constraints, bedroom requirements, amenities, preferences
Falls back to pure semantic search when no structured patterns detected

3. Hybrid Search Approach
Vector Search: Retrieves top 50 semantically similar properties
Strict Filters: Applies price and bedroom constraints
Smart Boosting: Enhances relevance with location, amenities, and school proximity

4. Advanced Ranking Algorithm
Base Similarity Score: 0-1 scale measuring semantic match
Location Boost: +0.15 for matching preferred areas
Amenity Boost: +0.05 for each matching feature
School Proximity: +0.1 for schools within 300 meters
Re-ranking: Combines all factors for final result ordering



