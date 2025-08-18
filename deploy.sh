#!/bin/bash

# Deployment script for PDF Parser API
# Can be deployed to Railway, Render, or any Python host

echo "🚀 Deploying PDF Parser API..."

# Option 1: Railway (Recommended)
# railway up

# Option 2: Render
# Create render.yaml first, then:
# render deploy

# Option 3: Heroku
# heroku create pdf-parser-api
# git push heroku main

# Option 4: Local Docker
docker build -t pdf-parser .
docker run -p 5001:5001 pdf-parser

echo "✅ Deployment complete!"