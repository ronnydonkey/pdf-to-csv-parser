#!/bin/bash

echo "🚀 Setting up GitHub repository for pdf-to-csv-agent"
echo ""
echo "This script will help you create a GitHub repo and connect it to Railway."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install it with: brew install gh"
    echo "Then run: gh auth login"
    exit 1
fi

echo "📝 Creating GitHub repository..."
echo ""

# Create the repository
gh repo create pdf-to-csv-parser \
    --private \
    --description "High-accuracy PDF and CSV parser API for bank statements with subscription detection" \
    --source=. \
    --remote=origin \
    --push

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Repository created successfully!"
    echo ""
    echo "🔗 Your repository URL:"
    gh repo view --web
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to your Railway dashboard"
    echo "2. Go to your subscription-tracker-parser service"
    echo "3. Go to Settings → Source"
    echo "4. Connect to GitHub and select the 'pdf-to-csv-parser' repository"
    echo "5. Railway will automatically deploy future changes!"
    echo ""
    echo "🎉 CSV support will be available after Railway redeploys!"
else
    echo ""
    echo "❌ Failed to create repository."
    echo "You can create it manually:"
    echo ""
    echo "1. Go to https://github.com/new"
    echo "2. Name it: pdf-to-csv-parser"
    echo "3. Make it private"
    echo "4. Don't initialize with README (we already have one)"
    echo "5. Create the repository"
    echo "6. Run these commands:"
    echo ""
    echo "git remote add origin https://github.com/YOUR_USERNAME/pdf-to-csv-parser.git"
    echo "git push -u origin main"
fi