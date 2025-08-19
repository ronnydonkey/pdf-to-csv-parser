# Manual GitHub Setup Instructions

Since we need to set up GitHub authentication, here are the steps to manually create the repository and connect it to Railway:

## Step 1: Create GitHub Repository

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `pdf-to-csv-parser`
3. Description: `High-accuracy PDF and CSV parser API for bank statements with subscription detection`
4. Make it **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push Code to GitHub

After creating the repository, GitHub will show you commands. Run these in your terminal:

```bash
cd /Users/aarongreenberg/Documents/amg_toolkit/pdf-to-csv-agent

# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pdf-to-csv-parser.git

# Push the code
git branch -M main
git push -u origin main
```

## Step 3: Connect Railway to GitHub

1. Go to your [Railway Dashboard](https://railway.app/dashboard)
2. Find your `subscription-tracker-parser` service
3. Click on the service to open it
4. Go to **Settings** tab
5. Scroll to **Source** section
6. Click **Connect GitHub repo**
7. Authorize Railway if needed
8. Select your `pdf-to-csv-parser` repository
9. Keep branch as `main`
10. Click **Connect**

## Step 4: Railway Will Auto-Deploy

Once connected, Railway will:
- Automatically pull the latest code
- Build and deploy the application
- Enable CSV support!

## Step 5: Verify Deployment

After deployment (takes 2-3 minutes):
1. Check the Railway logs for successful deployment
2. Test CSV upload in your Subscription Tracker
3. All features should now work!

## Future Updates

Now that Railway is connected to GitHub:
- Any push to the `main` branch will trigger automatic deployment
- No more manual deployments needed!
- Changes will be live within minutes

---

**Note**: If you prefer using GitHub CLI, run:
```bash
gh auth login
./setup-github.sh
```