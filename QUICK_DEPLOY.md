# Quick Deploy Guide - Railway (5 Minutes)

The fastest way to deploy your FastAPI backend and share it with developers.

## Step-by-Step Instructions

### 1. Create Railway Account (1 minute)

1. Go to **https://railway.app**
2. Click **"Start a New Project"**
3. Sign up with **GitHub** (recommended) or email

### 2. Deploy Your App (2 minutes)

**Option A: Deploy from GitHub (Recommended)**

1. **Push your code to GitHub first:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

2. **In Railway:**
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Choose your repository
   - Railway will auto-detect and deploy!

**Option B: Deploy Manually**

1. In Railway, click **"New Project"** → **"Empty Project"**
2. Click **"New"** → **"GitHub Repo"** or **"Empty Project"**
3. Upload these files:
   - `main.py`
   - `requirements.txt`
   - `Dockerfile`
4. Railway will automatically build and deploy

### 3. Get Your URL (Instant)

Once deployed, Railway gives you a URL like:
```
https://your-app-name.railway.app
```

**That's it!** Your API is live and ready to share! 🎉

### 4. Test Your Deployed API

**Status endpoint:**
```bash
curl https://your-app-name.railway.app/status
```

**API Documentation:**
- Swagger UI: `https://your-app-name.railway.app/docs`
- ReDoc: `https://your-app-name.railway.app/redoc`

**Register endpoint:**
```bash
curl -X POST https://your-app-name.railway.app/register \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com", "age": 25}'
```

### 5. Share with Developers

Send them:
1. **API Base URL:** `https://your-app-name.railway.app`
2. **API Docs:** `https://your-app-name.railway.app/docs`
3. **Quick Start:** Share `FASTAPI_README.md`

---

## What You Need

Make sure these files are in your project:
- ✅ `main.py`
- ✅ `requirements.txt`
- ✅ `Dockerfile`

---

## Railway Free Tier

- **$5 free credit per month**
- Perfect for testing and sharing
- Auto-deploys on git push
- No credit card required for free tier

---

## Troubleshooting

**Build fails?**
- Check that `requirements.txt` has all dependencies
- Make sure `Dockerfile` is in the root directory

**API not working?**
- Check Railway logs (click on your service → "View Logs")
- Verify the URL is correct
- Make sure the service is running (not stopped)

**Need help?**
- Railway has great docs: https://docs.railway.app
- Check `DEPLOYMENT.md` for more details

---

## That's It!

Your API is now live and shareable. Developers can access it from anywhere in the world! 🌍

