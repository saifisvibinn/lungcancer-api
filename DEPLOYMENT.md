# Deployment Guide

This guide will help you deploy your FastAPI backend to a free hosting service so you can share it with others.

## Quick Deploy Options

### Option 1: Railway (Recommended - Easiest) ⭐

**Railway** is the easiest option with a free tier.

#### Steps:

1. **Create a Railway account:**
   - Go to https://railway.app
   - Sign up with GitHub (recommended) or email

2. **Create a new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo" (if you have GitHub) OR
   - Select "Empty Project" and upload files manually

3. **If using GitHub:**
   - Push your code to GitHub first
   - Connect your repository
   - Railway will auto-detect the Dockerfile

4. **If uploading manually:**
   - Click "New" → "GitHub Repo" or "Empty Project"
   - Upload these files:
     - `main.py`
     - `requirements.txt`
     - `Dockerfile`
     - `railway.json`
   - Railway will automatically build and deploy

5. **Configure the service:**
   - Railway will auto-detect Python
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Or it will use the Dockerfile automatically

6. **Get your URL:**
   - Once deployed, Railway gives you a URL like: `your-app.railway.app`
   - Share this URL with developers!

**Railway Free Tier:**
- $5 free credit per month
- Perfect for testing and sharing
- Auto-deploys on git push

---

### Option 2: Render (Also Easy)

**Render** offers free hosting with easy setup.

#### Steps:

1. **Create a Render account:**
   - Go to https://render.com
   - Sign up with GitHub (recommended)

2. **Create a new Web Service:**
   - Click "New" → "Web Service"
   - Connect your GitHub repository OR upload files

3. **Configure the service:**
   - **Name:** `fastapi-backend` (or any name)
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free

4. **Deploy:**
   - Click "Create Web Service"
   - Render will build and deploy automatically
   - You'll get a URL like: `your-app.onrender.com`

**Render Free Tier:**
- Free tier available
- Spins down after 15 minutes of inactivity (wakes up on first request)
- Perfect for testing and sharing

---

### Option 3: Fly.io (Good Performance)

**Fly.io** offers global deployment with good performance.

#### Steps:

1. **Install Fly CLI:**
   ```bash
   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   
   # Or download from https://fly.io/docs/getting-started/installing-flyctl/
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **Initialize:**
   ```bash
   fly launch
   ```
   - Follow the prompts
   - Don't deploy yet (we'll configure first)

4. **Create `fly.toml`** (I'll create this for you)

5. **Deploy:**
   ```bash
   fly deploy
   ```

---

## Files Needed for Deployment

Make sure you have these files in your project:

✅ `main.py` - Your FastAPI application  
✅ `requirements.txt` - Python dependencies  
✅ `Dockerfile` - For containerized deployment  
✅ `.dockerignore` - Excludes unnecessary files  
✅ `railway.json` - Railway configuration (optional)  
✅ `render.yaml` - Render configuration (optional)  

---

## After Deployment

### Test Your Deployed API

Once deployed, test your endpoints:

**Status endpoint:**
```bash
curl https://your-app.railway.app/status
```

**Register endpoint:**
```bash
curl -X POST https://your-app.railway.app/register \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com", "age": 25}'
```

**View API docs:**
- Swagger UI: `https://your-app.railway.app/docs`
- ReDoc: `https://your-app.railway.app/redoc`

### Share with Developers

Send them:
1. **API Base URL:** `https://your-app.railway.app`
2. **API Documentation:** `https://your-app.railway.app/docs`
3. **Quick Start Guide:** Share `FASTAPI_README.md`

---

## Environment Variables (Optional)

If you need to configure settings, add environment variables in your hosting platform:

- `PORT` - Automatically set by hosting platform
- `ENVIRONMENT` - Set to "production" for production mode

---

## Troubleshooting

### Common Issues:

1. **Port binding error:**
   - Make sure you're using `--host 0.0.0.0` and `--port $PORT`
   - The `$PORT` variable is set automatically by hosting platforms

2. **Build fails:**
   - Check that `requirements.txt` has all dependencies
   - Make sure Python version is compatible (3.8+)

3. **API not accessible:**
   - Check that the service is running (not sleeping)
   - Verify the URL is correct
   - Check logs in your hosting platform dashboard

4. **CORS errors:**
   - If calling from a browser, you may need to add CORS middleware
   - Add this to `main.py` if needed:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # In production, specify actual origins
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

---

## Recommended: Railway

For sharing with developers, **Railway** is recommended because:
- ✅ Easiest setup
- ✅ Free tier with $5 credit/month
- ✅ Auto-deploys from GitHub
- ✅ Good performance
- ✅ Easy to share URLs

---

## Need Help?

If you run into issues:
1. Check the logs in your hosting platform dashboard
2. Verify all files are uploaded correctly
3. Make sure `requirements.txt` includes all dependencies
4. Test locally first: `uvicorn main:app --reload`

