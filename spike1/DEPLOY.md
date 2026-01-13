# Deploy Spike1 Web App

## Local Testing

```bash
# Install dependencies
pip install -r spike1/requirements.txt

# Run the app
streamlit run spike1/app.py
```

The app will open in your browser at `http://localhost:8501`

## Deploy to Streamlit Cloud (Free)

1. **Push your code to GitHub** (already done!)

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Sign in with GitHub**

4. **Click "New app"**

5. **Fill in the deployment settings:**
   - Repository: `RalphX1/SpikeSpike`
   - Branch: `main`
   - Main file path: `spike1/app.py`

6. **Click "Deploy"**

Your app will be live at: `https://[your-app-name].streamlit.app`

## Configuration

Streamlit will automatically:
- Install dependencies from `spike1/requirements.txt`
- Run the app from `spike1/app.py`
- Provide a public URL

## Updating the App

Any push to the `main` branch will automatically redeploy the app.

## Custom Domain (Optional)

Streamlit Cloud allows custom domains on paid plans, or you can use the free `.streamlit.app` subdomain.
