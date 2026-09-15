# Free deployment guide for this website

This project is a Flask app for sign-language recognition. The easiest free hosting options are:

- Render
- Railway
- Hugging Face Spaces

## Important note about webcam access

This app uses webcam and MediaPipe on the server side. In most free hosting platforms, the server does not have access to your local camera. So the webcam feature usually works best on:

- your local machine, or
- a machine/VM with a real camera attached

For a public website, you can still deploy the web interface, but a real webcam testing flow may need a browser-based JavaScript version instead of a server-side camera stream.

---

## Option 1: Deploy on Render (recommended)

### 1) Prepare the project

Make sure these files exist in the root of the project:

- `app.py`
- `requirements.txt`
- `templates/`
- `src/`

Add this package to `requirements.txt` if it is not already there:

```txt
gunicorn==22.0.0
```

### 2) Create a start command

Use this command in Render:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

### 3) Create a GitHub repo

```bash
git init
git add .
git commit -m "Deploy-ready app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 4) Deploy on Render

1. Go to https://render.com
2. Sign in with GitHub
3. Click New > Web Service
4. Connect this repository
5. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
6. Click Create Web Service

### 5) Open the deployed URL

Render gives you a public URL like:

```text
https://your-app-name.onrender.com
```

---

## Option 2: Deploy on Railway

1. Go to https://railway.app
2. Sign in with GitHub
3. Create a new project
4. Deploy from this repository
5. Set the start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

6. Railway will give you a public URL automatically.

---

## Option 3: Deploy on Hugging Face Spaces

This is easier for simple Flask apps, but it may not be the best fit for a webcam-heavy ML app.

1. Go to https://huggingface.co/spaces
2. Create a new Space
3. Choose Python as the framework
4. Upload the project or connect GitHub
5. Use the app entry file as your Flask app

Example:

```bash
python app.py
```

If the app expects a local camera, it may not work properly on a hosted environment.

---

## Recommended deployment setup for this exact app

For this project, the most practical free deployment is:

- use Render or Railway for the web app
- keep webcam recognition local if the app needs direct camera access
- deploy only the UI and model-serving flow if the browser handles the camera

---

## Required environment notes

This app depends on:

- Flask
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- scikit-learn
- pandas

These are already listed in `requirements.txt`, but free hosting may have limits on large ML packages and startup time. If deployment fails due to size or build time:

- use a smaller model
- remove unnecessary packages
- keep only production dependencies
- use a stronger free plan or a paid tier

---

## Quick deploy example

```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

Then configure the host with:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## Best advice

If your goal is a free public website:

- Deploy the Flask app to Render or Railway
- Keep the model inference and UI available there
- Use local webcam testing if direct camera access is required

If you want, I can also create a ready-to-use `render.yaml` file or a production-ready `Procfile` for this project.
