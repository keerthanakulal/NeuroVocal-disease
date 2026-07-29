from app_launcher import app, load_models

# Pre-load the models in the Gunicorn worker thread
load_models()

if __name__ == '__main__':
    app.run()
