"""
wsgi.py
Production entry point for deployment platforms (Render, Railway, PythonAnywhere,
or any host that runs a WSGI server like gunicorn).

Flask's own built-in server (python app/app.py) is fine for development, but
is not meant to face the internet — a WSGI server like gunicorn is what
actually gets used in production. This file just points gunicorn at the
Flask app object defined in app/app.py.

Deploy command (see Procfile): gunicorn wsgi:application
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app import app as application

if __name__ == "__main__":
    application.run()
