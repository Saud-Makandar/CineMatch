"""
server.py
=========
Web service entrypoint for CineMatch Movie Recommendation System.
Imports and runs the Flask app from movie_recommender.py.
"""

import os
from movie_recommender import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting CineMatch web service on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
