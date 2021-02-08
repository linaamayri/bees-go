from flask import Flask
from main.views import main

port = 5100

def create_app():
    app = Flask(__name__, static_url_path="/")
    
    # register blueprint
    app.register_blueprint(main)
    
    return app

if __name__ == "__main__":
    create_app().run()