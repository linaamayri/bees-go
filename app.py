from flask import Flask
from main.views import main

port = 5100

def create_app():
    app = Flask(__name__, static_url_path="/")
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config.update(GEOIPIFY_API_KEY='at_LSNKTbc11U0k0I3ToGHosGGluWX22')
    # register blueprint
    app.register_blueprint(main)
    
    return app

if __name__ == "__main__":
    create_app().run()