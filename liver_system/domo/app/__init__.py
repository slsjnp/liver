from flask import Flask, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from domo.config import config
import logging
from flask_bootstrap import Bootstrap
from domo.app.extensions import bcrypt
from flask_dropzone import Dropzone
from flasgger import Swagger, swag_from

db = SQLAlchemy()
dropzone = Dropzone()
swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['title'] = 'LIVER SYSTEM API'
swagger_config['description'] = 'powered by 519'
swagger = Swagger()
bootstrap = Bootstrap()


# login_manager = LoginManager()
class MyResponse(Response):
    @classmethod
    def force_type(cls, rv, environ=None):
        if isinstance(rv, dict):
            rv = jsonify(rv)
        return super(MyResponse, cls).force_type(rv, environ)


def create_app(config_name):
    app = Flask(__name__)
    app.response_class = MyResponse
    swagger.init_app(app)
    app.config['BOOTSTRAP_SERVE_LOCAL'] = True
    bootstrap.init_app(app)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    dropzone.init_app(app)

    # login_manager.init_app(app)
    db.init_app(app)
    bcrypt.init_app(app)

    handler = logging.FileHandler(filename="server.log", encoding='utf-8')
    handler.setLevel("DEBUG")
    format_ = "%(asctime)s[%(name)s][%(levelname)s] :%(levelno)s: %(message)s"
    formatter = logging.Formatter(format_)
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint, url_prefix='/')

    return app
