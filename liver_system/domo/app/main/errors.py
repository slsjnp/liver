from flask import render_template,current_app
from . import main
import flask.logging
import logging.handlers
@main.app_errorhandler(404)
def page_not_found(e):
    return render_template('pages-error-404.html'), 404

# 设置日志
