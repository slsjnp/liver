# -*- coding: utf-8 -*-
import os
import logging
import sqlite3

basedir = os.path.abspath(os.path.dirname(__file__))

con = sqlite3.connect("Test.db")


class InfoFilter(logging.Filter):
    def filter(self, record):
        """only use INFO
        筛选, 只需要 INFO 级别的log
        :param record:
        :return:
        """
        if logging.INFO <= record.levelno < logging.ERROR:
            # 已经是INFO级别了
            # 然后利用父类, 返回 1
            return super().filter(record)
        else:
            return 0


class config(object):
    CSRF_ENABLED = True
    SECRET_KEY = 'this is a secret string'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_AS_ASCCII = False
    JSONIFY_MIMETYPE = "application/json;charset=utf-8"
    DROPZONE_ALLOWED_FILE_CUSTOM = True
    DROPZONE_ALLOWED_FILE_TYPE = '.dcm'
    SQLALCHEMY_ECHO = True
    BOOTSTRAP_SERVE_LOCAL = True
    STATIC = '/data/bucket'
    
    # STATIC = '/domo/app/static'
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://liver:liver@liver-pg11:5432/liver'
    # SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://liver:liver@127.0.0.1:5432/liver'
    DROPZONE_DEFAULT_MESSAGE = '请点击或拖拽上传.'

    # LOG_PATH = os.path.join(basedir, 'logs')
    # LOG_PATH_ERROR = os.path.join(LOG_PATH, 'error.log')
    # LOG_PATH_INFO = os.path.join(LOG_PATH, 'info.log')
    # LOG_FILE_MAX_BYTES = 100 * 1024 * 1024
    # 轮转的数量是10个
    # LOG_FILE_BACKUP_COUNT = 10
    def init_app(app):
        pass


class DevelopmentConfig(config):
    DEBUG = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False


class TestingConfig(config):
    TESTING = True


class ProductionConfig(config):
    pass


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
