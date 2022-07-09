# -*- coding: utf-8 -*-
from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from domo.app import db
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3


def get_conn():
    return sqlite3.connect("test.db")


class Case(db.Model):
    __tablename__ = 'cases'
    case_id = db.Column(db.String(32), primary_key=True)
    identify_card = db.Column(db.String(18), ForeignKey("patients.identify_card"))
    height = db.Column(db.String(10))
    weight = db.Column(db.String(10))
    body_surface_area = db.Column(db.String(100))
    real_liver_volume = db.Column(db.String(100))
    liver_smoothness = db.Column(db.String(100))
    liver_spleen_ratio = db.Column(db.String(100))
    real_spleen_volume = db.Column(db.String(100))
    standard_liver_volume = db.Column(db.String(100))
    tumor_volume = db.Column(db.String(100))
    date = db.Column(db.DateTime, default=datetime.now)
    dcm_count = db.Column(db.Integer)
    patient = relationship("Patient", backref="my_cases")
    information = db.Column(db.String(100))

    __table_args__ = {'extend_existing': True}


class Patient(db.Model):
    __tablename__ = 'patients'
    doctor_id = db.Column(db.String(32), ForeignKey("doctors.doctor_id"))
    identify_card = db.Column(db.String(18), primary_key=True)
    name = db.Column(db.String(20))
    sex = db.Column(db.Boolean)
    age = db.Column(db.String(3))
    doctors = relationship("Doctor", backref="my_patients")
    patient = relationship("Case", backref="my_cases", cascade="all,delete-orphan", single_parent=True)
    __table_args__ = {'extend_existing': True}


class Doctor(db.Model, UserMixin):
    __tablename__ = 'doctors'
    doctor_id = db.Column(db.String(32), primary_key=True)
    username = db.Column(db.String(20), unique=True)
    hash_password = db.Column(db.String(255))
    phone_number = db.Column(db.String(11))
    department = db.Column(db.String(20))
    name = db.Column(db.String(20))
    __table_args__ = {'extend_existing': True}

    def get_id(self):
        return self.id

    def __repr__(self):
        return '<Doctor %r>' % self.username

    @property
    def password(self):
        raise AttributeError('password cannot be read')

    @password.setter
    def password(self, password):
        self.hash_password = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.hash_password, password)

    @staticmethod
    def get(user_id):
        if not user_id:
            return None
