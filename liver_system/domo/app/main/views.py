# -*- coding: utf-8 -*-
import datetime
import pynvml
import psutil
import torch
import os
import random
import re
import shutil
from aliyunsdkcore.request import CommonRequest
import stat
import tempfile
import time
import zipfile
from io import BytesIO
from multiprocessing import Process
import numpy as np
from aliyunsdkcore.client import AcsClient
import cv2
import pydicom
# from datetime import datetime
import PIL.Image as Image
from flask import request as requ
from flask import session, redirect, url_for, flash, current_app, request, jsonify, make_response, \
    send_file
from werkzeug.utils import secure_filename
import uuid

from Smooth.smooth import smooth
from All_hospital.bmp_to_nii import bmp_to_nii
from All_hospital.test import model_test

from domo.app import db
# ALLOWED_EXTENSIONS = set(['txt', 'dcm', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from domo.app.load_dcm import get_pixeldata, setDicomWinWidthWinCenter
from domo.app.models import *
from domo.app.s3 import CephS3BOTO3, getInformation
from domo.manage import app
from vol_3d.vol2obj import obj_make
from volume.get_volume import liver_tumor_spleen
from . import main

ALLOWED_EXTENSIONS = set(['dcm', 'png', 'jpg', 'gif'])
conn = CephS3BOTO3()
join = os.path.join


# @main.before_request
# def create_db():
# 
#     db.drop_all()
#     db.create_all()
#     db.session.commit()
# 

# def login_check():
#     if session.get('user') is None:
#         form = LoginForm()
#         sign_form = Sign_inForm()
#         return render_template('login.html', form=sign_form)

def get_avaliable_memory(device):
    ava_mem = 0
    if device == torch.device('cuda:0'):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示第�~@�~]~W�~X��~M�
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        ava_mem = round(meminfo.free / 1024 ** 2)
        print('current available video memory is' + ' : ' + str(round(meminfo.free / 1024 ** 2)) + ' MIB')

    elif device == torch.device('cpu'):
        mem = psutil.virtual_memory()
        print('current available memory is' + ' : ' + str(round(mem.used / 1024 ** 2)) + ' MIB')
        ava_mem = round(mem.used / 1024 ** 2)

    return ava_mem


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@main.route('/indexImage/<int:flag>', methods=['GET'])
def indexImage(flag):
    """
    This is the image API
        Call this 获取首页图片 api passing a flag and get back its features
            ---
            tags:
              - Private Cloud Platform API
            parameters:
              - name: flag
                in: path
                required: true
            responses:
              500:
                description: 系统错误
              200:
                description: 首页图片获取
                schema:
                  id: index_image
                  properties:
                    code:
                      type: integer
                      description: 0 成功， 其他值表示失败
                    data:
                      type: object
                      properties:
                        image:
                          type: object
                          properties:
                            background:
                              type: string
                            module:
                              type: object
                              properties:
                                name:
                                  type: string
                            word:
                              type: array
                              items:
                                type: array
                                items:
                                  type: string
                    msg:
                      type: string

    """
    if flag == 1:
        conn.get_list_object('private', 'image{}st'.format(flag))
    return {"code": "0", "data": "", "msg": ""}


@main.route('/login', methods=['POST'])
def login():
    """
    This is the login API
        Call this 登录 api passing a 用户名或手机号 and 密码 and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              username:
                type: string
                description: 用户名
              password:
                type: string
                description: 密码
        responses:
          500:
            description: 系统错误
          200:
            description: 登录成功
            schema:
              id: login
              properties:
                code:
                  type: integer
                  description: 0 成功， 其他值表示失败
                  default:
                data:
                  type: object
                  properties:
                    doctor_id:
                      type: string
                      description: 医生标识  doctor_id
                    doctor_name:
                      type: string
                      description: 医生姓名
                    doctor_dep:
                      type: string
                      description: 医生科室
                msg:
                  type: string
                  description: 消息信息

    """

    data = request.get_json()
    username = data['username']
    password = data['password']
    if username and password:
        if len(username) == 11:
            user = Doctor.query.filter_by(phone_number=username).first()
            if user is None:
                user = Doctor.query.filter_by(username=username).first()
        else:
            user = Doctor.query.filter_by(username=username).first()
        if user is not None:
            if user.verify_password(password) is True:
                # session['user'] = user.username
                # session['doctor_id'] = user.doctor_id
                result = {"code": 0, "data": {"doctor_id": user.doctor_id,
                                              "doctor_name": user.name,
                                              "doctor_dep": user.department
                                              }, "msg": "登录成功"}
                return result
            return {"code": 10001, "data": "", "msg": "密码错误"}
        return {"code": 10003, "data": "", "msg": "用户名不存在"}
    return {"code": 10002, "data": "", "msg": "用户名或密码为空"}


@main.route('/department', methods=['GET'])
def get_department():
    """
    This is the get_department API
        Call this 科室 get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        responses:
          500:
            description: 错误
          200:
            description: 医生所属科室信息
            schema:
              id: department
              properties:
                code:
                  type: integer
                  description: 0 成功，其他值表示失败
                data:
                  type: array
                  items:
                    type: string
                    description: 获取科室
    """
    result = {"code": 0, "data": ["内科", "消化内科"]}
    return jsonify(result)


@main.route('/sign_in', methods=['POST'])
def sign_in():
    """
    This is the sign_in API
        Call this 注册 api passing 注册信息 and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              username:
                type: string
                required: true
                description: The doctor username
              name:
                type: string
                required: true
                description: The doctor name
              password:
                type: string
                required: true
                description: The doctor password
              department:
                type: string
                required: true
                description: The doctor department
              pre_phone_number:
                type: string
                required: true
                description: The doctor pre_phone_number
              phone_number:
                type: string
                required: true
                description: The doctor phone_number
              identify:
                type: string
                required: true
                description: The doctor phone_number identify
        responses:
          500:
            description: Error
          200:
            description: A doctor login information
            schema:
              id: sign_in
              properties:
                code:
                  type: integer
                  description: 0 成功， 其他值表示失败
                data:
                  type: string
                msg:
                  type: string
        """
    global uuid
    data = request.get_json()
    department = data['department']
    identify = data['identify']
    name = data['name']
    password = data['password']
    phone_number = data['phone_number']
    pre_phone_number = data['pre_phone_number']
    username = data['username']
    # if username != phone_number
    if (datetime.now().second - session['time'].second) > 300:
        return {
            "code": 10003,
            "data": "",
            "msg": "验证码超时"
        }
    if identify.lower() != session['identify']:
        return {
            "code": 10002,
            "data": "",
            "msg": "验证码错误"
        }
    phone = Doctor.query.filter_by(phone_number=phone_number).first()
    user = Doctor.query.filter_by(username=username).first()
    if phone is None and user is None:
        uid = uuid.uuid1().hex
        doctor = Doctor(doctor_id=uid, username=username, password=password,
                        phone_number=phone_number,
                        department=department, name=name)  # 与表单对应数据库
        db.session.add(doctor)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return {"code": 10001, "data": "", "msg": "注册信息有误"}
        session['user'] = doctor.username
        session['doctor_id'] = doctor.doctor_id
        result = {"code": 0, "data": "", "msg": "注册成功"}
        return result
    return {"code": 10002, "data": "", "msg": "该用户已注册"}


@main.route('/reset_password', methods=['POST'])
def reset_password():
    """
    This is the reset_password API
        Call this 重置密码 api passing phone number and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: data
            in: body
            required: true
            properties:
              password:
                type: string
                required: true
                description: The doctor password
              pre_phone_number:
                type: string
                required: true
                description: The doctor pre_phone_number
              phone_number:
                type: string
                required: true
                description: The doctor phone_number
              identify:
                type: string
                required: true
                description: The doctor phone_number identify
        responses:
          500:
            description: Error
          200:
            description: A doctor reset_password information
            schema:
              id: reset-password
              properties:
                code:
                  type: string
                  description: 0 成功， 其他值表示失败
                  default:
                data:
                  type: string
                msg:
                  type: string
        """
    data = request.get_json()
    phone_number = data['phone_number']
    identify = data['identify']
    if datetime.now().second - session['time'].second > 300:
        return {
            "code": 10003,
            "data": "",
            "msg": "验证码过期"
        }
    if identify.lower() != session['identify']:
        return {
            "code": 10002,
            "data": "",
            "msg": "验证码错误"
        }
    password = data['password']
    doctor = Doctor.query.filter_by(phone_number=phone_number).first()
    try:
        doctor.hash_password = generate_password_hash(password)
        db.session.commit()
    except Exception as e:
        print(repr(e))
        db.session.rollback()
        # flash('User reset password failed')
        return {"code": 10001, "data": "", "msg": "重置密码失败"}
    return {"code": 0, "data": "", "msg": "重置密码成功"}


@main.route('/identify', methods=['POST'])
def identify():
    """
    This is the identify API
        Call this 验证码api passing phone number and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: data
            in: body
            required: true
            properties:
              pre_phone_number:
                type: string
                required: true
                description: The doctor pre_phone_number
              phone_number:
                type: string
                required: true
                description: The doctor phone_number
        responses:
          500:
            description: Error
          200:
            description: A doctor phone number identify information
            schema:
              id: identify
              properties:
                code:
                  type: integer
                  description: 0 成功，其他值表示失败
                data:
                  type: string
                  description: 验证码
        """
    data = requ.get_json()  # request被阿里云接口占用 此处用别名requ
    phone_number = data['phone_number']
    if len(phone_number) != 11:
        return {
            "code": 10001,
            "data": "请输入正确手机号"
        }
    client = AcsClient('LTAI4GHR3opdFMbbGA5W6CoW', '5N3V2R5ZfIKZjrkBPn6B7RwB2YquuY', 'cn-hangzhou')
    str = 'QWERTYUPASDFGHJKZXCVBNMqwertyupasdfghjkzxcvbnm987654321'  # 由此产生四个随机数
    code = ''  # 先占位
    for i in range(4):
        ran = random.randint(0, len(str) - 1)  # 因为下标从零开始，所以长度减一
        code += str[ran]
    request = CommonRequest()
    request.set_accept_format('json')
    request.set_domain('dysmsapi.aliyuncs.com')
    request.set_method('POST')
    request.set_protocol_type('https')  # https | http
    request.set_version('2017-05-25')
    request.set_action_name('SendSms')

    request.add_query_param('RegionId', "cn-hangzhou")
    request.add_query_param('PhoneNumbers', phone_number)
    request.add_query_param('SignName', "肝脏影像分析短信平台")
    request.add_query_param('TemplateCode', "SMS_194635535")
    identify = "{\"code\":\"" + code + "\"}"
    request.add_query_param('TemplateParam', identify)
    session['identify'] = code.lower()
    session['time'] = datetime.now()
    response = client.do_action_with_exception(request)
    print(response)
    return {
        "code": 0,
        "data": ""
    }


@main.route('/patientInfo/patientFiles/<string:doctor_id>', methods=['GET'])
def patients_file(doctor_id):
    """
    This is the patients_file API
        Call this 查询患者档案 api passing doctor_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: doctor_id
            in: path
            required: true
          - name: query
            in: query
            type: string
            required: false
            description: 查询关键字
          - name: pagenum
            in: query
            type: integer
            required: true
            default: 1
            description: The page number
          - name: pagesize
            in: query
            type: integer
            default: 10
            required: true
            description: The page size
        responses:
          500:
            description: Error
          200:
            description: A doctor's patients_file information
            schema:
              id: doctor_patients
              properties:
                code:
                  type: integer
                  description: The doctor name
                  default:
                data:
                  type: object
                  description: The doctor's patients
                  properties:
                    department:
                      type: string
                    name:
                      type: string
                    sum_pages:
                      type: integer
                    sum_patients:
                      type: integer
                    patients:
                      type: array
                      items:
                        type: object
                        properties:
                          identify_card:
                            type: string
                            description: 患者身份证号
                          name:
                            type: string
                          sex:
                            type: boolean
                          age:
                            type: string
                msg:
                  type: string
        """
    query = request.args.get('query')
    doctor_id = doctor_id
    pagenum = int(request.args.get('pagenum'))
    pagesize = int(request.args.get('pagesize'))
    doctor = Doctor.query.filter_by(doctor_id=doctor_id).first()
    if doctor is None:
        return {
            "code": 10001,
            "data": "",
            "msg": "医生不存在"
        }
    doctor_name = doctor.name
    department = doctor.department
    patients = []
    if query:
        if len(query) < 15:
            patient_num = Patient.query.filter(Patient.name.like(r'%{keyword}%'.format(keyword=query))).all()
            # patient_num = Patient.query.filter_by(name=query).all()
        else:
            patient_num = Patient.query.filter_by(identify_card=query).all()
        if patient_num:
            sum_patients = int(len(patient_num))
            sum_pages = sum_patients // pagesize
            if sum_patients % pagesize > 0:
                sum_pages += 1
            if sum_pages < pagenum:
                return {
                    "code": 10003,
                    "data": "",
                    "msg": "无效页码"
                }
            for patient in patient_num[pagesize * (pagenum - 1):(pagesize * (pagenum - 1) + pagesize)]:
                patient_info = {}
                patient_info['name'] = patient.name
                patient_info['identify_card'] = patient.identify_card
                sex, age = getInformation(patient_info['identify_card'])
                if sex:
                    sex = '男'
                else:
                    sex = '女'
                patient_info['sex'] = sex
                patient_info['age'] = age
                patients.append(patient_info)
            return {
                "code": 0,
                "data": {
                    "sum_pages": sum_pages,
                    "sum_patients": sum_patients,
                    "department": department,
                    "name": doctor_name,
                    "patients": patients
                },
                "msg": "成功"
            }
        return {
            "code": 10002,
            "data": "",
            "msg": "医生未上传病人信息"
        }
    sum_patients = len(doctor.my_patients)
    sum_pages = sum_patients // pagesize
    if sum_patients % pagesize > 0:
        sum_pages += 1
    if sum_pages < pagenum:
        return {
            "code": 10003,
            "data": "",
            "msg": "无效页码"
        }
    if doctor.my_patients:
        for patient in doctor.my_patients[pagesize * (pagenum - 1):(pagesize * (pagenum - 1) + pagesize)]:
            patient_info = {}
            patient_info['name'] = patient.name
            patient_info['identify_card'] = patient.identify_card
            sex, age = getInformation(patient_info['identify_card'])
            if sex:
                sex = '男'
            else:
                sex = '女'
            patient_info['sex'] = sex
            patient_info['age'] = age
            patients.append(patient_info)
        return {
            "code": 0,
            "data": {
                "sum_pages": sum_pages,
                "sum_patients": sum_patients,
                "department": department,
                "name": doctor_name,
                "patients": patients
            },
            "msg": "ok"
        }
    return {
        "code": 10002,
        "data": {
            "sum_pages": sum_pages,
            "sum_patients": sum_patients,
            "department": department,
            "name": doctor_name,
            "patients": patients
        },
        "msg": "医生未上传病人信息"
    }


@main.route('/patientInfo/newPatientFiles/submission', methods=['POST'])
def new_patients():
    """
    This is the new_patients API
        Call this 新建患者档案 api passing a name and identify_card and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              doctor_id:
                type: string
                required: true
              description: The doctor id
              name:
                type: string
                required: true
                description: The patient name
              identify_card:
                type: string
                required: true
                description: The patient identify_card
        responses:
          500:
            description: Error The identify_card is not correct!
          200:
            description: 新建患者档案成功
            schema:
              id: new_patients
              properties:
                code:
                  type: integer
                  description: 0 成功，其他值表示失败
                data:
                  type: string
                  description:
    """
    data = request.get_json()
    doctor_id = data.get('doctor_id')
    identify_card = data.get('identify_card')
    if len(identify_card) != 18:
        return {
            "code": 10001,
            "data": "请输入正确身份证号"
        }
    if doctor_id is None:
        return {
            "code": 10001,
            "data": "参数错误"
        }
    patient = Patient.query.filter_by(identify_card=identify_card).first()
    if patient:
        return {"code": 10001, "data": "身份证号已注册"}
    name = data['name']
    sex, age = getInformation(identify_card)
    sex = bool(sex)
    if age < 0:
        return {"code": 10001, "data": "身份证信息不合法"}
    age = str(age)

    patient = Patient(name=name, identify_card=identify_card, doctor_id=doctor_id, sex=sex, age=age)
    db.session.add(patient)
    try:
        db.session.commit()
        flash('new patient created !')
    except Exception as e:
        db.session.rollback()
        return {"code": 10001, "data": "身份证信息不合法"}
    session['identify_card'] = identify_card
    return {"code": 0, "data": ""}


@main.route('/patientInfo/newPatientCases/quality', methods=['POST'])
def quality():
    """
    This is the new_case API
        Call this 新增病例 api passing 身高，体重，dcm数据 and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              height:
                type: string
                required: true
              description: The patient height
              weight:
                type: string
                required: true
                description: The patient weight
              information:
                type: string
              identify_card:
                type: string
                required: true
                description: The patient identify_card
        responses:
          500:
            description: Error
          200:
            description: The patient new_cases
            schema:
              id: new_patient_case
              properties:
                code:
                  type: integer
                data:
                  type: string
                  description: case_id

    """
    data = request.get_json()
    height = data.get('height')
    weight = data.get('weight')
    info = data.get('information')
    identify_card = data.get('identify_card')
    if len(height) > 2:
        return {
            "code": 10001,
            "data": "请输入正确身高"
        }
    # if len(weight) > 10:
    #     return {
    #         "code": 10001,
    #         "data": "请输入正确体重"
    #     }
    # if len(identify_card) != 18:
    #     return {
    #         "code": 10001,
    #         "data": "请输入正确身份证号"
    #     }
    uid = uuid.uuid1().hex
    # case = Case(case_id=uid, height=height, weight=weight, identify_card=identify_card, information=info)
    # db.session.add(case)
    # try:
    #     db.session.commit()
    #     flash('Case created !')
    # except Exception as e:
    #     print(repr(e))
    #     db.session.rollback()
    #     flash('Case create failed')
    #     return {
    #         "code": 10001,
    #         "data": ""
    #     }
    # case_id = case.case_id
    return {
        "code": 0,
        "data": uid
    }
    # dcm文件上传


@main.route('/patientInfo/newPatientCases/submission', methods=['POST'])
def new_cases():
    """
    This is the new_case API
        Call this 新增病例 api passing 身高，体重，dcm数据 and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              height:
                type: string
                required: true
              description: The patient height
              weight:
                type: string
                required: true
                description: The patient weight
              information:
                type: string
              identify_card:
                type: string
                required: true
                description: The patient identify_card
        responses:
          500:
            description: Error
          200:
            description: The patient new_cases
            schema:
              id: new_patient_case
              properties:
                code:
                  type: integer
                data:
                  type: string
                  description: case_id

    """
    data = request.get_json()
    height = data.get('height')
    weight = data.get('weight')
    info = data.get('information')
    identify_card = data.get('identify_card')
    if len(height) > 10:
        return {
            "code": 10001,
            "data": "请输入正确身高"
        }
    if len(weight) > 10:
        return {
            "code": 10001,
            "data": "请输入正确体重"
        }
    if len(identify_card) != 18:
        return {
            "code": 10001,
            "data": "请输入正确身份证号"
        }
    uid = uuid.uuid1().hex
    case = Case(case_id=uid, height=height, weight=weight, identify_card=identify_card, information=info)
    db.session.add(case)
    try:
        db.session.commit()
        flash('Case created !')
    except Exception as e:
        print(repr(e))
        db.session.rollback()
        flash('Case create failed')
        return {
            "code": 10001,
            "data": ""
        }
    case_id = case.case_id
    return {
        "code": 0,
        "data": case_id
    }


@main.route('/patient/schedule/<string:case_id>', methods=['GET'])
def schedule(case_id):
    """
    This is the schedule API
        Call this 处理进度 api passing a name and case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: case_id
            in: path
            type: string
            required: true
            description: 患者病历号
        responses:
          500:
            description: Error The case_id is not correct!
          200:
            description: 查询处理进度
            schema:
              id: identify
              properties:
                code:
                  type: integer
                  description: 0 处理完成，其他值表示未完成
                msg:
                  type: string
                  description:
    """
    case = Case.query.filter_by(case_id=case_id).first()
    # sm = case.real_liver_volume
    if session[case_id]['volume'] == 1:
        return {
            "code": 0,
            "msg": "结果处理完毕"
        }
    elif os.path.exists(os.path.join(os.getcwd(), '/data/bucket/{}/dcm/'.format(case.case_id))):
        return {
            "code": 10001,
            "msg": "结果正在处理中"
        }
    return {
        "code": 10002,
        "msg": "未上传dcm文件"
    }


@main.route('/patientInfo/del', methods=['POST'])
def delete():
    """
    This is the patient_info delete API
        Call this 删除患者档案 api passing identify_card and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              identify_card:
                type: array
                items:
                  type: string
                  required: true
                  description: The patient identify_card
        responses:
          500:
            description: Error
          200:
            description: The patient deleted !
            schema:
              id: patient_del
              properties:
                code:
                  type: integer
                  description: 0表示成功，10001表示失败,删除失败的用户列表在data中
                data:
                  type: array
                  items:
                    type: string
                    description: 删除失败的用户列表

    """
    data = request.get_json()
    data = data.get('identify_card')
    result = []
    for identify_card in data:
        patient = Patient.query.filter_by(identify_card=identify_card).first()
        if patient is None:
            result.append(identify_card)
        # return {"code": 10001, "data": ""}  # 患者不存在
        cases = patient.my_cases
        filepath=os.getcwd()
        # cases_list = []
        for i in cases:
            delete_file(os.path.join(filepath + app.config['STATIC'], i.case_id))
            # conn.delete('bucket', i.case_id)

        db.session.delete(patient)
        try:
            db.session.commit()
            flash('Patient deleted !')
        except Exception as e:
            print(repr(e))
            db.session.rollback()
            flash('Patient delete failed')
            return 10002  # 删除失败
    if result:
        return {"code": 10001, "data": result}
    return {"code": 0, "data": ""}


@main.route('/patientInfo/patientCases/del', methods=['POST'])
def case_delete():
    """
    This is the patient_cases delete API
        Call this 删除患者病例 api passing cases_card and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              case_id:
                type: array
                items:
                  type: integer
                  required: true
                  description: The patient case_id
        responses:
          500:
            description: Error
          200:
            description: The patient's cases deleted !
            schema:
              id: patient-cases-del
              properties:
                code:
                  type: integer
                  description: 0表示成功，10001表示失败,删除失败的用户列表在data中
                data:
                  type: array
                  items:
                    type: string
                    description: 删除失败的用户列表

    """
    data = request.get_json()
    result = []
    data = data.get('case_id')
    for case_id in data:
        patient_case = Case.query.filter_by(case_id=case_id).first()
        if patient_case is None:
            result.append(case_id)
        else:
            db.session.delete(patient_case)
        try:
            db.session.commit()
            flash('Patient_case deleted !')
        except Exception as e:
            print(repr(e))
            db.session.rollback()
            flash('Patient_case delete failed')
            # return {"code": 10001}  # 删除失败
        conn.delete('bucket', case_id)
    if result:
        return {"code": 0, "data": ""}
    return {"code": 10001, "data": "{}".format(result)}


@main.route('/patientInfo/patientcases/<string:identify_card>', methods=['GET'])
def patient_case(identify_card):
    """
    This is the patient_case API
        Call this 查看病人所有病例api passing identify_card and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: identify_card
            in: path
            type: string
            required: true
            description: The patient identify_card
        responses:
          500:
            description: Error
          200:
            description: The patient_case
            schema:
              id: patient-cases
              properties:
                code:
                  type: integer
                data:
                  type: object
                  properties:
                    case:
                      type: array
                      items:
                        type: object
                        properties:
                          case_id:
                            type: string
                          time:
                            type: string
                          height:
                            type: string
                          weight:
                            type: string
                          body_surface_area:
                            type: string
                          real_liver_volume:
                            type: string
                          standard_liver_volume:
                            type: string
                          real_spleen_volume:
                            type: string
                          liver_spleen_ratio:
                            type: string
                          liver_smoothness:
                            type: string
                          information:
                            type: string
                          tumor_volume:
                            type: string
                          schedule:
                            type: integer
                msg:
                  type: string
                case_count:
                  type: integer
    """
    patient = Patient.query.filter_by(identify_card=identify_card).first()
    if patient is None:
        return {"code": 10001, "data": "", "msg": " 病人不存在", "case_count": 0}
    result = []
    for case in patient.my_cases:
        case_info = {}
        case_info['real_liver_volume'] = case.real_liver_volume
        case_info['case_id'] = case.case_id
        case_info['body_surface_area'] = case.body_surface_area
        case_info['height'] = case.height
        case_info['weight'] = case.weight
        case_info['liver_smoothness'] = case.liver_smoothness
        case_info['liver_spleen_ratio'] = case.liver_spleen_ratio
        case_info['standard_liver_volume'] = case.standard_liver_volume
        case_info['time'] = case.date.strftime('%Y-%m-%d')
        case_info['tumor_volume'] = case.tumor_volume
        case_info['real_spleen_volume'] = case.real_spleen_volume
        case_info['information'] = case.information
        if case_info['real_liver_volume']:
            case_info['schedule'] = 1
        else:
            try:
                conn.get_list_object('bucket', '{}/dcm'.format(case.case_id))
            except:
                Case.query.filter_by(case_id=case.case_id).delete()
                try:
                    db.session.commit()
                    continue
                except:
                    db.session.rollback()
                    continue
            else:
                case_info['schedule'] = 0
        result.append(case_info)
    case_count = len(result)
    result.sort(key=lambda x: x['time'])
    return {
        "code": 0,
        "data": result,
        "msg": "成功",
        "case_count": case_count
    }


@main.route('/patientInfo/chart/<string:identify_card>', methods=['GET'])
def patient_chart(identify_card):
    """
    This is the patient_chart API
        Call this 图表api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: type
            in: query
            type: string
            required: true
            description: 查询图表种类 volume:肝体积，smooth：平滑度，ratio：肝脾比
          - name: identify_card
            in: path
            type: string
            required: true
            description: The patient case_id
        responses:
          500:
            description: Error
          200:
            description: The patient_case_chart
            schema:
              id: patient-case-chart
              properties:
                code:
                  type: integer
                data:
                  type: array
                  items:
                    type: object
                    properties:
                      date:
                        type: string
                      value:
                        type: string
                msg:
                  type: string
                  description: 返回消息
"""
    query = request.args.get('type')
    patient = Patient.query.filter_by(identify_card=identify_card).first()
    if patient is None:
        return {"code": 10001, "data": "", "msg": " 病人不存在"}
    res = []

    if query == 'volume':
        for case in patient.my_cases:
            try:
                result = {}
                result['date'] = case.date.strftime('%Y-%m-%d')
                result['value'] = float(case.real_liver_volume)
                res.append(result)
            except:
                result = {
                    "date": case.date.strftime('%Y-%m-%d'),
                    "value": 0
                }
                res.append(result)
    elif query == 'smooth':
        for case in patient.my_cases:
            try:
                result = {}
                result['date'] = case.date.strftime('%Y-%m-%d')
                result['value'] = float(case.liver_smoothness)
                res.append(result)
            except:
                result = {
                    "date": case.date.strftime('%Y-%m-%d'),
                    "value": 0
                }
                res.append(result)
    elif query == 'ratio':
        for case in patient.my_cases:
            try:
                result = {}
                result['date'] = case.date.strftime('%Y-%m-%d')
                result['value'] = float(case.liver_spleen_ratio)
                res.append(result)
            except:
                result = {
                    "date": case.date.strftime('%Y-%m-%d'),
                    "value": 0
                }
                res.append(result)
    res.sort(key=lambda x: x['date'])
    return {
        "code": 0,
        "data": res,
        "msg": "成功"
    }


@main.route('/three/<string:case_id>', methods=['GET'])
def three(case_id):
    """
    This is three_images API
        Call this 3D api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: case_id
            in: path
            type: string
            required: true
            description: The patient case_id
          - name: type
            in: query
            type: string
            description: liver：肝脏模型， tumor：肿瘤模型，spleen：脾脏模型
        responses:
          500:
            description: Error
          200:
            description: The patient's cases' 3D models !
            schema:
              id: obj_url
              properties:
                code:
                  type: integer
                data:
                  type: string
                  description: obj url

    """
    organ = request.args.get('type')
    prefix = '{}/{}/obj/{}.obj'.format(case_id, organ, organ)
    try:
        obj_list = conn.get_list_object('bucket', prefix)
    except:
        return {
            "code": 10001,
            "data": "",
            "msg": "图片获取失败"
        }
    for obj in obj_list:
        if prefix == obj['Key']:
            return {
                "code": 0,
                "data": prefix
            }
    return {
        "code": 10001,
        "data": "",
        "msg": "不存在obj文件"
    }


@main.route('/two/<string:case_id>', methods=['GET'])
def two(case_id):
    """
    This is images API
        Call this 2D api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: case_id
            in: path
            type: string
            required: true
            description: The patient case id
          - name: organ
            in: query
            type: string
            description: The type of organ
        responses:
          500:
            description: Error
          200:
            description: The patient's cases' image !
            schema:
              id: bmp_url
              properties:
                code:
                  type: integer
                data:
                  type: array
                  items:
                    type: string
                    description: bmp名称

    """
    organ = request.args.get('organ')
    num = request.args.get('imagenum')
    if organ is None:
        organ = 'liver'
    prefix = '{}/{}/min_bmp/'.format(case_id, organ)
    try:
        bmp_list = conn.get_list_object('bucket', prefix)
    except:
        return {
            "code": 10001,
            "data": "",
            "msg": "图片获取失败"
        }
    result = []
    for bmp in bmp_list:
        result.append(bmp['Key'])
    pattern = re.compile(r'\d+')
    result.sort(key=lambda filenames: int(pattern.findall(filenames)[-1]))
    if num != 'undefined':
        length = len(result)
        res = []
        a = length // 6
        for i in range(0, length - 1, a):
            res.append(result[i])
        return {
            "code": 0,
            "data": res[1:5]
        }
    return {
        "code": 0,
        "data": result
    }


@main.route('/getFile')
def getFile():
    """
    This is the getFile API
        Call this 获取文件 passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: file_path
            in: query
            type: string
            required: true
            description: 获取文件
        responses:
          500:
            description: Error
          200:
            description: The patient_case_chart

            schema:
              id: patient-case-file
              properties:
"""
    path = request.args.get('file_path')
    try:
        resp = conn.download('bucket', path)
    except:
        return {
            "code": 10000,
            "msg": "no such file"
        }
    resp = resp['Body'].read()
    response = make_response(resp)
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set(
        'Content-Disposition', 'attachment', filename=path)
    return response


@main.route('/uploads/<string:case_id>', methods=['POST'])
def upload_file(case_id):
    """
    This is the upload API
        Call this 上传文件api passing dcm and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        consumes: ["multipart/form-data"]
        parameters:
          - in: formData
            name: file
            type: file
            required: true
            description: The patient dcm
          - name: case_id
            in: path
            required: true
            type: string
            description: The patient case_id
        responses:
          500:
            description: Error
          200:
            description: The patient new_cases 上传成功
            schema:
              id: patient
              properties:
                code:
                  type: integer
                data:
                  type: string
                msg:
                  type: string

    """
    case = Case.query.filter_by(case_id=case_id).first()
    if case is None:
        return {
            "code": 10002,
            "data": "",
            "msg": "病例不存在"
        }
    if request.method == 'POST' and request.files:
        file = request.files['file']
        file_name = file.filename
        filepath = os.getcwd()
        if not os.path.exists(os.path.join(filepath + app.config['STATIC'], case_id)):
            os.mkdir(os.path.join(filepath + app.config['STATIC'], case_id))
            os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcm'))
            os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcmtopng'))
        dcm_path = os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcm')
        png_path = os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcmtopng')
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
        file.save(os.path.join(dcm_path, file_name))
        # multi_dcm(dcm_path, png_path)
        # conn.upload('bucket', '{}/dcm/{}'.format(case_id, file_name),
        #             os.path.join(dcm_path, file_name))
        # os.listdir(dcm_path)
        return {
            "code": 0,
            "data": "ok",
            "msg": "ok"
        }
    return {
        "code": 10001,
        "data": "upload Error",
        "msg": "上传失败"
    }


def compress(src, save_path, quality=70):
    image = cv2.imread(src)
    old_size = os.path.getsize(src)
    cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    new_size = os.path.getsize(save_path)
    return old_size, new_size


# @main.route('/uploadsImage/<string:uid>', methods=['POST'])
# def upload_image(uid):
#     """
#     This is the upload API
#         Call this 上传文件api passing jpg, png, gif and get back its compression size
#         ---
#         tags:
#           - Imaging compression API READY
#         consumes: ["multipart/form-data"]
#         parameters:
#           - in: formData
#             name: file
#             type: file
#             required: true
#             description: The patient dcm
#           - in: path
#             name: id
#             required: true
#             type: string
#             description: image id
#         responses:
#           500:
#             description: Error
#           200:
#             description: The image new_cases 上传成功
#             schema:
#               id: upload_image
#               properties:
#                 code:
#                   type: integer
#                 data:
#                   type: object
#                   properties:
#                     filename:
#                       type: string
#                     guid:
#                       type: string
#                     id:
#                       type: string
#                     newsize:
#                       type: integer
#                     oldsize:
#                       type: integer
#                 msg:
#                   type: string
#
#     """
#     # 判断条件
#     if request.method == 'POST' and request.files:
#         file = request.files['file']
#         file_name = file.filename
#         # file_id = file.id
#         # guid = uuid.uuid1().hex
#         file_id = uid
#         id = request.args.get('id')
#
#         # file_guid = file.guid
#         filepath = os.getcwd()
#         day_time = time.strftime('%Y_%m_%d', time.localtime())
#
#         # 创建存储文件夹
#         if not os.path.exists(os.path.join(filepath + app.config['STATIC'], day_time)):
#             os.mkdir(os.path.join(filepath + app.config['STATIC'], day_time))
#             os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], day_time), 'src'))
#             os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], day_time), 'compress'))
#
#         # 目标路径
#         src_image_path = join(join(filepath + app.config['STATIC'], day_time), 'src')
#         compress_image_path = join(join(filepath + app.config['STATIC'], day_time), 'compress')
#         if file and allowed_file(file.filename):
#             file_name = secure_filename(file.filename)
#
#         # 存储到本地
#         file.save(join(src_image_path, file_id + file_name[-4:]))
#
#         # 压缩
#
#         src_image = join(src_image_path, file_id + file_name[-4:])
#         compress_image_name = join(compress_image_path, file_id) + '.jpg'
#
#         new_size, old_size = compress(src_image, compress_image_name)
#
#         # 上传到s3
#         conn.upload('image', 'src/{}'.format(file_id),
#                     compress_image_name)
#         conn.upload('image', 'compress/{}'.format(file_id),
#                     compress_image_name)
#
#         return {
#             "code": 0,
#             "data": {
#                 "filename": file_name,
#                 "guid": uid,
#                 "id": id,
#                 "new_size": new_size,
#                 "old_size": old_size,
#             },
#             "msg": "ok"
#         }
#     return {
#         "code": 10001,
#         "data": "upload Error",
#         "msg": "上传失败"
#     }

@main.route('/changeImage/<string:uid>', methods=['POST'])
def change_image(uid):
    """
    This is the upload API
        Call this 上传文件api passing jpg, png, gif and get back its compression size
        ---
        tags:
          - Todo Imaging Change API
        consumes: ["multipart/form-data"]
        parameters:
          - in: formData
            name: file
            type: file
            required: true
            description: The image which changes the label, Red is remove, Blue is add.
          - name: body
            in: body
            required: true
            properties:
              case_id:
                type: string
                required: true
                description: The patient case_id
              image_id:
                type: string
                required: true
                description: The image id of organ in case
              organ:
                type: string
                required: true
                default: liver
                description: [liver, spleen, tumor]

        responses:
          500:
            description: Error
          200:
            description: The image new_cases 上传成功
            schema:
              id: upload_image
              properties:
                code:
                  type: integer
                  description: 0代表成功， 其余代表失败
                data:
                  type: object
                  properties:
                    information:
                      type: string
                msg:
                  type: string

    """
    # 判断条件
    if request.method == 'POST' and request.files:
        file = request.files['file']
        file_name = file.filename
        # file_id = file.id
        # guid = uuid.uuid1().hex
        file_id = uid
        id = request.args.get('id')

        # file_guid = file.guid
        filepath = os.getcwd()
        day_time = time.strftime('%Y_%m_%d', time.localtime())

        # 创建存储文件夹
        if not os.path.exists(os.path.join(filepath + app.config['STATIC'], day_time)):
            os.mkdir(os.path.join(filepath + app.config['STATIC'], day_time))
            os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], day_time), 'src'))
            os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], day_time), 'compress'))

        # 目标路径
        src_image_path = join(join(filepath + app.config['STATIC'], day_time), 'src')
        compress_image_path = join(join(filepath + app.config['STATIC'], day_time), 'compress')
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)

        # 存储到本地
        file.save(join(src_image_path, file_id + file_name[-4:]))

        # 压缩

        src_image = join(src_image_path, file_id + file_name[-4:])
        compress_image_name = join(compress_image_path, file_id) + '.jpg'

        new_size, old_size = compress(src_image, compress_image_name)

        # 上传到s3
        conn.upload('image', 'src/{}'.format(file_id),
                    compress_image_name)
        conn.upload('image', 'compress/{}'.format(file_id),
                    compress_image_name)

        return {
            "code": 0,
            "data": {
                "filename": file_name,
                "guid": uid,
                "id": id,
                "new_size": new_size,
                "old_size": old_size,
            },
            "msg": "ok"
        }
    return {
        "code": 10001,
        "data": "upload Error",
        "msg": "上传失败"
    }

# @main.route('/compression/<string:type>', methods=['POST'])
# def compression_file(type):
#     """
#     This is the upload for compression API
#         Call this 上传文件api passing dcm and get back its features
#         ---
#         tags:
#           - Imaging Platform API READY
#         consumes: ["multipart/form-data"]
#         parameters:
#           - in: formData
#             name: file
#             type: file
#             required: true
#             description: The patient dcm
#           - name: type
#             in: path
#             required: true
#             type: string
#             description: The compression type
#           - name: args
#             in: path
#             required: false
#             description: The type arguements
#         responses:
#           500:
#             description: Error
#           200:
#             description: The compression new_cases 上传成功
#             schema:
#               id: patient
#               properties:
#                 code:
#                   type: integer
#                 data:
#                   type: string
#                 msg:
#                   type: string
#
#     """
#     # case = Case.query.filter_by(case_id=case_id).first()
#     # if case is None:
#     #     return {
#     #         "code": 10002,
#     #         "data": "",
#     #         "msg": "病例不存在"
#     #     }
#     '''
#     file: 只上传并返回当前用户的所有列表
#
#     '''
#     if session['file'] is None:
#         session['file'] = []
#     # session[file] =
#     if request.method == 'POST' and request.files:
#         file = request.files['file']
#         file_name = file.filename
#         filepath = os.getcwd()
#         if not os.path.exists(os.path.join(filepath + app.config['STATIC'], case_id)):
#             os.mkdir(os.path.join(filepath + app.config['STATIC'], case_id))
#             os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcm'))
#         dcm_path = os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), 'dcm')
#         if file and allowed_file(file.filename):
#             file_name = secure_filename(file.filename)
#         file.save(os.path.join(dcm_path, file_name))
#         conn.upload('bucket', '{}/dcm/{}'.format(case_id, file_name),
#                     os.path.join(dcm_path, file_name))
#         return {
#             "code": 0,
#             "data": "ok",
#             "msg": "ok"
#         }
#     return {
#         "code": 10001,
#         "data": "upload Error",
#         "msg": "上传失败"
#     }


def exec(case_id, path, dcm_path, model_path, bmp_path, cut_bmp_path, nii_path, nii_name, nii_obj, obj_path, obj_name):
    # try:
    if True:
        filepath = os.getcwd()
        model_test(dcm_path, model_path, bmp_path, cut_bmp_path)
        # bmp_to_nii(cut_bmp_path, nii_path, nii_name)
        # obj_make(nii_obj, obj_path, obj_name)
        organ = path
        # liver_min_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/min_bmp/'.format(case_id, 'liver'))
        # liver_max_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/bmp/'.format(case_id, 'liver'))
        min_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/min_bmp/'.format(case_id, organ))
        max_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/bmp/'.format(case_id, organ))
        min_list = os.listdir(max_bmp)
        min_list.sort(key=lambda x: int(x[:-4]))
        IMG_SIZE = 300
        print('bbbbbbbbbb')
        for x in min_list:
            name = os.path.join(max_bmp, x)
            im = cv2.imread(name, cv2.IMREAD_COLOR)
            new_array = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
            img_name = os.path.join(min_bmp, x)
            cv2.imwrite(img_name, new_array)
    # except:
    #     i = 1


@main.route('/execute', methods=['POST'])
def execute():
    """
    This is the execute API
        Call this 处理dcm api passing dcm and get back its features
        ---
        tags:
          - Liver Imaging Platform API READY
        parameters:
          - name: body
            in: body
            required: true
            properties:
              case_id:
                type: string
                required: true
                description: The patient case_id
              flag:
                type: integer
                required: true
                description: 0表示重试，1表示正常执行
        responses:
          500:
            description: Error
          200:
            description: The patient execute information
            schema:
              id: execute
              properties:
                code:
                  type: integer
                data:
                  type: string
                msg:
                  type: string

    """
    data = request.get_json()
    try:
        case_id = data['case_id']
    except:
        return {
            "code": 10003,
            "data": "",
            "msg": "未输入身高体重"
        }
    flag = data.get('flag')
    filepath = os.getcwd()
    if flag:
        # 正常执行状态
        dcm_path = os.path.join(filepath + app.config['STATIC'], '{}/dcm'.format(case_id))
    else:
        dcm_path = os.path.join(filepath + '/data/bucket', '{}/dcm'.format(case_id))
    paths = ['liver', 'tumor', 'spleen']
    if not os.path.exists(dcm_path):
        session[case_id]['volume'] = 2
        return {
            "code": 10002,
            "data": "",
            "msg": "未上传dcm文件"
        }
    process = []
    for path in paths:
        # 图像分割
        # device = torch.device('cuda:0')
        # gpu_memory = get_avaliable_memory(device)
        # while gpu_memory < 1720:
        #     print(gpu_memory)
        #     time.sleep(7)
        #     gpu_memory = get_avaliable_memory(device)
        if not os.path.exists(os.path.join(filepath + app.config['STATIC'], case_id)):
            os.mkdir(os.path.join(filepath + app.config['STATIC'], case_id))
        if not os.path.exists(os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), path)):
            os.mkdir(os.path.join(os.path.join(filepath + app.config['STATIC'], case_id), path))
            os.mkdir(os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp'.format(case_id, path)))
            os.mkdir(os.path.join(filepath + app.config['STATIC'], '{}/{}/bmp'.format(case_id, path)))
            os.mkdir(os.path.join(filepath + app.config['STATIC'], '{}/{}/min_bmp'.format(case_id, path)))
            os.mkdir(os.path.join(filepath + app.config['STATIC'], '{}/{}/nii'.format(case_id, path)))
            os.mkdir(os.path.join(filepath + app.config['STATIC'], '{}/{}/obj'.format(case_id, path)))
        bmp_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/bmp'.format(case_id, path))
        model_path = os.path.join(filepath + app.config['STATIC'], 'ckp/' + path + '.pth')
        nii_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/nii'.format(case_id, path))
        cut_bmp_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp/'.format(case_id, path))
        nii_name = path + '.nii'
        obj_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/obj'.format(case_id, path))
        nii_obj = os.path.join(nii_path, nii_name)
        obj_name = path + '.obj'
        process_ = Process(target=exec, args=(case_id, path, dcm_path, model_path, bmp_path, cut_bmp_path, nii_path, nii_name, nii_obj, obj_path,obj_name))
        process.append(process_)
        device = torch.device('cuda:0')
        gpu_memory = get_avaliable_memory(device)
        while gpu_memory < 1730:
            print(gpu_memory)
            time.sleep(14.5)
            gpu_memory = get_avaliable_memory(device)

        process_.start()
        process_.join()
        # print('aaaaaaaaaaaaaaaaaaaaaaaaa')
    # for i in range(len(paths)):
    #     process[i].join()

    # 体积计算
    liver_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp/'.format(case_id, 'liver'))
    tumor_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp/'.format(case_id, 'tumor'))
    spleen_bmp = os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp/'.format(case_id, 'spleen'))
    liver_volume, liver_compare_spleen, spleen_volume, tumor_volume = liver_tumor_spleen(dcm_path, liver_bmp,
                                                                                         tumor_bmp,
                                                                                         spleen_bmp)
    # dcm_list = os.listdir(dcm_path)
    # re_pattern = re.compile(r'(\d+)')
    # dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[0]))
    # dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[1]))
    # # dcm_list.sort(key=lambda dcm_list: int(dcm_list[:-4]))
    # for index, file_name in enumerate(dcm_list):
    #     dicom = pydicom.read_file(join(dcm_path, file_name))
    #     pixel_array, dicom.Rows, dicom.Columns = get_pixeldata(dicom)
    #     img_data = pixel_array
    #     winwidth = 500
    #     wincenter = 50
    #     rows = dicom.Rows
    #     cols = dicom.Columns
    #     dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)
    #     # dcm_img = cv2.imshow('imgshow', dcm_temp)
    #
    #     dcm_img = Image.fromarray(dcm_temp)
    #     dcm_img = dcm_img.convert('L')
    #     # dcm_img.show()
    #     save_path = os.path.join(filepath + app.config['STATIC'], '{}/dcmtopng/'.format(case_id))
    #     save_file_path = save_path + '{}.png'.format(index)
    #     dcm_img.save(save_file_path)
    #     conn.upload('bucket', '{}/dcmtopng/{}.png'.format(case_id, index),
    #                 save_file_path)
    #     conn.upload('bucket', '{}/dcm/{}.dcm'.format(case_id, index),
    #                 join(dcm_path, file_name))
    # 数据上传
    files_path = os.path.join(filepath + app.config['STATIC'], '{}'.format(case_id))
   # for path in paths:
   #     bmp_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/bmp/'.format(case_id, path))
   #     cut_bmp_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/cut_bmp/'.format(case_id, path))
   #     bmp_list = os.listdir(bmp_path)
   #     bmp_list.sort(key=lambda bmp_list: int(bmp_list[:-4]))
   #     cut_bmp_list = os.listdir(cut_bmp_path)
   #     cut_bmp_list.sort(key=lambda bmp_list: int(bmp_list[:-4]))
   #     min_bmp_path = os.path.join(filepath + app.config['STATIC'], '{}/{}/min_bmp'.format(case_id, path))
   #     min_bmp_list = os.listdir(min_bmp_path)
   #     min_bmp_list.sort(key=lambda min: int(min[:-4]))

   #     for file_name in min_bmp_list:
   #         bmp_upload_path = '{}/{}/min_bmp/{}'.format(case_id, path, file_name)
   #         # conn.upload('bucket', bmp_upload_path, os.path.join(min_bmp_path, file_name))
   #     for file_name in cut_bmp_list:
   #         bmp_upload_path = '{}/{}/bmp/{}'.format(case_id, path, file_name)
   #         # conn.upload('bucket', bmp_upload_path, os.path.join(bmp_path, file_name))
   #     for file_name in cut_bmp_list:
   #         cut_bmp_upload_path = '{}/{}/cut_bmp/{}'.format(case_id, path, file_name)
            # conn.upload('bucket', cut_bmp_upload_path, os.path.join(cut_bmp_path, file_name))
        # nii_upload_path = '{}/{}/nii/{}.nii'.format(case_id, path, path)
        # conn.upload('bucket', nii_upload_path, os.path.join(files_path, '{}/nii/{}.nii'.format(path, path)))
        # obj_upload_path = '{}/{}/obj/{}.obj'.format(case_id, path, path)
        # conn.upload('bucket', obj_upload_path, os.path.join(files_path, '{}/obj/{}.obj'.format(path, path)))
    # delete_file(os.path.join(filepath + app.config['STATIC'], case_id))
    case = Case.query.filter_by(case_id=case_id).first()
    sex, age = getInformation(case.identify_card)
    if sex:
        body_surface_area = 0.0057 * float(case.height) + 0.0121 * float(case.weight) + 0.0882
    else:
        body_surface_area = 0.0073 * float(case.height) + 0.0127 * float(case.weight) - 0.2106
    case.body_surface_area = body_surface_area
    case.real_liver_volume = liver_volume
    case.real_spleen_volume = spleen_volume
    case.liver_spleen_ratio = liver_compare_spleen
    case.standard_liver_volume = liver_volume / body_surface_area
    case.tumor_volume = tumor_volume

    try:
        db.session.commit()
        flash('User updated !')
    except Exception as e:
        print(repr(e))
        db.session.rollback()
        msg = 'User updated failed'
        return {"code": 10001, "data": "", "msg": msg}
    # while session[case_id]['volume']!= 1:
    #    time.sleep(10)
    return {"code": 0, "data": "", "msg": "ok"}


@main.route('/download/<string:case_id>', methods=['GET'])
def download(case_id):
    """
    This is download API
        Call this 下载分析报告 api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API
        parameters:
          - name: case_id
            in: path
            type: string
            required: true
            description: The patient case_id
        responses:
          500:
            description: Error
          200:
            description: The patient's report !
            schema:
              id: download_url
              properties:
                code:
                  type: integer
                data:
                  type: file
                  description: download_zip

    """
    file_path = os.getcwd()
    print(file_path)
    save = file_path + '/data/bucket/{}/dcm/'.format(case_id)
    print(save)
    if not os.path.exists(save):
        return {
            "code": 10001,
            "data": "未上传dcm文件"
        }
    filename = os.listdir(save)
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for _file in filename:
            with open(os.path.join(save, _file), 'rb') as fp:
                zf.writestr(_file, fp.read())
    memory_file.seek(0)
    return send_file(memory_file, attachment_filename='liver.zip', as_attachment=True)


@main.route('/downloadImage/<string:case_id>', methods=['GET'])
def downloadImage(case_id):
    """
    This is download API
        Call this 下载分析报告 api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API
        parameters:
          - name: case_id
            in: path
            type: string
            required: true
            description: The patient case_id
        responses:
          500:
            description: Error
          200:
            description: The patient's report !
            schema:
              id: download_url
              properties:
                code:
                  type: integer
                data:
                  type: file
                  description: download_zip

    """
    file_path = os.getcwd()
    print(file_path)

    save_path = file_path + '/123.jpg'
    down = conn.download('image', key='compress/' + case_id, save_path=save_path)

    # save = file_path + '/data/bucket/{}/dcm/'.format(case_id)
    # print(save)
    # if not os.path.exists(save):
    #     return {
    #         "code": 10001,
    #         "data": "未上传dcm文件"
    #     }
    # filename = os.listdir(save)

    # _tmp_file = tempfile.TemporaryFile()
    # a = down['Body']
    # _tmp_file.write(down['Body'].read())
    # memory_file = BytesIO()
    # with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
    # for _file in filename:
    #     with open(os.path.join(save, _file), 'rb') as fp:
    #         zf.writestr(_file, fp.read())
    # memory_file.seek(0)
    a = send_file(save_path, conditional=True, attachment_filename='.jpg'.format(case_id), as_attachment=True)
    return send_file(save_path, conditional=True, attachment_filename='{}.jpg'.format(case_id), as_attachment=True)


def delete_file(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return "delete ok"
    else:
        return "no filepath"


def dcmToPng(path, save_path):
    dicom = pydicom.read_file(path)
    pixel_array, dicom.Rows, dicom.Columns = get_pixeldata(dicom)
    img_data = pixel_array
    winwidth = 500
    wincenter = 50
    rows = dicom.Rows
    cols = dicom.Columns
    dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)
    # dcm_img = cv2.imshow('imgshow', dcm_temp)
    dcm_img = Image.fromarray(dcm_temp)
    dcm_img = dcm_img.convert('L')
    # dcm_img.show()
    dcm_img.save(save_path)


@main.route('/smoothDegreeImage', methods=['GET'])
def smoothDegreeImage():
    """
    This is download API
        Call this smoothDegreeImage api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API

        parameters:
          - name: num
            in: query
            type: int
            required: true
            description: The image num
          - name: case_id
            in: query
            type: string
            required: true
        responses:
          500:
            description: Error
          200:
            description: The patient's report !
            schema:
              id: download_url
              properties:
                code:
                  type: integer
                data:
                  type: string
                  description: image path
    """
    num = int(request.args.get('num'))-1
    case_id = request.args.get('case_id')
    filepath = os.getcwd()
    dcm_path = os.path.join(filepath + app.config['STATIC'], '{}/dcm'.format(case_id))
    dcm_list = os.listdir(dcm_path)
    re_pattern = re.compile(r'(\d+)')
    dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[0]))
    dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[1]))
    # dcm_list.sort(key=lambda dcm_list: int(dcm_list[:-4]))
    if dcm_list[num] is not None:
    # for index, file_name in enumerate(dcm_list):
        index = num
        file_name = dcm_list[num]
        dicom = pydicom.read_file(join(dcm_path, file_name))
        pixel_array, dicom.Rows, dicom.Columns = get_pixeldata(dicom)
        img_data = pixel_array
        winwidth = 500
        wincenter = 50
        rows = dicom.Rows
        cols = dicom.Columns
        dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)
        # dcm_img = cv2.imshow('imgshow', dcm_temp)
        dcm_img = Image.fromarray(dcm_temp)
        dcm_img = dcm_img.convert('L')
        # dcm_img.show()
        save_path = os.path.join(filepath + app.config['STATIC'], '{}/dcmtopng/'.format(case_id))
        save_file_path = save_path + '{}.png'.format(index)
        dcm_img.save(save_file_path)
        # conn.upload('bucket', '{}/dcmtopng/{}.png'.format(case_id, index),
        #             save_file_path)
        # conn.upload('bucket', '{}/dcm/{}.dcm'.format(case_id, index),
        #             join(dcm_path, file_name))

    path = '{}/dcmtopng/{}.png'.format(case_id, num)
    return {
        "code": 0,
        "data": path,
    }


def smoothDegree(dcm, label, location):
    a = pydicom.read_file(dcm)

    pass


@main.route('/smoothDegreeExec', methods=['POST'])
def smoothDegreeExec():
    """
    This is download API
        Call this smoothDegreeImage api passing case_id and get back its features
        ---
        tags:
          - Liver Imaging Platform API
        parameters:
          - name: body
            in: body
            required: true
            properties:
              case_id:
                type: string
                required: true
                description: The patient case_id
              num:
                type: integer
                required: true
                description: image number
              location:
                type: array
                items:
                  type: number
                  required: true
                  description: image cut location
        responses:
          500:
            description: Error
          200:
            description: The patient's report !
            schema:
              id: smooth results
              properties:
                code:
                  type: integer
                  description: 0 success 10001
                data:
                  type: string
                  description: smooth results
    """
    data = request.get_json()
    case_id = data['case_id']
    num = int(data['num'])-1
    location = data['location']
    axis = {}
    img_axis = location[0]
    crop_axis = location[1]
    x = (img_axis['x2'] - img_axis['x1'])
    y = (img_axis['y2'] - img_axis['y1'])
    axis['x1'] = (crop_axis['x1'] - img_axis['x1']) / x * 512
    axis['x2'] = (crop_axis['x2'] - img_axis['x1']) / x * 512
    axis['y1'] = (crop_axis['y1'] - img_axis['y1']) / y * 512
    axis['y2'] = (crop_axis['y2'] - img_axis['y1']) / y * 512
    # dcm_path = '{}/dcm/{}.dcm'.format(case_id, num)

    filepath = os.getcwd()
    print(filepath)
    dcm_path = os.path.join(filepath + app.config['STATIC'], '{}/dcm'.format(case_id))
    label_path = os.path.join(filepath + app.config['STATIC'], '{}/liver/cut_bmp/{}.bmp'.format(case_id, num))
    print(label_path)
    label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)

    dcm_list = os.listdir(dcm_path)
    re_pattern = re.compile(r'(\d+)')
    dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[0]))
    dcm_list.sort(key=lambda x: int(re_pattern.findall(x)[1]))
    file_name = dcm_list[num]
    dicom = pydicom.read_file(join(dcm_path, file_name))


    # dcm_data = conn.download('bucket', dcm_path)
    # label_data = conn.download('bucket', label_path)
    # dcm = dcm_data['Body'].read()
    # dicom = ''
    # _tmp_file = tempfile.TemporaryFile()  # 查看该类发现是以：w+b方式写入文件的，该模式支持热读写，并且要求写入格式是二进制
    # try:
    #     print(_tmp_file.name)
    #     # _tmp_file.write(b"something\n")
    #     _tmp_file.write(dcm)  # 需要写入二进制
    #     _tmp_file.seek(0)
    #     # 读取并销毁临时文件
    #     dicom = pydicom.read_file(_tmp_file)
    # finally:
    #     _tmp_file.close()
    pixel_array, dicom.Rows, dicom.Columns = get_pixeldata(dicom)
    img_data = pixel_array
    winwidth = 500
    wincenter = 50
    rows = dicom.Rows
    cols = dicom.Columns
    dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)

    # dcm_img = cv2.imshow('imgshow', dcm_temp)
    # dcm_img = cv2.fromarray(dcm_temp)
    # cv2.
    # dcm_img = dcm_temp.convert('L')

    # label = label_data['Body'].read()
    #
    # label_file = cv2.imdecode(np.frombuffer(label, np.uint8), 0)

    # label_file = ''
    # _tmp_file1 = tempfile.TemporaryFile()  # 查看该类发现是以：w+b方式写入文件的，该模式支持热读写，并且要求写入格式是二进制
    # try:
    #     print(_tmp_file1.name)
    #     _tmp_file.write(b"something\n")
    # _tmp_file1.write(label)  # 需要写入二进制
    # _tmp_file1.seek(1)
    # 读取并销毁临时文件
    # label_file = cv2.imread(_tmp_file1, 0)
    # finally:
    #     _tmp_file.close()
    # try:
    #     smooth_data = smooth_(dcm_temp, label_file, round(axis['x1']), round(axis['y1']), round(axis['x2']), round(axis['y2']))
    # except:
    #     return {
    #         "code": 10001,
    #         "data": '',
    #     }
    smooth_data = smooth(dcm_temp, label, round(axis['x1']), round(axis['y1']), round(axis['x2']), round(axis['y2']))
    print(smooth_data)
    return {
        "code": 0,
        "data": str(smooth_data),
    }



@main.route('/logout')
def logout():
    """
    登出，注销用户
    :return:
    """
    session.pop('user', None)
    return redirect(url_for('main.login'))


@main.route("/log", methods=["GET"])
def log():
    """
    日志系统
    :return:
    """
    current_app.logger.info("this is info")
    current_app.logger.debug("this is debug")
    current_app.logger.warning("this is warning")
    current_app.logger.error("this is error")
    current_app.logger.critical("this is critical")
    return "ok"
