# # -*- coding: utf-8 -*-
# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField, PasswordField, SelectField, BooleanField, ValidationError, TextField, \
#     TextAreaField
# from wtforms.validators import DataRequired, Length, EqualTo
#
#
# class PatientForm(FlaskForm):
#     name = StringField('name', validators=[DataRequired(message=u"姓名不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入姓名'})
#     # 密码
#     identify_card = StringField('identify_card', validators=[DataRequired(message=u"身份证号不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入身份证号'})
#     submit = SubmitField('submit')
#
#
# class CaseForm(FlaskForm):
#     height = StringField('height', validators=[DataRequired(message=u"身高不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入身高'})
#     # 密码
#     weight = StringField('weight', validators=[DataRequired(message=u"体重不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入体重'})
#     submit = SubmitField('submit')
#
#
# class LoginForm(FlaskForm):
#     username = StringField('username', validators=[DataRequired(message=u"用户名不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入用户名'})
#     # 密码
#     password = PasswordField('password', validators=[DataRequired(message=u"密码不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入密码'})
#     submit1 = SubmitField('submit')
#
#
# class Sign_inForm(FlaskForm):
#     username = StringField('username', validators=[DataRequired(message=u"用户名不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入用户名'})
#     name = StringField('name', validators=[DataRequired(message=u"姓名不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入姓名'})
#     # 密码
#     password = PasswordField('password', validators=[DataRequired(message=u"密码不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入密码'})
#     # department = SelectField('de_select')
#     department = StringField('department', validators=[DataRequired(message=u"姓名不能为空")
#         , Length(max=255)], render_kw={'placeholder': u'输入科室'})
#     # pre_phonenumber = SelectField('pre_select')
#     phone_number = StringField('phone_number', validators=[DataRequired(message=u"手机号不能为空")
#         , Length(max=254)], render_kw={'placeholder': u'输入手机号'})
#     get_check = SubmitField('check_submit')
#     sign_submit = SubmitField('sign_submit')
#
#
# class BaseLogin(FlaskForm):
#     # 用户名
#     name = StringField('name', validators=[DataRequired(message=u"用户名不能为空")
#         , Length(10, 20, message=u'长度位于10~20之间')], render_kw={'placeholder': u'输入用户名'})
#     # 密码
#     password = PasswordField('password', validators=[DataRequired(message=u"密码不能为空")
#         , Length(10, 20, message=u'长度位于10~20之间')], render_kw={'placeholder': u'输入密码'})
