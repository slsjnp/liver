import gevent.monkey
gevent.monkey.patch_all()
import multiprocessing

debug = False
loglevel = 'debug'
bind = '0.0.0.0:5000'
pidfile = 'gun_log/gunicorn.pid'
logfile = 'gun_log/debug.log'

# 启动的进程数
workers = multiprocessing.cpu_count()
worker_class = 'gunicorn.workers.ggevent.GeventWorker'
chdir = '/workspace/liver_back'
x_forwarded_for_header = 'X-FORWARDED-FOR'
