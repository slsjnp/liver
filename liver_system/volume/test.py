import os
import re
path = os.getcwd()
filename = path + '/liver_system/domo/app/static/han'
filename_a = path + '/liver_system/domo/app/static/dcm1'
filenames = os.listdir(filename)

for i in filenames:
    with open(filename + '//' + i, 'rb') as o:
        with open(filename_a + '//' + i + '.dcm', 'bw') as w:
            w.write(o.read())
