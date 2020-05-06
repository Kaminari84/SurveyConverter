#!/usr/bin/python3
import sys

#sys.path.insert(0,"/var/www/")
#sys.path.insert(0,"/var/www/testapp")


path = '/var/www/'
if path not in sys.path: sys.path.append(path)

path = '/var/www/testapp' 
if path not in sys.path: sys.path.append(path)


from testapp import app as application
