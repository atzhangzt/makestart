"""
This script runs the FlaskDeepServer application using a development server.
"""

from os import environ
from FlaskDeepServer import app
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='log.log',
                filemode='w')
    app.run('0.0.0.0', 8888)