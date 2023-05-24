# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:06:44 2022

@author: 97503
"""
import os
import logging

path = "./result"
#判断是否存在文件夹如果不存在则创建为文件夹
if not os.path.exists(path):  
    os.makedirs(path)


# 第一步，创建一个logger
myLogger = logging.getLogger("myLogger")
myLogger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

# 第二步，创建一个handler，用于写入日志文件
logfile = os.path.join(path,'log.txt')
fileLogging = logging.FileHandler(logfile)  # open的打开模式这里可以进行参考
fileLogging.setLevel(logging.INFO)  # 输出到file的log等级的开关

# 第三步，再创建一个handler，用于输出到控制台
controlLogging = logging.StreamHandler()
controlLogging.setLevel(logging.DEBUG)   # 输出到console的log等级的开关


# 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: \n%(message)s")
fileLogging.setFormatter(formatter)
controlLogging.setFormatter(formatter)

# 第五步，创建一个handler，将result信息输出到文件
# warning类型的信息作为result信息
resultsfile = os.path.join(path,'results.txt')
resultLogging = logging.FileHandler(resultsfile, encoding= "utf-8")
resultLogging.setLevel(logging.INFO)
resultfilter = logging.Filter()
resultfilter.filter = lambda record: record.levelno == logging.WARNING
resultLogging.addFilter(resultfilter)

# 第六步，将logger添加到handler里面
myLogger.addHandler(fileLogging)
myLogger.addHandler(controlLogging)
myLogger.addHandler(resultLogging)


if __name__ == '__main__':
    # 日志级别
    myLogger.debug('这是 logger debug message')
    myLogger.info('这是 logger info message')
    myLogger.warning('这是 logger warning message')
    myLogger.error('这是 logger error message')
    myLogger.critical('这是 logger critical message')
