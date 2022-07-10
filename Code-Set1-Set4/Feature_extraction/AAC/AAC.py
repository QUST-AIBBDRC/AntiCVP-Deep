# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:43:00 2019

@author: 85010
"""
import re,os,sys,csv
from collections import Counter
pPath = re.sub(r'AAC$', '', os.path.split(os.path.realpath(__file__))[0])
sys.path.append(pPath)
import readFasta
import pandas as pd

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

fastas = readFasta.readFasta("test.Set1.biaohao.txt")###输入样本数据
kw=  {'path': "D:\学习\生信\生育力_identifying fertility-related proteins\AAC",'data':"Independent_Non.txt",'order':'ACDEFGHIKLMNPQRSTVWY'}
#修改path和data即可
data_AAC=AAC(fastas, **kw)
AAC=pd.DataFrame(data=data_AAC)
AAC.to_csv('AAC_test_Set_1_biaohao.csv')#保存