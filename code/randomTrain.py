#-*- coding:utf-8 -*-
import sys,csv
import random

if __name__ == "__main__":
	train = open('./train_hash.csv','r').readlines()
	print len(train)
	random.shuffle(train)
	with open('./train_hash.csv','w') as f:
		for item in train:
			f.write(item)
