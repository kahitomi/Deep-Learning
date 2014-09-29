# -*- coding: utf-8 -*-
# import shelve

# def saveData(data, name = 'data'):
# 	db = shelve.open(name, 'c')
# 	db['data'] = data
# 	db.close()
# 	print 'save success: '+name
          
# def loadData(filename):  
# 	db = shelve.open(filename, 'r')
# 	for item in db.items():  
# 		temp = item
# 	# temp = db.items()['data']
# 	db.close()
# 	return temp

import pickle

def saveData(data, name = 'data'):
	f = open(name,"wb")
	pickle.dump(data, f)
	f.close()
	print 'save success: '+name
          
def loadData(filename):  
	f = open(filename,"rb")
	temp = pickle.load(f)
	f.close()
	return temp