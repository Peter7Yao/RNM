import numpy as np


def loadfile(fn, num=1):
	print('loading a file...' + fn)
	ret = []
	with open(fn, encoding='utf-8') as f:
		for line in f:
			th = line[:-1].split('\t')
			x = []
			for i in range(num):
				x.append(int(th[i]))
			ret.append(tuple(x))
	return ret
