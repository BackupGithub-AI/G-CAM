import math, os, sys
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import numpy as np
import torch.nn.functional as F
DEBUG_UTIL = False

def gen_A(p, num_classes, t, adj_file):
	import pickle
	with open(adj_file, "rb") as f:
		result = pickle.load(f)
	_adj = result['adj']
	_nums = result['nums']
	_nums = _nums[:, np.newaxis]  
	_adj = _adj / _nums
	_adj[_adj < t] = 0  
	_adj[_adj >= t] = 1
	_adj = _adj * p / (_adj.sum(0, keepdims=True) + 1e-6)
	_adj = _adj + np.identity(num_classes, np.int)
	return _adj


def gen_adj(A):
	D = torch.pow(A.sum(1).float(), -0.5)  
	D = torch.diag(D)
	adj = torch.matmul(torch.matmul(A, D).t(), D)  
	return adj
