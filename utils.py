# -*- coding=utf-8 -*-
######################################################################
# utils.py文件主要描述了获取训练集和测试集，学习率，以及多标签分类函数的计算过程，
# 同时封装了对attention heatmaps进行翻转过程，为后面的attention consistency
# loss做准备。整体来看，这部分可以先不用管。
######################################################################
import torch
from wider import get_subsets
from voc2007 import get_voc2007
from coco import get_coco
from mirflickr25k import get_mirflickr25k
from nuswide import get_nuswide
from torch.autograd import Variable


def get_dataset(opts):
	if opts["dataset"] == "WIDER":
		data_dir = opts["data_dir"]
		anno_dir = opts["anno_dir"]
		trainset, testset = get_subsets(anno_dir, data_dir)
	elif opts['dataset'] == 'VOC2007':
		voc_inp_dir = opts['voc_inp_dir']
		data_dir = opts['data_dir']
		trainset, testset = get_voc2007(voc_inp_dir, data_dir)
	elif opts['dataset'] == 'COCO':
		coco_inp_dir = opts['coco_inp_dir']
		data_dir = opts['data_dir']
		trainset, testset = get_coco(coco_inp_dir, data_dir)
	elif opts['dataset'] == 'MIRFLICKR25K':
		mirflickr25k_inp_dir = opts['mirflickr25k_inp_dir']
		data_dir = opts['data_dir']
		trainset, testset = get_mirflickr25k(mirflickr25k_inp_dir, data_dir)
	elif opts['dataset'] == 'NUSWIDE':
		nuswide_inp_dir = opts['nuswide_inp_dir']
		data_dir = opts['data_dir']
		trainset, testset = get_nuswide(nuswide_inp_dir, data_dir)
	
	else: pass

	return trainset, testset


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed every 30 epochs"""
	lr = args.learning_rate * (args.decay ** (epoch // args.stepsize))
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
#
#
# def SigmoidCrossEntropyLoss(x, y, w_p, w_n):
# 	# weighted sigmoid cross entropy loss defined in Li et al. ACPR'15
# 	loss = 0.0
# 	if not x.size() == y.size():
# 		print("x and y must have the same size")
# 	else:
# 		N = y.size(0)
# 		L = y.size(1)
# 		for i in range(N):
# 			w = torch.zeros(L).cuda()
# 			w[y[i].data == 1] = w_p[y[i].data == 1]
# 			w[y[i].data == 0] = w_n[y[i].data == 0]
#
# 			w = Variable(w, requires_grad = False)
# 			temp = - w * ( y[i] * (1 / (1 + (-x[i]).exp())).log() + \
# 				(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
# 			loss += temp.sum()
#
# 		loss = loss / N
# 	return loss


def SigmoidCrossEntropyLoss(x, y, w_p, w_n):
	# weighted sigmoid cross entropy loss defined in Li et al. ACPR'15
	loss = 0.0
	if not x.size() == y.size():
		print("x and y must have the same size")
	else:
		N = y.size(0)
		L = y.size(1)
		for i in range(N):
			# w = torch.zeros(L).cuda()
			# w[y[i].data == 1] = w_p[y[i].data == 1]
			# w[y[i].data == 0] = w_n[y[i].data == 0]
			#
			# w = Variable(w, requires_grad = False)
			temp = -1.0 * ( y[i] * (1 / (1 + (-x[i]).exp())).log() +
							(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
			loss += temp.sum()

		loss = loss / N
	return loss



def generate_flip_grid(w, h):
	# used to flip attention maps
	x_ = torch.arange(w).view(1, -1).expand(h, -1)		# w columns, h lines
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)		# h lines, w columns
	# print('x_=\n{0}, y_=\n{1},\nx_.shape={2}, y_.shape={3}'.format(x_, y_, x_.shape, y_.shape))
	# x_ and y_ stack along with the dim-0 axis, grid.shape = [2, h, w]
	grid = torch.stack([x_, y_], dim=0).float().cuda() if torch.cuda.is_available() \
		else torch.stack([x_, y_], dim=0).float()
	# print('(0) grid = \n{0},\ngrid.shape={1}'.format(grid, grid.shape))
	grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
	# print('(1) grid = \n{0},\ngrid.shape={1}'.format(grid, grid.shape))
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	# print('(2) grid = \n{0},\ngrid.shape={1}'.format(grid, grid.shape))
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
	# print('(3) grid = \n{0},\ngrid.shape={1}'.format(grid, grid.shape))

	grid[:, 0, :, :] = -grid[:, 0, :, :]
	# print('(4) grid = \n{0},\ngrid.shape={1}'.format(grid, grid.shape))
	return grid


if __name__ == "__main__":
	grid_l = generate_flip_grid(7,7)
	grid_x = grid_l.expand(2, -1, -1, -1)
	print('\n', grid_x, '\n', grid_x.shape)
	grid_y = grid_x.permute([0,2,3,1])
	print('\n', grid_y, '\n',grid_y.shape)
	
