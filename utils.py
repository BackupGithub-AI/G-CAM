# -*- coding=utf-8 -*-
import torch
from wider import get_subsets
from mirflickr25k import get_mirflickr25k
from torch.autograd import Variable


def get_dataset(opts):
	if opts['dataset'] == 'MIRFLICKR25K':
		mirflickr25k_inp_dir = opts['mirflickr25k_inp_dir']
		data_dir = opts['data_dir']
		trainset, testset = get_mirflickr25k(mirflickr25k_inp_dir, data_dir)
	elif opts['dataset'] == 'COCO':
		pass
		# will be added later
	elif opts['dataset'] == 'NUSWIDE':
		pass
		# will be added later
	
	else: pass

	return trainset, testset


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed every 30 epochs"""
	lr = args.learning_rate * (args.decay ** (epoch // args.stepsize))
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr



def SigmoidCrossEntropyLoss(x, y, w_p, w_n):
	loss = 0.0
	if not x.size() == y.size():
		print("x and y must have the same size")
	else:
		N = y.size(0)
		L = y.size(1)
		for i in range(N):
			temp = -1.0 * ( y[i] * (1 / (1 + (-x[i]).exp())).log() +
							(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
			loss += temp.sum()

		loss = loss / N
	return loss



def generate_flip_grid(w, h):
	x_ = torch.arange(w).view(1, -1).expand(h, -1)		
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)		
	grid = torch.stack([x_, y_], dim=0).float().cuda() if torch.cuda.is_available() \
		else torch.stack([x_, y_], dim=0).float()
	grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

	grid[:, 0, :, :] = -grid[:, 0, :, :]
	return grid

	
