# -*- coding=utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm, trange
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from test import test
from configs import get_configs
from utils import get_dataset, adjust_learning_rate, SigmoidCrossEntropyLoss, \
	generate_flip_grid

import matplotlib.pyplot as plt
import numpy as np

import sys
import argparse
import math
import time, datetime
import os

def get_parser():
	parser = argparse.ArgumentParser(description = 'CNN Attention Consistency')
	parser.add_argument("--dataset", default="wider", type=str,
		help="select a dataset to train models")
	parser.add_argument("--arch", default="resnet50", type=str,
		help="ResNet architecture")

	parser.add_argument('--train_batch_size', default = 16, type = int,
		help = 'default training batch size')
	parser.add_argument('--train_workers', default = 4, type = int,
		help = '# of workers used to load training samples')
	parser.add_argument('--test_batch_size', default = 8, type = int,
		help = 'default test batch size')
	parser.add_argument('--test_workers', default = 4, type = int,
		help = '# of workers used to load testing samples')

	parser.add_argument('--learning_rate', default = 0.001, type = float,
		help = 'base learning rate')
	parser.add_argument('--momentum', default = 0.9, type = float,
		help = "set the momentum")
	parser.add_argument('--weight_decay', default = 0.0005, type = float,
		help = 'set the weight_decay')
	parser.add_argument('--stepsize', default = 3, type = int,
		help = 'lr decay each # of epoches')
	parser.add_argument('--decay', default=0.5, type=float,
		help = 'update learning rate by a factor')

	parser.add_argument('--model_dir',
		default = './checkpoint',
		type = str,
		help = 'path to save checkpoints')
	parser.add_argument('--model_prefix',
		default = 'model',
		type = str,
		help = 'model file name starts with')

	# optimizer
	parser.add_argument('--optimizer',
		default = 'SGD',
		type = str,
		help = 'Select an optimizer: TBD')

	# general parameters
	parser.add_argument('--epoch_max', default = 12, type = int,
		help = 'max # of epcoh')
	parser.add_argument('--display', default = 100, type = int,
		help = 'display')
	parser.add_argument('--snapshot', default = 1, type = int,
		help = 'snapshot')
	parser.add_argument('--start_epoch', default = 0, type = int,
		help = 'resume training from specified epoch')
	parser.add_argument('--resume', default = '', type = str,
		help = 'resume training from specified model state')
	parser.add_argument('--w2v_file', default = './data/voc/voc_word2vec.pkl', type = str,)
	parser.add_argument('--adj_file', default='./data/voc/voc_adj.pkl', type=str,)
	parser.add_argument("--p", type=float, default=0.15)  # the p value used in the adjecent matrix
	parser.add_argument("--tao", type=float, default=0.4) # as above
	parser.add_argument("--gcn_lr", type=float, default=0.1)  # as above
	parser.add_argument("--acc_steps", type=int, default=0)  # as above
	parser.add_argument("--lr_scale", type=int, default=10)  # as above
	parser.add_argument('-e', '--evaluate', action='store_true')

	parser.add_argument('--test', default = True, type = bool,
		help = 'conduct testing after each checkpoint being saved')

	return parser


def main():
	parser = get_parser()
	print(parser)
	args = parser.parse_args()
	print(args)
	arch_flag = False
	dataset_flag = False
	max_point_step = -1
	max_ap = -1
	ALL_RESULT = {}


	# load data
	opts = get_configs(args.dataset)
	print(opts)
	if args.dataset in ["wider", "pa-100k"]:
		pos_ratio = torch.FloatTensor(opts["pos_ratio"])
		w_p = (1 - pos_ratio).exp().cuda() if torch.cuda.is_available() else (1 - pos_ratio).exp()
		w_n = pos_ratio.exp().cuda() if torch.cuda.is_available() else pos_ratio.exp()
	else: w_p, w_n = 1.0, 1.0

	trainset, testset = get_dataset(opts)

	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size = args.train_batch_size,
		shuffle = True,
		num_workers = args.train_workers)
	test_loader = torch.utils.data.DataLoader(testset,
		batch_size = args.test_batch_size,
		shuffle = False,
		num_workers = args.test_workers)
	
	batch_amount = train_loader.__len__()


	# path to save models
	if not os.path.isdir(args.model_dir):
		print("Make directory: " + args.model_dir)
		os.makedirs(args.model_dir)

	# prefix of saved checkpoint
	model_prefix = args.model_dir + '/' + args.model_prefix + args.dataset


	# define the model: use ResNet50 as an example
	if args.arch == "resnet50":
		from resnet import resnet50
		model = resnet50(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_resnet50"
	elif args.arch == "resnet101":
		from resnet import resnet101
		model = resnet101(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_resnet101"
	elif args.arch == 'grn101':
		from models import grn101
		model = grn101(pretrained=True, num_labels=opts['num_labels'], adj_file=args.adj_file, tao=args.tao,
					   p = args.p)
		model_prefix = model_prefix + '_grn101'
		arch_flag = True
	else:
		raise NotImplementedError("To be implemented!")
		
	# print(model)
	if torch.cuda.is_available():    model.cuda()

	## resume with exiting `pretrained-model`
	if args.start_epoch != 0:
		resume_model = torch.load(args.resume)
		resume_dict = resume_model.state_dict()
		model_dict = model.state_dict()
		resume_dict = {k:v for k,v in resume_dict.items() if k in model_dict}
		model_dict.update(resume_dict)
		model.load_state_dict(model_dict)
	if args.evaluate:
		## only test once, use command '-e'
		if os.path.isfile(args.resume):
			resume_model = torch.load(args.resume)
			resume_dict = resume_model.state_dict()
			model_dict = model.state_dict()
			resume_dict = {k: v for k, v in resume_dict.items() if k in model_dict}
			model_dict.update(resume_dict)
			model.load_state_dict(model_dict)
			model.eval()
			test_start = time.clock()
			test_result = test(model, test_loader, 'evaluation', flag=arch_flag)
			test_time = (time.clock() - test_start)
			print("test time: ", test_time)
			return


	if args.arch=='resnet50' or args.arch=='resnet101':
		if args.optimizer == 'Adam':
			optimizer = optim.Adam(
				model.parameters(),
				lr = args.learning_rate
			)
		elif args.optimizer == 'SGD':
			optimizer = optim.SGD(
				model.parameters(),
				lr = args.learning_rate,
				momentum = args.momentum,
				weight_decay = args.weight_decay
			)
		else:
			raise NotImplementedError("For other datasets, this optimizer Not supported yet!")
	else:
		if args.optimizer == 'Adam':
			optimizer = optim.Adam(
				# model.parameters(),
				model.get_config_optim(args.learning_rate, scale=float(args.lr_scale)),
				lr = args.learning_rate
			)
		elif args.optimizer == 'SGD':
			optimizer = optim.SGD(
				# model.parameters(),
				model.get_config_optim(args.learning_rate, scale=float(args.lr_scale)),
				lr = args.learning_rate,
				momentum = args.momentum,
				weight_decay = args.weight_decay,
			)
		else:
			raise NotImplementedError("For our datasets, this optimizer Not supported yet!")

	# training the network
	model.train()

	# attention map size
	w1 = 7
	h1 = 7
	grid_l = generate_flip_grid(w1, h1)

	w2 = 6
	h2 = 6
	grid_s = generate_flip_grid(w2, h2)

	# least common multiple, 42
	lcm = w1 * w2

	criterion = SigmoidCrossEntropyLoss
	criterion_mse = nn.MSELoss(reduction='mean')
	# start training processing
	for epoch in range(args.start_epoch, args.epoch_max):
		backward_count = 0
		epoch_start = time.clock()
		print("\n({0})Training Processing...".format(int(epoch)))
		if not args.stepsize == 0:
			adjust_learning_rate(optimizer, epoch, args)
		train_loader = tqdm(train_loader, desc="({0})Training".format(int(epoch)))
		for step, batch_data in enumerate(train_loader):
			# every batch_data includes 5 items
			# index 0~1 are two 224*224 image matrix [batchsize, channel(3), 224,224], one origin, one flip
			# index 2~3 are two 192*192 image matrix, [batchsize, channel(3), 192, 192], one origin, one flip
			# index 4 is batchsize*num_classes(for wider attribute is 14)
			
			batch_images_lo = batch_data[0]
			batch_images_lf = batch_data[1]
			batch_images_so = batch_data[2]
			batch_images_sf = batch_data[3]
			batch_labels = batch_data[4]
			if args.dataset in  ["coco", 'voc2007', 'mirflickr25k', 'nuswide']:
				dataset_flag = True
				batch_inp = batch_data[5]

			batch_labels[batch_labels == -1] = 0	# if one label==-1, transfer 0

			# batch_images_l size is [2*batchsize, 3, 224, 224]
			batch_images_l = torch.cat((batch_images_lo, batch_images_lf))
			# batch_images_l size is [2*batchsize, 3, 192, 192]
			batch_images_s = torch.cat((batch_images_so, batch_images_sf))
			# four images, their labels are identical
			batch_labels = torch.cat((batch_labels, batch_labels, batch_labels, batch_labels))
			
			if torch.cuda.is_available():
				batch_images_l = batch_images_l.cuda()
				batch_images_s = batch_images_s.cuda()
				batch_labels = batch_labels.cuda()
				if arch_flag:
					inp_var = Variable(batch_inp).float().detach().cuda()

			inputs_l = Variable(batch_images_l)
			inputs_s = Variable(batch_images_s)
			labels = Variable(batch_labels)
			if arch_flag:
				inp_var = Variable(batch_inp).float().detach()
				with torch.no_grad():
					inp_var = torch.autograd.Variable(inp_var).float()

			# return [`output`, `heatmap`]
			# hm_l size is [2*batchsize, num_classes, 7, 7], hm_s size is [2*batchsize, num_classes, 6, 6]
			# out_l size is identical with output_s, equals [2*batchsize, num_classes]
			if arch_flag:
				output_l, hm_l = model(inputs_l, inp_var)
				output_s, hm_s = model(inputs_s, inp_var)
			else:
				output_l, hm_l = model(inputs_l)
				output_s, hm_s = model(inputs_s)
			
			# output size is [4*batchsize, num_classes]
			output = torch.cat((output_l, output_s))
			loss = criterion(output, labels, w_p, w_n)

			# flip
			num = hm_l.size(0) // 2

			## hm2.shape = [batchsize, num_classes, 7(or 6) , 7(or 6)]
			hm1, hm2 = hm_l.split(num)	# `batchsize` piece of imgs, `num_classes` piece of feature-maps
			flip_grid_large = grid_l.expand(num, -1, -1, -1)
			# flip_grid_large = Variable(flip_grid_large, requires_grad = False)
			flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
			# print("flip_grid_large=\n{0}\n, flip_grid_large.shape={1}".format(flip_grid_large,flip_grid_large.shape))
			hm2_flip = F.grid_sample(hm2, flip_grid_large, mode = 'bilinear',
				padding_mode = 'border')
			# print('type(hm2)={0}, hm2.shape={1}, hm2=\n{2}\n'.format(type(hm2), hm2.shape, hm2[0,0,:,:]))
			# print('type(hm2_flip)={0}, hm2_flip.shape={1}, hm2_flip=\n{2}\n'	\
			# 	  .format(type(hm2_flip), hm2_flip.shape, hm2_flip[0,0,:,:]))
			flip_loss_l = F.mse_loss(hm1, hm2_flip)

			hm1_small, hm2_small = hm_s.split(num)
			flip_grid_small = grid_s.expand(num, -1, -1, -1)
			# flip_grid_small = Variable(flip_grid_small, requires_grad = False)
			flip_grid_small = flip_grid_small.permute(0, 2, 3, 1)
			hm2_small_flip = F.grid_sample(hm2_small, flip_grid_small, mode = 'bilinear',
				padding_mode = 'border')
			flip_loss_s = F.mse_loss(hm1_small, hm2_small_flip)

			# scale loss
			num = hm_l.size(0)
			# two upsample process
			hm_l = F.interpolate(hm_l, lcm)		#F.upsample(hm_l, lcm)
			hm_s = F.interpolate(hm_s, lcm)		#F.upsample(hm_s, lcm)
			scale_loss = F.mse_loss(hm_l, hm_s)
			
			# calc the overall loss, incorporate four parts
			#losses = #loss + (flip_loss_l + flip_loss_s + scale_loss) if args.lr_scale  else \
		
			losses = 0.9 * loss + 0.1 * (flip_loss_l + flip_loss_s + scale_loss) \
				if args.arch == 'grn101' or args.arch=='grn50' else loss + flip_loss_l + flip_loss_s + scale_loss
			
			if args.acc_steps>0:
				losses /= args.acc_steps		# this will impact the losses, loss, flip_loss_l, flip_loss_s, scale_loss
												# the ouput losses multiply the acc_steps can meet the above `losses` formula
				losses.backward()
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
				if int(step + 1) % int(args.acc_steps) == 0 or int(step+1) == int(batch_amount) :
					backward_count += 1
					optimizer.step()
					optimizer.zero_grad()
			else:
				optimizer.zero_grad()		# set the all grad to zero
				losses.backward()			# use the `losses` scalar value for back propagation
				nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
				optimizer.step()			# update the all learning-parameters
			
			if False:
				print("Here print all the net parameters:\n")
				for name, params in model.named_parameters():
					if 'gc' in str(name):
						print("name:\n ", name)
						print('content:\n ', params)
						print("content-size: \n", params.shape)
			
			if (step) % args.display == 0:
				print(
					'epoch: {},\ttrain step: {}\tLoss: {:.6f}'.format(epoch+1,
					step, losses.item() )#losses.data[0])
				)
				print(
					'\tcls loss: {:.4f};\tflip_loss_l: {:.4f}'
					'\tflip_loss_s: {:.4f};\tscale_loss: {:.4f}'.format(
						#loss.data[0],
						loss.item(),
						flip_loss_l.item(), #data[0],
						flip_loss_s.item(), #data[0],
						scale_loss.item(), #data[0]
					)
				)
		
		if False:
			print("Here print all the net parameters:\n")
			for name, params in model.named_parameters():
				if 'gc' in str(name):
					print("name:\n ", name)
					print('content:\n ', params)
					print("content-size: \n", params.shape)
					
		print('backward count is :{0}\n'.format(backward_count))
		epoch_end = time.clock()
		elapsed = epoch_end - epoch_start
		print("Epoch time: ", elapsed)

		# test
		if (epoch+1) % args.snapshot == 0:

			model_file = model_prefix + '_epoch{}'+ datetime.datetime.now().strftime('%Y%m%d%H%M%S') +'.pth'
			print("Saving model to " + model_file.format(epoch))
			torch.save(model, model_file.format(epoch))

			if args.test:
				model.eval()
				test_start = time.clock()
				test_result = test(model, test_loader, epoch, flag=arch_flag)
				ALL_RESULT[str(epoch)] = test_result
				test_time = (time.clock() - test_start)
				print("test time: ", test_time)
				model.train()

	final_model = model_prefix + '_final' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pth'
	print("Saving model to " + final_model)
	torch.save(model, final_model)
	model.eval()
	last_result = test(model, test_loader, epoch+1, flag=arch_flag)
	ALL_RESULT['-1'] = last_result
	
	print("\n******************************* ALL THE RESULTS *******************************")
	for k, v in ALL_RESULT.items():
		print('epoch{0} result:{1}'.format(k, v))

if __name__ == '__main__':
	start_time = datetime.datetime.now()
	print("\nSTART TIME:", start_time.strftime('%Y-%m-%d %H:%M:%S'), "\n")  # print local time
	main()
	end_time = datetime.datetime.now()
	print("\nENE TIME:", end_time.strftime('%Y-%m-%d %H:%M:%S'))  # print local time
	use_time = (end_time - start_time).seconds
	# calculate elapse time
	m, s = divmod(use_time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	d = (end_time - start_time).days
	# print elapse time, format hours:mins:secs
	print("[Elapse time]:%02d-days:%02d-hours:%02d-minutes:%02d-seconds\n" % (d, h, m, s))
