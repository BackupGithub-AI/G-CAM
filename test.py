# -*- coding=utf-8 -*-
####################################################################################
# test.py文件计算并打印评估指标。
####################################################################################
import torch, sys
import torch.nn as nn
from tqdm import tqdm, trange

import torchvision.models as models

from torch.autograd import Variable

import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from sklearn.metrics import average_precision_score
from wider import get_subsets, imshow

import scipy.io
import os


# num_attr = 14

def calc_average_precision(y_true, y_score, num_classes):
	# print("(2)y_true = ", y_true.shape, y_true)
	# print("(3)y_score = ", y_score.shape, y_score)
	aps = np.zeros(num_classes)
	for i in range(num_classes):
		true = y_true[i]
		score = y_score[i]
		
		non_index = np.where(true == 0)
		score = np.delete(score, non_index)
		true = np.delete(true, non_index)
		
		true[true == -1.] = 0
		
		# print("(4)true = ", true, true.shape)
		# print('(5)score = ', score, score.shape)
		ap = average_precision_score(true, score)
		aps[i] = ap
	
	return aps


def calc_acc_pr_f1(y_true, y_pred, num_classes):
	precision = np.zeros(num_classes)
	recall = np.zeros(num_classes)
	accuracy = np.zeros(num_classes)
	f1 = np.zeros(num_classes)
	for i in range(num_classes):
		true = y_true[i]
		pred = y_pred[i]
		
		true[true == -1.] = 0
		
		precision[i] = metrics.precision_score(true, pred)
		recall[i] = metrics.recall_score(true, pred)
		accuracy[i] = metrics.accuracy_score(true, pred)
		f1[i] = metrics.f1_score(true, pred)
	
	return precision, recall, accuracy, f1


def calc_mean_acc(y_true, y_pred, num_classes):
	macc = np.zeros(num_classes)
	for i in range(num_classes):
		true = y_true[i]  # -1, 0, 1
		pred = y_pred[i]  # 0, 1
		
		true[true == -1.] = 0
		
		temp = true + pred
		tp = (temp[temp == 2]).size
		tn = (temp[temp == 0]).size
		p = (true[true == 1]).size
		n = (true[true == 0]).size
		
		macc[i] = .5 * tp / (p) + .5 * tn / (n)
	
	return macc


def calc_acc_pr_f1_overall(y_true, y_pred):
	true = y_true
	pred = y_pred
	
	true[true == -1.] = 0
	
	precision = metrics.precision_score(true, pred)
	recall = metrics.recall_score(true, pred)
	accuracy = metrics.accuracy_score(true, pred)
	f1 = metrics.f1_score(true, pred)
	
	return precision, recall, accuracy, f1


def calc_mean_acc_overall(y_true, y_pred):
	true = y_true  # 0, 1
	pred = y_pred  # 0, 1
	
	true[true == -1.] = 0
	
	temp = true + pred
	tp = (temp[temp == 2]).size
	tn = (temp[temp == 0]).size
	p = (true[true == 1]).size
	n = (true[true == 0]).size
	macc = .5 * tp / (p) + .5 * tn / (n)
	
	return macc


def eval_example(y_true, y_pred):
	# example-based metrics
	N = y_true.shape[1]
	
	acc = 0.
	prec = 0.
	rec = 0.
	f1 = 0.
	
	for i in range(N):
		true_exam = y_true[:, i]  # column: labels for an example
		pred_exam = y_pred[:, i]
		
		temp = true_exam + pred_exam
		
		yi = true_exam.sum()  # number of attributes for i
		fi = pred_exam.sum()  # number of predicted attributes for i
		ui = (temp > 0).sum()  # temp == 1 or 2 means the union of attributes in yi and fi
		ii = (temp == 2).sum()  # temp == 2 means the intersection
		
		if ui != 0:
			acc += 1.0 * ii / ui
		if fi != 0:
			prec += 1.0 * ii / fi
		if yi != 0:
			rec += 1.0 * ii / yi
	
	acc /= N
	prec /= N
	rec /= N
	f1 = 2.0 * prec * rec / (prec + rec)
	return acc, prec, rec, f1


def extract_top(pred, gtruth, k=3):
	# default find top k value
	# pred_tmp and gtruth_tmp shape are all [test_set_item, num_classes]
	pred_tmp = torch.transpose(pred, 0, 1)
	gtruth_tmp = torch.transpose(gtruth, 0, 1)
	# print('index = ', index[0], index[1])
	gtruth_tmp[gtruth_tmp == -1] = 0
	n, c = pred_tmp.shape[0], pred_tmp.shape[1]
	scores = np.zeros((n, c)) - 1
	index = pred_tmp.topk(k, 1, True, True)[1].cpu().numpy()		# index[1] is the `index` tensor
	tmp = pred_tmp.cpu().numpy()
	for i in range(n):
		for ind in index[i]:
			scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
	return evaluation(scores, gtruth_tmp.numpy())


def evaluation( scores_, targets_):
	n, n_class = scores_.shape
	Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
	for k in range(n_class):
		scores = scores_[:, k]
		targets = targets_[:, k]
		targets[targets == -1] = 0
		Ng[k] = np.sum(targets == 1)
		Np[k] = np.sum(scores >= 0)
		Nc[k] = np.sum(targets * (scores >= 0))
		# print("(7) Ng[{0}] = {1}".format(k, Ng[k]))
		# print("(8) Np[{0}] = {1}".format(k, Np[k]))
		# print("(9) Nc[{0}] = {1}".format(k, Nc[k]))
	Np[Np == 0] = 1
	OP = np.sum(Nc) / np.sum(Np)
	# print("(10) OP = ", OP)
	OR = np.sum(Nc) / np.sum(Ng)
	# print("(11) OR = ", OR)
	OF1 = (2 * OP * OR) / (OP + OR)
	# print("(11) OF1 = ", OF1)
	
	CP = np.sum(Nc / Np) / n_class
	# print("(12) CP = ", CP)
	CR = np.sum(Nc / Ng) / n_class
	# print("(13) CR = ", CR)
	CF1 = (2 * CP * CR) / (CP + CR)
	# print("(14) CF1 = ", CF1)
	return OP, OR, OF1, CP, CR, CF1


def test(model, test_loader, epoch, flag=False):
	print("({0})Testing *** ".format(epoch))
	num_classes = 0
	probs = torch.FloatTensor()
	gtruth = torch.FloatTensor()
	probs = probs.cuda()
	gtruth = gtruth.cuda()
	test_loader = tqdm(test_loader, desc='({0})Test'.format(epoch))
	for i, sample in enumerate(test_loader):
		images = sample[0]  # test just large
		labels = sample[4]
		if flag:
			inp = sample[5]
			inp = inp.type(torch.FloatTensor)
			test_inp = Variable(inp)
		labels = labels.type(torch.FloatTensor)
		
		images = images.cuda() if torch.cuda.is_available() else images
		labels = labels.cuda() if torch.cuda.is_available() else labels
		
		test_input = Variable(images)
		
		if flag:
			y, _ = model(test_input, test_inp)  # adj involves in the forward processing
		else:
			y, _ = model(test_input)
		
		num_classes = y.shape[1]
		# every loop :probs.size = [num_classes, batch_size], when loop end, the size is [num_classes, testset_item]
		probs = torch.cat((probs, y.data.transpose(1, 0)), 1)
		# every loop :gtruth.size = [num_classes, batch_size], when loop end, the size is [num_classes, testset_item]
		gtruth = torch.cat((gtruth, labels.transpose(1, 0)), 1)
		print("probs = {0}, gtruth = {1}".format(probs, gtruth))
		sys.exit()
	
	print('prediction finished ....')
	OP, OR, OF1, CP, CR, CF1 = extract_top(probs.cpu(), gtruth.cpu())
	print("top_3 performance: \nOP={0}, OR={1}, OF1={2}, CP={3}, CR={4} CF1={5}\n". \
		  format(OP, OR, OF1, CP, CR, CF1))
	# print('probs = ', probs, probs.shape)
	preds = np.zeros((probs.size(0), probs.size(1)))
	temp = probs.cpu().numpy()
	preds[temp > 0.] = 1
	# print('preds = ', preds, preds.shape)
	
	if not os.path.isdir('./preds'):
		os.mkdir('./preds')
	scipy.io.savemat('./preds/prediction_e{}.mat'.format(epoch), dict(gt=gtruth.cpu().numpy(), \
																	  prob=probs.cpu().numpy(), pred=preds))
	
	aps = calc_average_precision(gtruth.cpu().numpy(), probs.cpu().numpy(), num_classes=num_classes)
	print('>>>>>>>>>>>>>>>>>>>>>>>> Average for Each Attribute >>>>>>>>>>>>>>>>>>>>>>>>>>>')
	print("APs")
	print(aps)
	precision, recall, accuracy, f1 = calc_acc_pr_f1(gtruth.cpu().numpy(), preds, num_classes=num_classes)
	print('precision scores')
	print(precision)
	print('recall scores')
	print(recall)
	print('f1 scores')
	print(f1)
	print('')
	
	print("AP: {}".format(aps.mean()))
	print('F1-C: {}'.format(f1.mean()))
	print('P-C: {}'.format(precision.mean()))
	print('R-C: {}'.format(recall.mean()))
	print('')
	
	print('>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>')
	o_precision, o_recall, o_accuracy, o_f1 = calc_acc_pr_f1_overall(gtruth.cpu().numpy().flatten(),
																	 preds.flatten())
	
	print('F1_O: {}'.format(o_f1))
	print('P_O: {}'.format(o_precision))
	print('R_O: {}'.format(o_recall))
	print('\n')
	
	macc = calc_mean_acc(gtruth.cpu().numpy(), preds, num_classes=num_classes)
	print('mA scores')
	print(macc)
	print('mean mA')
	print(macc.mean())
	
	print('\n')
	
	return {'aps.mean':aps.mean(), 
		'f1.mean':f1.mean(), 
		'precision.mean':precision.mean(), 
		'overall_P':o_precision, 
		'overall_R':o_recall, 
		'overall_acc':o_accuracy, 
		'overall_f1':o_f1, 
		'macc':macc, 
		'mean_macc':macc.mean(),
		'top-3_OP': OP,
		'top-3_OR': OR,
		'top-3_OF1': OF1,
		'top-3_CP': CP,
		'top-3_CR':CR,
		'top-3_CF1': CF1, }


if __name__ == '__main__':
	anno_dir = '/path/to/wider_attribute_annotation'
	data_dir = '/path/to/Image'
	trainset, testset = get_subsets(anno_dir, data_dir)
	test_loader = torch.utils.data.DataLoader(testset,
											  batch_size=16,
											  shuffle=False,
											  num_workers=4)
	
	# modify to test multiple checkpoints continuously
	for i in range(11, 12):
		model_file = '/path/to/model_resnet50_{}.pth'.format(i)
		model = torch.load(model_file)
		print(model_file)
		model.eval()
		start_time = time.clock()
		test(model, test_loader, i)
		end_time = time.clock()
		print('Time: ', end_time - start_time)
		print('\n')
