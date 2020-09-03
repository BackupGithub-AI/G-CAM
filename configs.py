
# Dataset configurations

datasets = [ "coco",  'mirflickr25k', 'nuswide']

MIRFLICKR25K_DATA_DIR = '../ML-SGCN4TNNLS/data/mirflickr25k'
MIRFLICKR25K_INP_DIR = '../ML-SGCN4TNNLS/data/mirflickr25k/mirflickr25k_glove_word2vec.pkl'

# NUSWIDE dataset

# MS-COCO dataset

def get_configs(dataset, optdic={}):
	opts = {}
	if not dataset in datasets:
		raise Exception("Not supported dataset!")
	else:
		if dataset == 'mirflickr25k':
			opts['dataset'] = 'MIRFLICKR25K'
			opts["num_labels"] = 24
			opts['data_dir'] = MIRFLICKR25K_DATA_DIR
			opts["mirflickr25k_inp_dir"] = MIRFLICKR25K_INP_DIR
		
		elif dataset == 'coco':
			# will be added later
			pass		

		elif dataset == 'nuswide':
			# will be added later
			pass
		
		else: assert False, "No such a dataset name...\n"
		
	return opts
