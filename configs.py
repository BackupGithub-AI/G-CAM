
# Dataset configurations

datasets = ["wider", "pa-100k", "coco", 'voc2007', 'mirflickr25k', 'nuswide']


WIDER_DATA_DIR = "./data/wider_attribute/Image"
WIDER_ANNO_DIR = "./data/wider_attribute/wider_attribute_annotation"

PA100K_DATA_dir = "/data/hguo/Datasets/PA-100K/release_data"
PA100K_ANNO_FILE = "/data/hguo/Datasets/PA-100K/annotation/annotation.mat"

MSCOCO_DATA_DIR = "./data/coco"
MSCOCO_INP_DIR = '/data/coco/coco_glove_word2vec.pkl'

VOC2007_DATA_DIR = './data/voc'
VOC2007_INP_DIR = './data/voc/voc_glove_word2vec.pkl'

MIRFLICKR25K_DATA_DIR = '../ML-SGCN4TNNLS/data/mirflickr25k'
MIRFLICKR25K_INP_DIR = '../ML-SGCN4TNNLS/data/mirflickr25k/mirflickr25k_glove_word2vec.pkl'

NUSWIDE_DATA_DIR = '../ML-SGCN4TNNLS/data/nuswide'
NUSWIDE_INP_DIR = '../ML-SGCN4TNNLS/data/nuswide/nuswide_glove_word2vec.pkl'

# pre-calculated weights to balance positive and negative samples of each label
# as defined in Li et al. ACPR'15
# WIDER dataset
wider_pos_ratio = [0.5669, 0.2244, 0.0502, 0.2260, 0.2191, 0.4647, 0.0699, 0.1542, \
	0.0816, 0.3621, 0.1005, 0.0330, 0.2682, 0.0543]

# PA-100K dataset
pa100k_pos_ratio = [0.460444, 0.013456, 0.924378, 0.062167, 0.352667, 0.294622, \
	0.352711, 0.043544, 0.179978, 0.185000, 0.192733, 0.160100, 0.009522, \
	0.583400, 0.416600, 0.049478, 0.151044, 0.107756, 0.041911, 0.004722, \
	0.016889, 0.032411, 0.711711, 0.173444, 0.114844, 0.006000]

# MS-COCO dataset

def get_configs(dataset, optdic={}):
	opts = {}
	if not dataset in datasets:
		raise Exception("Not supported dataset!")
	else:
		if dataset == "wider":
			opts["dataset"] = "WIDER"
			opts["num_labels"] = 14
			opts["data_dir"] = WIDER_DATA_DIR
			opts["anno_dir"] = WIDER_ANNO_DIR
			opts["pos_ratio"] = wider_pos_ratio
		
		elif dataset == "pa-100k":
			# will be added later
			pass
		
		elif dataset == 'coco':
			opts['dataset'] = 'COCO'
			opts["num_labels"] = 80
			opts['data_dir'] = MSCOCO_DATA_DIR
			opts["coco_inp_dir"] = MSCOCO_INP_DIR
		
		elif dataset == 'voc2007':
			opts['dataset'] = 'VOC2007'
			opts["num_labels"] = 20
			opts['data_dir'] = VOC2007_DATA_DIR
			opts["voc_inp_dir"] = VOC2007_INP_DIR
		
		elif dataset == 'mirflickr25k':
			opts['dataset'] = 'MIRFLICKR25K'
			opts["num_labels"] = 24
			opts['data_dir'] = MIRFLICKR25K_DATA_DIR
			opts["mirflickr25k_inp_dir"] = MIRFLICKR25K_INP_DIR

		elif dataset == 'nuswide':
			opts['dataset'] = 'NUSWIDE'
			opts['num_labels'] = 81
			opts['data_dir'] = NUSWIDE_DATA_DIR
			opts['nuswide_inp_dir'] = NUSWIDE_INP_DIR
		
		else: assert False, "No such a dataset name...\n"
		
	return opts
