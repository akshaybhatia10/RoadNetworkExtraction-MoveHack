import sys
sys.path.append('./fastai')
import matplotlib.pyplot as plt
from fastai.conv_learner import *
from fastai.dataset import *

f = resnet34
cut, lr_cut = model_meta[f]

class SaveFeatures():
    features=None
    
    def __init__(self, m): 
    	self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output): 
    	self.features = output
    
    def remove(self): 
    	self.hook.remove()

def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax

def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)

def dice(pred, targs):
    m1 = (pred[:,0]>0).float()
    m2 = targs[...,0]
    return 2. * (m1*m2).sum() / (m1+m2).sum()

def mask_loss(pred,targ):
    return F.binary_cross_entropy_with_logits(pred[:,0],targ[...,0])

def mask_acc(pred,targ): 
	return accuracy_multi(pred[:,0], targ[...,0], 0.)

def get_file_list(path):
	train_sat_files = '{}/train/sat'.format(path)
	train_mask_files = '{}/train/map'.format(path)
	train_x = glob(os.path.join(train_sat_files, "*.png"))
	train_y = glob(os.path.join(train_mask_files, "*.png"))

	valid_sat_files = '{}/valid/sat'.format(path)
	valid_map_files = '{}/valid/map'.format(path)
	valid_x = glob(os.path.join(valid_sat_files, "*.png"))
	valid_y = glob(os.path.join(valid_map_files, "*.png"))

	test_sat_files = '{}/test/sat'.format(path)
	test_map_files = '{}/test/map'.format(path)
	test_x = glob(os.path.join(test_sat_files, "*.png"))
	test_y = glob(os.path.join(test_map_files, "*.png"))

	return train_x, train_y, valid_x, valid_y, test_x, test_y