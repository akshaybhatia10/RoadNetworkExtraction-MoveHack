import os
import sys
import argparse
import torch
from dataset import *
from helper import *
from model import *
from fastai.conv_learner import *
from fastai.dataset import *
from pathlib import Path
import json

sys.path.append('./fastai')

if torch.cuda.is_available():
	torch.backends.cudnn.enabled = True

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='mass_roads', \
			help='Path to the dataset')
	parser.add_argument('--learning_rate', type=float, default='0.1', \
			help='Learning Rate')
	parser.add_argument('--model_dir', type=str, default='models/', \
			help='Path to the complete trained model file(.h5)')
	parser.add_argument('--mode', type=str, default='test', \
			help='Learning Rate')
	parser.add_argument('--num_epochs', type=int, default='1', \
			help='Number of epochs')
	parser.add_argument('--cycle_len', type=int, default='2', \
			help='Cycle Length')
	parser.add_argument('--test_img', type=str, default='test.png', \
			help='Test Image Path')

	args = parser.parse_args()
	cwd = os.getcwd()
	
	f = resnet34
	cut, lr_cut = model_meta[f]

	if args.mode == 'train':
		new_base_path = convert_and_resize(args.data_dir)
		train_x, train_y, valid_x, valid_y, test_x, test_y \
					= get_file_list(os.path.join(cwd, new_base_path))
		
		PATH = Path(os.path.join(cwd, new_base_path))
		img_size = 1024
		batch_size = 2
		aug_tfms = []

		tfms = tfms_from_model(resnet34, img_size, crop_type=CropType.NO, \
							tfm_y=TfmType.NO, aug_tfms=aug_tfms)
		datasets = ImageData.get_ds(MatchedFilesDataset, (train_x, train_y),
							(valid_x, valid_y), tfms, path=PATH)
		md = ImageData(PATH, datasets, batch_size, num_workers=4, classes=None)
		denorm = md.trn_ds.denorm
		x, y = next(iter(md.trn_dl))
		print (x.shape, y.shape)
		m_base = get_base()
		m = to_gpu(Unet34(m_base))
		models = UpsampleModel(m)
		
		learn = ConvLearner(md, models)
		learn.opt_fn=optim.Adam
		learn.crit=mask_loss
		learn.metrics=[mask_acc, dice]
		learn.freeze_to(1)
		learn.load(os.path.join(cwd, 'models/1024Deepglobe-tmp'))
		print ('Started Training...')
		learn.fit(args.learning_rate, args.num_epochs, cycle_len=args.cycle_len, use_clr=(20,4))
		learn.save(os.path.join(cwd, 'models/Mnih-final-1024'))

	elif args.mode == 'test':
		PATH = Path('./')
		img_size = 1024
		batch_size = 1
		aug_tfms = []
		m_base = get_base()
		m = to_gpu(Unet34(m_base))
		models = UpsampleModel(m)

		t_img = [args.test_img]
		save_path = [args.test_img.split('/')[-1]]
		print (save_path, args.test_img)
		img = Image.open(args.test_img).resize((1024,1024)).save('1024_' + save_path[0])
		
		tfms = tfms_from_model(resnet34, img_size, crop_type=CropType.NO, tfm_y=TfmType.NO, aug_tfms=aug_tfms)
		datasets = ImageData.get_ds(MatchedFilesDataset, (t_img, t_img), (t_img, t_img), tfms, path=PATH)
		md = ImageData(PATH, datasets, batch_size, num_workers=4, classes=None)
		denorm = md.trn_ds.denorm
		
		learn = ConvLearner(md, models)
		learn.load(os.path.join(cwd, 'models/1024DeepGlobe-Mnih-tmp'))
		
		x, _ = next(iter(md.trn_dl))
		start = time.time()
		py = to_np(learn.model(V(x)))
		end = time.time()
		print ('Prediction Time', (end-start), 'seconds')
		s = py[0][0]*255.0
		cv2.imwrite('./' + 'masked_' + save_path[0], s)

		inp = './1024_'+ save_path[0]
		out = './masked_' + save_path[0]
		img = cv2.imread(inp)
		mask = cv2.imread(out, 0)
		copy = img.copy()
		new = np.zeros(img.shape, img.dtype)
		new[:,:] = (255, 10, 10)
		new_mask = cv2.bitwise_and(new, new, mask=mask)
		cv2.addWeighted(new_mask, 1, img, 0.6, 0, img)
		cv2.imwrite('./' + 'overlay_' + save_path[0], img)

	else:
		pass

if __name__ == '__main__':
	main()