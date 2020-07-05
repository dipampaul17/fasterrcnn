import numpy as np
import copy

import scipy.ndimage as ndimage
import skimage.transform as transform

def augment(img_data, config, augment=True):
	'''
	From the result of simple parser, load the image and do augmentation
	'''
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data
	assert 'depth' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = np.load(img_data_aug['filepath'])

	if augment:
		depth, rows, cols = img.shape[:3]
		if config.trans_prespective_y and np.random.randint(0, 2) == 0:
			img = img.transpose((2,1,0))
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x3 = bbox['x3']
				bbox['x1'] = x3
				bbox['x3'] = x1			

		if config.trans_prespective_x and np.random.randint(0, 2) == 0:
			img = img.transpose(1,0,2)
			for bbox in img_data_aug['bboxes']:
				x2 = bbox['x2']
				x3 = bbox['x3']
				bbox['x2'] = x3
				bbox['x3'] = x2

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			#img = np.fliplr(img) left-right reflection
			img = np.array([np.fliplr(img[i,:,:]) for i in range(img.shape[0])])
			for bbox in img_data_aug['bboxes']:
				x3 = bbox['x3']
				bbox['x3'] = cols - x3

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			#b[::-1] bottom-up reflection
			img = np.array([img[i,:,:][::-1] for i in range(img.shape[0])])
			for bbox in img_data_aug['bboxes']:
				x2 = bbox['x2']
				bbox['x2'] = rows - x2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			#anti-circle
			if angle == 270:
				#b=a[::-1].T
				img = np.array([img[i,:,:][::-1].T for i in range(img.shape[0])])
				for bbox in img_data_aug['bboxes']:
					x2 = bbox['x2']
					x3 = bbox['x3']
					bbox['x2'] = x3
					bbox['x3'] = rows-x2

			elif angle == 180:
				#np.fliplr(b)[::-1]
				img = np.array([np.fliplr(img[i,:,:])[::-1] for i in range(img.shape[0])])
				for bbox in img_data_aug['bboxes']:
					x2 = bbox['x2']
					x3 = bbox['x3']
					bbox['x2'] = rows-x2
					bbox['x3'] = cols-x3

			elif angle == 90:
				#b.T[::-1]
				img = np.array([img[i,:,:].T[::-1] for i in range(img.shape[0])])
				for bbox in img_data_aug['bboxes']:
					x2 = bbox['x2']
					x3 = bbox['x3']
					bbox['x2'] = cols-x3
					bbox['x3'] = x2

			elif angle == 0:
				pass


	img_data_aug['width'] = img.shape[2]
	img_data_aug['height'] = img.shape[1]
	img_data_aug['depth'] = img.shape[0]
	return img_data_aug, img

