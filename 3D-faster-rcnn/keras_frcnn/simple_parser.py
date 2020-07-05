import numpy as np

def get_data(input_path):
	'''
	parse the data from txt input.
	Input: a txt file, each line is filepath,x1,x2,y3,r,class_name
		Example:
			/data/imgs/img_001.npy,837,346,981,100,1kp8 
			/data/imgs/img_001.npy,500,316,428,88,1BXR
			/data/imgs/img_002.npy,49,34,158,30,1kp8 
	Output:
		A dictionary: 
			all_imgs[filename]['filepath'] = filename
			all_imgs[filename]['width'] = cols
			all_imgs[filename]['height'] = rows
			all_imgs[filename]['depth'] = dens		
			all_imgs[filename]['bboxes']= a list of :
				{'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'x3': int(float(x3)), 'r': int(float(r))}

	'''
	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,x2,x3,r,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = np.load(filename)
				(dens,rows,cols) = img.shape
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['depth'] = dens
				all_imgs[filename]['bboxes'] = []
				if np.random.randint(0,6) > 0:
					all_imgs[filename]['imageset'] = 'trainval'
				else:
					all_imgs[filename]['imageset'] = 'test'

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'x3': int(float(x3)), 'r': int(float(r))})


		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping
