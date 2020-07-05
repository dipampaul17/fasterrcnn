import numpy as np
import pdb
import math
from . import data_generators
import copy


def calc_iou(roi, img_data, C, class_mapping):
	"""Calculate IoU(groundtruth bboxes, roi bboxes) and use IoU to classify predicted roi bboxes into neg / pos classes."""
	bboxes = img_data['bboxes']
	(width, height, depth) = (img_data['width'], img_data['height'], img_data['depth'])
	# get image dimensions for resizing
	(resized_width, resized_height, resized_depth) = data_generators.get_new_img_size(width, height, depth, C.im_size)

	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_depth / float(depth)) / C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_height / float(height)) / C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['x3'] * (resized_width / float(width)) / C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['r'] * (resized_height / float(height)) / C.rpn_stride))

	x_roi = []  # final rois predicted for training
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = []  # for debugging only

	for ix in range(roi.shape[0]):  # for each predicted roi box
		(x1, x2, x3, r) = roi[ix, :]
		x1 = int(round(x1))
		x2 = int(round(x2))
		x3 = int(round(x3))
		r = int(round(r))

		best_iou = 0.0
		best_bbox = -1
		# Iterate all gt bboxes to find best iou and match bbox class
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou_r([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
											 [x1, x2, x3, r])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
			continue
		else:
			x_roi.append([x1, x2, x3, r])
			IoUs.append(best_iou)

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				x1_gt = gta[best_bbox, 0]
				x2_gt = gta[best_bbox, 1]
				x3_gt = gta[best_bbox, 2]

				tx1 = (x1_gt - x1) / float(r * 2)
				tx2 = (x2_gt - x2) / float(r * 2)
				tx3 = (x3_gt - x3) / float(r * 2)
				tr = np.log((gta[best_bbox, 3]) / float(r))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		class_num = class_mapping[cls_name]
		class_label = len(class_mapping) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		coords = [0] * 4 * (len(class_mapping) - 1)
		labels = [0] * 4 * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = 4 * class_num
			sx1, sx2, sx3, sr = C.classifier_regr_std
			coords[label_pos:4 + label_pos] = [sx1 * tx1, sx2 * tx2, sx3 * tx3, sr * tr]
			labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	if len(x_roi) == 0:
		return None, None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x1, x2, x3, r, tx1, tx2, tx3, tr):
	"""Apply regr layer to the anchor, for testing fcnn"""
	try:
		cx1 = tx1 * 2 * r + x1
		cx2 = tx2 * 2 * r + x2
		cx3 = tx3 * 2 * r + x3

		r = math.exp(tr) * r

		x1 = int(round(cx1))
		x2 = int(round(cx2))
		x3 = int(round(cx3))
		r = int(round(r))
		return x1, x2, x3, r

	except ValueError:
		return x1, x2, x3, r
	except OverflowError:
		return x1, x2, x3, r
	except Exception as e:
		print(e)
		return x1, x2, x3, r


def apply_regr_np(X, T):
	"""Apply regr layer to all anchors per feature map, for training frcnn"""
	try:
		x1 = X[0, :, :]
		x2 = X[1, :, :]
		x3 = X[2, :, :]
		r = X[3, :, :]

		tx1 = T[0, :, :]
		tx2 = T[1, :, :]
		tx3 = T[2, :, :]
		tr = T[3, :, :]

		cx1 = tx1 * 2 * r + x1
		cx2 = tx2 * 2 * r + x2
		cx3 = tx3 * 2 * r + x3

		r = np.exp(tr.astype(np.float64)) * r

		x1 = np.round(cx1)
		x2 = np.round(cx2)
		x3 = np.round(cx3)
		r = np.round(r)
		return np.stack([x1, x2, x3, r])
	except Exception as e:
		print(e)
		return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# modified to 3d version

	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	x2 = boxes[:, 1]
	x3 = boxes[:, 2]
	r = boxes[:, 3]

	np.testing.assert_array_less(0, r)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# calculate the volumes
	volume = 8 * r * r * r

	# sort the bounding boxes
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value to the list of picked indexes
		last = len(idxs) - 1
		picked_idx = idxs[last]
		pick.append(picked_idx)

		# find the intersect volume
		lu_x1_inter = np.maximum(x1[picked_idx] - r[picked_idx], x1[idxs[:last]] - r[idxs[:last]])
		lu_x2_inter = np.maximum(x2[picked_idx] - r[picked_idx], x2[idxs[:last]] - r[idxs[:last]])
		lu_x3_inter = np.maximum(x3[picked_idx] - r[picked_idx], x3[idxs[:last]] - r[idxs[:last]])
		rd_x1_inter = np.minimum(x1[picked_idx] + r[picked_idx], x1[idxs[:last]] + r[idxs[:last]])
		rd_x2_inter = np.minimum(x2[picked_idx] + r[picked_idx], x2[idxs[:last]] + r[idxs[:last]])
		rd_x3_inter = np.minimum(x3[picked_idx] + r[picked_idx], x3[idxs[:last]] + r[idxs[:last]])

		dx1_inter = np.maximum(0, rd_x1_inter - lu_x1_inter)
		dx2_inter = np.maximum(0, rd_x2_inter - lu_x2_inter)
		dx3_inter = np.maximum(0, rd_x3_inter - lu_x3_inter)

		volume_inter = dx1_inter * dx2_inter * dx3_inter

		# find the union volume
		volume_union = volume[picked_idx] + volume[idxs[:last]] - volume_inter

		# compute the ratio of overlap (IoU)
		overlap = volume_inter / (volume_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the int data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs


def rpn_to_roi(rpn_cls_layer, rpn_regr_layer, C, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
	"""Convert rpn output to roi bboxes"""
	rpn_regr_layer = rpn_regr_layer / C.std_scaling

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	assert rpn_cls_layer.shape[0] == 1
	if dim_ordering == 'channels_first':
		(depths, rows, cols) = rpn_cls_layer.shape[2:]
	elif dim_ordering == 'channels_last':
		(depths, rows, cols) = rpn_cls_layer.shape[1:4]

	curr_layer = 0
	# A: anchor position, shape: (4, depths, rows, cols, n_anchors_per_feature_map)
	if dim_ordering == 'channels_last':
		A = np.zeros(
			(4, rpn_cls_layer.shape[1], rpn_cls_layer.shape[2], rpn_cls_layer.shape[3], rpn_cls_layer.shape[4]))
	elif dim_ordering == 'channels_first':
		A = np.zeros(
			(4, rpn_cls_layer.shape[2], rpn_cls_layer.shape[3], rpn_cls_layer.shape[4], rpn_cls_layer.shape[1]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:
			anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride
			if dim_ordering == 'channels_first':
				regr = rpn_regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :, :]
			else:
				regr = rpn_regr_layer[0, :, :, :, 4 * curr_layer:4 * curr_layer + 4]
				regr = np.transpose(regr, (3, 0, 1, 2))

			# x1, x2, x3, r before regr
			x1, x2, x3 = np.meshgrid(np.arange(rows), np.arange(depths), np.arange(cols))  # Output order as z,y,x
			A[0, :, :, :, curr_layer] = x1
			A[1, :, :, :, curr_layer] = x2
			A[2, :, :, :, curr_layer] = x3
			A[3, :, :, :, curr_layer] = anchor_x  # anchor_x is radius

			if use_regr:
				A[:, :, :, :, curr_layer] = apply_regr_np(A[:, :, :, :, curr_layer], regr)

			# restrict x1, x2, x3: > 0 and < max_x_i
			A[0, :, :, :, curr_layer] = np.maximum(1, A[0, :, :, :, curr_layer])
			A[1, :, :, :, curr_layer] = np.maximum(1, A[1, :, :, :, curr_layer])
			A[2, :, :, :, curr_layer] = np.maximum(1, A[2, :, :, :, curr_layer])
			A[0, :, :, :, curr_layer] = np.minimum(depths - 1, A[0, :, :, :, curr_layer])
			A[1, :, :, :, curr_layer] = np.minimum(rows - 1, A[1, :, :, :, curr_layer])
			A[2, :, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, :, curr_layer])

			# Question: do we need to ensure radius inside boundary at this stage ???
			# In the dataset, many of the target object lays near to the boundary, we still want to predict them.
			# if we shrinked the radius at the stage, what is the influence ? 
			# Will it cause the finial detection results much smaller than the groundtruth, IoU<threshold, and be marked as misdetected?
			# I am not sure, need to be checked.
			# 
			# restrict r: < x_i and < max_x_i - x_i and > 0
			A[3, :, :, :, curr_layer] = np.maximum(1, A[3, :, :, :, curr_layer])
			'''
			A[3, :, :, :, curr_layer] = np.minimum(A[0, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(A[1, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(A[2, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(depths - A[0, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(rows - A[1, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(cols - A[2, :, :, :, curr_layer], A[3, :, :, :, curr_layer])
			'''

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0, 4, 1, 2, 3)), (4, -1)).transpose((1, 0))
	all_probs = rpn_cls_layer.transpose((0, 4, 1, 2, 3)).reshape((-1))

	# NMS and return bboxes only, because rpn_probs is not needed later.
	roi_bboxes = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]
	# return boxes = [depth,heighth,width,r]
	return roi_bboxes
