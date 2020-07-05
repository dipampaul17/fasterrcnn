import numpy as np
from sklearn.metrics import average_precision_score
####
#Notes:
#Each real image contains ~ 6 roi, 48*48*48
# Training set 150K 2D, 30K val, 96K testing
#real No.7835, has 3050 rois???
#In 8000 images, 44contains > 10 roi (Total 15,418 rois), 18 contains>100 roi
#
#Each sim  image contains ~1000roi, 512*512*128
#
####

#####
#Check:
#Same image are actually different pictures in my model, that is, 
#     oneimage with multiple roi -> now one image with one roi
#     check infulence
#####


####
#Q1: Test: To many 2D images.  Test only on x and y? Step?
#Q2: Assemble method: assemble centre point for bbox? ->bbox now
####

result_file = '2Dresults.txt'
#label_file = '../label_test.txt'
gt_file = 'sim-3D-annotation-test.txt'

overlap_thresh=0.5
dis_threshold=7
'''
steps:
2D assemble -> 3D
find corresponding gt
average label in 3 dimentions, compare with 3D => get mAP
'''

'''
TODO and NOTES:
0. re-prepare trainning and testing data
1. now only implement test x,y plane, step=2
2. what to do if get different label from each plane? =>choose high probs
'''
def read_result_file(fname='2Dresults.txt'):
    
    f = open(fname,'r')

    all_imgs = {} 

    for line in f:
        line_split = line.strip().split(',')
        (filename,x1,x2,y1,y2,class_name,probs,fx,fy) = line_split
        tomo = filename.split('/')[1].split('_roi')[0] # e.g. str '0145'
        roi = filename.split('/')[1].split('_roi')[1].split('_')[0][:-1] # e.g. str '0'
        direction =  filename.split('/')[1].split('_roi')[1].split('_')[0][-1]# e.g. str 'x'
        ind_direction =  filename.split('/')[1].split('_roi')[1].split('_')[1].split('.')[0] #e.g. '11'
        
        if tomo not in all_imgs:
            all_imgs[tomo] = {}
        if direction not in all_imgs[tomo]:
            all_imgs[tomo][direction] = {}
        if ind_direction not in all_imgs[tomo][direction]:
            all_imgs[tomo][direction][ind_direction] = []
        all_imgs[tomo][direction][ind_direction].append({'class': class_name,'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2)), 'probs': float(probs), 'fx':float(fx), 'fy':float(fy)})

    return all_imgs

def read_gt_file(fname='real-3D-annotation-test.txt'):
    
    f = open(fname,'r')

    all_imgs = {} 

    for line in f:
        line_split = line.strip().split(',')
        (tomo,x1,y1,z1,x2,y2,z2,class_name) = line_split
        #(tomo, z1, y1, x1, z2, y2, x2, class_name) = line_split
        #tomo e.g. str '0145'
        if tomo not in all_imgs:
            all_imgs[tomo] = {}
            all_imgs[tomo]['bbox']=[]

        all_imgs[tomo]['bbox'].append({'class': class_name,'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2)),'z1': int(float(z1)), 'z2': int(float(z2))})

    return all_imgs


#Do NMS for all images along one directions, get roughly 3D results
def nms_for_plane(dic, overlap_thresh=0.9, dis_threshold=10, max_boxes=300):
    
    inds = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    label = []
    probs = []
    # a group a images have the same size
    for ind in dic:
        for roi_item in dic[ind]:
            inds.append(int(ind))
            x1.append(roi_item['x1'])
            x2.append(roi_item['x2'])
            y1.append(roi_item['y1'])
            y2.append(roi_item['y2'])
            probs.append(roi_item['probs'])
            label.append(roi_item['class'])
            fx = roi_item['fx']
            fy = roi_item['fy']
    
    dis_threshold = dis_threshold/fx

    # if there are no boxes, return an empty list
    if len(inds) == 0:
        return []

    inds = np.array(inds )
    x1   = np.array(x1   )
    x2   = np.array(x2   )
    y1   = np.array(y1   )
    y2   = np.array(y2   )
    label= np.array(label)
    probs= np.array(probs)
    '''
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    '''

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)
    
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    #if boxes.dtype.kind == "i":
    #    boxes = boxes.astype("float")
    if x1.dtype.kind == "i":
        x1 = x1.astype("float")
    if x2.dtype.kind == "i":
        x2 = x2.astype("float")
    if y1.dtype.kind == "i":
        y1 = y1.astype("float")
    if y2.dtype.kind == "i":
        y2 = y2.astype("float")

    # initialize the list of picked indexes 
    pick = []

    min_range = []
    max_range = []
    central_ind = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        ###3D here
        c_ind = inds[idxs[last]]
        rec_ind = []
        rec_ind.append(c_ind)
        central_ind.append(c_ind)
        if len(np.where(overlap > overlap_thresh)[0]) > 0: 
            target_ids = np.where(overlap > overlap_thresh)[0]
            for iii in range(len(target_ids)):
                ind = inds[idxs[target_ids[iii]]]
                if abs(ind - c_ind) > dis_threshold:
                    #exceed distance, should be another one
                    target_ids.pop(iii)
                else:
                    rec_ind.append(ind)
            if len(target_ids)>0:
                idxs = np.delete(idxs, np.concatenate(([last],np.array(target_ids))))
            else:
                idxs = np.delete(idxs, [last])                
        else:
            idxs = np.delete(idxs, [last])

        min_range.append(np.min(rec_ind))
        max_range.append(np.max(rec_ind))
        
        #idxs = np.delete(idxs, np.concatenate(([last],
        #    np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break   

    # return only the bounding boxes that were picked using the integer data type
    boxes = np.array([x1[pick],x2[pick],y1[pick],y2[pick]]).astype("int") #boxes[0] is an array of x1
    
    probs = probs[pick]
    label = label[pick]
    return boxes, probs, label,np.array(min_range), np.array(max_range),np.array(central_ind), (fx,fy)




#Do NMS for 3 Directions,get precise results
def nms_for_assemble():

    return

def assemble_process(all_imgs, thres1, thres2):
    res_tomos = {}

    for tomo in all_imgs:
        if tomo not in res_tomos:
            res_tomos[tomo] = {}
            res_tomos[tomo]['bbox']=[]
        
        if 'x' not in all_imgs[tomo] or 'y' not in all_imgs[tomo] or 'z' not in all_imgs[tomo]:
            print('Warning: tomo {} contains no roi'.format(tomo))
            continue


        xboxes, xprobs, xlabel, xmin_range, xmax_range, x_ind,fxx = nms_for_plane(all_imgs[tomo]['x'], overlap_thresh=thres1, dis_threshold=thres2)
        yboxes, yprobs, ylabel, ymin_range, ymax_range, y_ind,fyy = nms_for_plane(all_imgs[tomo]['y'], overlap_thresh=thres1, dis_threshold=thres2)
        zboxes, zprobs, zlabel, zmin_range, zmax_range, z_ind,fzz = nms_for_plane(all_imgs[tomo]['z'], overlap_thresh=thres1, dis_threshold=thres2)
        
        #xboxes = [y1,y2,z1,z2]
        #yboxes = [x1,x2,z1,z2]
        xmin_range=(xmin_range/fyy[0]).astype("int"); xmax_range=(xmax_range/fyy[0]).astype("int"); x_ind = (x_ind/fyy[0]).astype("int")
        ymin_range=(ymin_range/fxx[0]).astype("int"); ymax_range=(ymax_range/fxx[0]).astype("int"); y_ind = (y_ind/fxx[0]).astype("int")
        zmin_range=(zmin_range/fxx[1]).astype("int"); zmax_range=(zmax_range/fxx[1]).astype("int"); z_ind = (z_ind/fxx[1]).astype("int")


        #nums_for_assemble()
        #for roi in range(len(xprobs)):
        #    res_tomos[tomo]['bbox'].append({'class': xlabel[roi],'x1': xmin_range[roi], 'x2': xmax_range[roi]+1, 'y1': xboxes[0][roi], 'y2': xboxes[1][roi],'z1':xboxes[2][roi],'z2':xboxes[3][roi], 'probs': xprobs[roi]})
        #for roi in range(len(yprobs)):
        #    res_tomos[tomo]['bbox'].append({'class': ylabel[roi],'y1': ymin_range[roi], 'y2': ymax_range[roi]+1, 'x1': yboxes[0][roi], 'x2': yboxes[1][roi],'z1': yboxes[2][roi],'z2': yboxes[3][roi], 'probs': yprobs[roi]})
        for roi in range(len(zprobs)):
            res_tomos[tomo]['bbox'].append({'class': zlabel[roi],'z1': zmin_range[roi], 'z2': zmax_range[roi]+1, 'x1': zboxes[0][roi], 'x2': zboxes[1][roi],'y1': zboxes[2][roi],'y2': zboxes[3][roi], 'probs': zprobs[roi]})
        res_tomos[tomo]['f'] = {'fxx':fxx, 'fyy':fyy, 'fzz':fzz}
    return res_tomos

def union(au, bu, area_intersection):
    area_a = (au[3] - au[0]) * (au[4] - au[1]) * (au[5] - au[2])
    area_b = (bu[3] - bu[0]) * (bu[4] - bu[1]) * (bu[5] - bu[2])
    area_union = area_a + area_b - area_intersection
    return area_union
def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    z = max(ai[2], bi[2])
    w = min(ai[3], bi[3]) - x
    h = min(ai[4], bi[4]) - y
    d = min(ai[5], bi[5]) - z
    if w < 0 or h < 0 or d < 0:
        return 0
    return w*h*d
def cube_iou(a, b):
    # a and b should be (x1,y1,z1,x2,y2,z2)
    if a[0] >= a[3] or a[1] >= a[4] or a[2]>=a[5] or b[0] >= b[3] or b[1] >= b[4] or b[2]>=b[5]:
        print('WARNING: indice mismatch')
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)


def cube_map(pred,gt,f):
    T = {}
    P = {}
    #fx, fy = f
    fxx, fxy = f['fxx'] #y,z
    fyx, fyy = f['fyy'] #x,z
    fzx, fzy = f['fzz'] #x,y

    ##
    if fxx!=fzy or fxy!=fyy or fyx!=fzx:
        print('Warning: rescale size mismatch..')

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['probs'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_z1 = pred_box['z1']
        pred_z2 = pred_box['z2']
        pred_prob = pred_box['probs']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            #####################
            ###NEED TO CHECK HERE
            #####################
            gt_x1 = gt_box['x1']/fyx
            gt_x2 = gt_box['x2']/fyx
            gt_y1 = gt_box['y1']/fxx
            gt_y2 = gt_box['y2']/fxx
            gt_z1 = gt_box['z1']/fxy
            gt_z2 = gt_box['z2']/fxy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = cube_iou((pred_x1, pred_y1,pred_z1, pred_x2, pred_y2, pred_z2), (gt_x1, gt_y1,gt_z1, gt_x2, gt_y2, gt_z2))

            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched']:
        #if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    #import pdb
    #pdb.set_trace()
    return T, P


if __name__ == '__main__':

    result_dic= read_result_file(result_file)
    predict_result = assemble_process(result_dic, overlap_thresh, dis_threshold)

    gt_dic = read_gt_file(gt_file)

    #get maps
    T={}
    P={}
    images_map = []
    miss = 0
    for tomo_id in predict_result:
        if tomo_id not in gt_dic:
            print('WARINGING: {} not in test set'.format(tomo_id))
        else:
            if 'f' not in predict_result[tomo_id]:
                predict_result[tomo_id]['f']={'fxx':[0.106,0.106], 'fyy':[0.106,0.106], 'fzz':[0.106,0.106]}

            t,p = cube_map(predict_result[tomo_id]['bbox'], gt_dic[tomo_id]['bbox'], predict_result[tomo_id]['f'])
            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])
            all_aps = []
            for key in T.keys():
                ap = average_precision_score(T[key], P[key])
                if np.isnan(ap):
                    ap = 1.0
                #print('{} AP: {}'.format(key, ap))
                all_aps.append(ap)
            images_map.append(np.mean(np.array(all_aps)))
            print('mAP = {}'.format(np.mean(np.array(all_aps))))
            #print(T)
            #print(P)
    print('All mAP = {}'.format(np.mean(np.mean(np.array(all_aps)))))
