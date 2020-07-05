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
#States:
# Training preiod selected distinguish images, => so the detector are not good at predicting edge, 
#                                                 i.e., possible one roi for each 3D-roi, NMS along direction are not so useful
# Soultions:
#    After NMS along directions, assemble
#                                steps: valid indice overlap > threshold (0.7 ?), non-valid indice inside valid indice
#                                       select the valid indice with high probs
#    Then 3D NMS?
#####

result_file = '2Dresults.txt'
#label_file = '../label_test.txt'
gt_file = 'sim-3D-annotation-test.txt'

overlap_thresh=0.5
dis_threshold=7
line_threshold = 0.6
thres_3nms=0.5
'''
steps:
2D assemble -> 3D
find corresponding gt
average label in 3 dimentions, compare with 3D => get mAP
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

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # return only the bounding boxes that were picked using the integer data type
    boxes = np.array([x1,x2,y1,y2]).astype("int") #boxes[0] is an array of x1
    
    return boxes, probs, label,np.zeros(len(inds)), np.zeros(len(inds)),inds, (fx,fy)


def line_iou(x1, x2, y1, y2):
    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)
    area_i = max(min(x2,y2) - max(x1, y1),0)
    area_u = (x2-x1) + (y2-y1) - area_i
    return float(area_i) / float(area_u + 1e-6)

#Do NMS for 3 Directions,get precise results
def nms_for_cube(indice, probs,label,overlap_thresh=thres_3nms,max_boxes=300):
    x1,x2,y1,y2,z1,z2 = indice

    # if there are no boxes, return an empty list
    if len(x1) == 0:
        return []

    x1   = np.array(x1   )
    x2   = np.array(x2   )
    y1   = np.array(y1   )
    y2   = np.array(y2   )
    z1   = np.array(z1   )
    z2   = np.array(z2   )
    label= np.array(label)
    probs= np.array(probs)

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)
    np.testing.assert_array_less(z1, z2)
    
    if x1.dtype.kind == "i":
        x1 = x1.astype("float")
    if x2.dtype.kind == "i":
        x2 = x2.astype("float")
    if y1.dtype.kind == "i":
        y1 = y1.astype("float")
    if y2.dtype.kind == "i":
        y2 = y2.astype("float")
    if z1.dtype.kind == "i":
        z1 = z1.astype("float")
    if z2.dtype.kind == "i":
        z2 = z2.astype("float")

    # initialize the list of picked indexes 
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1) *(z2 - z1)

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
        zz1_int = np.maximum(z1[i], z1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])
        zz2_int = np.minimum(z2[i], z2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)
        dd_int = np.maximum(0, zz2_int - zz1_int)

        area_int = ww_int * hh_int * dd_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
        
        #idxs = np.delete(idxs, np.concatenate(([last],
        #    np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break   

    # return only the bounding boxes that were picked using the integer data type
    boxes = np.array([x1[pick],x2[pick],y1[pick],y2[pick],z1[pick],z2[pick]]).astype("int") #boxes[0] is an array of x1
    
    probs = probs[pick]
    label = label[pick]
    return boxes, probs, label

def assemble(x_info, y_info, z_info):
    xbbx, xprobs, xlabel, xmin_range, xmax_range, x_centre = x_info
    ybbx, yprobs, ylabel, ymin_range, ymax_range, y_centre = y_info
    zbbx, zprobs, zlabel, zmin_range, zmax_range, z_centre = z_info
    # boxes are [x1, x2 ,y1 ,y2, z1, z2]
    xboxes = np.concatenate([np.array([xmin_range,xmax_range]), xbbx]) # Uncertain ind: 0, 1
    yboxes = np.concatenate([ybbx[0:2], np.array([ymin_range,ymax_range]), ybbx[2:]]) # Uncertain ind: 2,3
    zboxes = np.concatenate([zbbx, np.array([zmin_range, zmax_range])]) # Uncertain ind: 4,5
    
    x1 = []; y1 = []; z1 = []; x2 =[]; y2=[]; z2=[]
    probs = []
    label = []
    # Assemble, use valid indice
    ### X-Y
    y_search_space = range(len(yboxes[0]))
    for xi in range(len(xboxes[0])):
        flag = 0
        for yj in y_search_space:
            ct1 = line_iou(xboxes[4][xi],xboxes[5][xi], yboxes[4][yj], yboxes[5][yj]) > line_threshold
            #ct2 = xboxes[0][xi] >= yboxes[0][yj] and xboxes[1][xi] <= yboxes[1][yj]
            #ct3 = yboxes[2][yj] >= xboxes[2][xi] and yboxes[3][yj] <= xboxes[3][xi]
            ct2 = x_centre[xi] >= yboxes[0][yj] and x_centre[xi] <= yboxes[1][yj]
            ct3 = y_centre[yj] >= xboxes[2][xi] and y_centre[yj] <= xboxes[3][xi]
            if ct1 and ct2 and ct3:
                flag = 1
                x1.append(yboxes[0][yj])
                x2.append(yboxes[1][yj])
                y1.append(xboxes[2][xi])
                y2.append(xboxes[3][xi])
                if xprobs[xi] > yprobs[yj]:
                    z1.append(xboxes[4][xi])
                    z2.append(xboxes[5][xi])
                    probs.append(xprobs[xi])
                    label.append(xlabel[xi])
                else:
                    z1.append(yboxes[4][yj])
                    z2.append(yboxes[5][yj])
                    probs.append(yprobs[yj])
                    label.append(ylabel[yj])
                #y_search_space.remove(yj)
                #break
        #if flag == 0:
        #    print('Warning: Unable to find boxes to assemble')
        
    ### X-Z
    z_search_space = range(len(zboxes[0]))
    for xi in range(len(xboxes[0])):
        flag = 0
        for zj in z_search_space:
            ct1 = line_iou(xboxes[2][xi],xboxes[3][xi], zboxes[2][zj], zboxes[3][zj]) > line_threshold
            ct2 = x_centre[xi] >= zboxes[0][zj] and x_centre[xi] <= zboxes[1][zj]
            ct3 = z_centre[zj] >= xboxes[4][xi] and z_centre[zj] <= xboxes[5][xi]
            if ct1 and ct2 and ct3:
                flag = 1
                x1.append(zboxes[0][zj])
                x2.append(zboxes[1][zj])
                z1.append(xboxes[4][xi])
                z2.append(xboxes[5][xi])
                if xprobs[xi] > zprobs[zj]:
                    y1.append(xboxes[2][xi])
                    y2.append(xboxes[3][xi])
                    probs.append(xprobs[xi])
                    label.append(xlabel[xi])
                else:
                    y1.append(zboxes[2][zj])
                    y2.append(zboxes[3][zj])
                    probs.append(zprobs[zj])
                    label.append(zlabel[zj])
                #z_search_space.remove(zj)
                #break
        #if flag == 0:
        #    print('Warning: Unable to find boxes to assemble')

    ### Y-Z
    z_search_space = range(len(zboxes[0]))
    for yi in range(len(yboxes[0])):
        flag = 0
        for zj in z_search_space:
            ct1 = line_iou(yboxes[0][yi],yboxes[1][yi], zboxes[0][zj], zboxes[1][zj]) > line_threshold
            ct2 = y_centre[yi] >= zboxes[2][zj] and y_centre[yi] <= zboxes[3][zj]
            ct3 = z_centre[zj] >= yboxes[4][yi] and z_centre[zj] <= yboxes[5][yi]
            if ct1 and ct2 and ct3:
                flag = 1
                y1.append(zboxes[2][zj])
                y2.append(zboxes[3][zj])
                z1.append(yboxes[4][yi])
                z2.append(yboxes[5][yi])
                if yprobs[yi] > zprobs[zj]:
                    x1.append(yboxes[0][yi])
                    x2.append(yboxes[1][yi])
                    probs.append(yprobs[yi])
                    label.append(ylabel[yi])
                else:
                    x1.append(zboxes[0][zj])
                    x2.append(zboxes[1][zj])
                    probs.append(zprobs[zj])
                    label.append(zlabel[zj])
                #z_search_space.remove(zj)
                #break
        #if flag == 0:
        #    print('Warning: Unable to find boxes to assemble')

    # NMS
    if len(label)> 0:
        ffboxes, ffprobs, fflabel = nms_for_cube((x1,x2,y1,y2,z1,z2),probs,label)
        #print('success')
    else:
        print('Warning: Unable to find boxes to assemble')
        return [],[],[]

    #x1   = np.array(x1); x2 = np.array(x2); y1 = np.array(y1); y2 = np.array(y2); z1=np.array(z1); z2=np.array(z2)
    #label= np.array(label)
    #probs= np.array(probs)

    #boxes = np.array([x1,x2,y1,y2,z1,z2]).astype("int")
    return ffboxes, ffprobs, fflabel

def assemble_process(all_imgs, thres1, thres2):
    res_tomos = {}

    for tomo in all_imgs:
        if tomo not in res_tomos:
            res_tomos[tomo] = {}
            res_tomos[tomo]['bbox']=[]
        
        option = []
        if 'x' in all_imgs[tomo]:
            option.append('x')
        if 'y' in all_imgs[tomo]:
            option.append('x')
        if 'z' in all_imgs[tomo]:
            option.append('z')
        if len(option) == 0:
            print('Warning: tomo {} contains no roi'.format(tomo))
            continue
        elif len(option) == 1:
            print('Warning: tomo {} contains only one direction'.format(tomo))
            continue
        elif len(option) == 2:
            print('Warning: tomo {} contains only two direction'.format(tomo))
            continue


        xboxes, xprobs, xlabel, xmin_range, xmax_range, x_ind,fxx = nms_for_plane(all_imgs[tomo]['x'], overlap_thresh=thres1, dis_threshold=thres2)
        yboxes, yprobs, ylabel, ymin_range, ymax_range, y_ind,fyy = nms_for_plane(all_imgs[tomo]['y'], overlap_thresh=thres1, dis_threshold=thres2)
        zboxes, zprobs, zlabel, zmin_range, zmax_range, z_ind,fzz = nms_for_plane(all_imgs[tomo]['z'], overlap_thresh=thres1, dis_threshold=thres2)
        
        #xboxes = [y1,y2,z1,z2]
        #yboxes = [x1,x2,z1,z2]
        xmin_range=(xmin_range/fyy[0]).astype("int"); xmax_range=(xmax_range/fyy[0]).astype("int"); x_ind = (x_ind/fyy[0]).astype("int")
        ymin_range=(ymin_range/fxx[0]).astype("int"); ymax_range=(ymax_range/fxx[0]).astype("int"); y_ind = (y_ind/fxx[0]).astype("int")
        zmin_range=(zmin_range/fxx[1]).astype("int"); zmax_range=(zmax_range/fxx[1]).astype("int"); z_ind = (z_ind/fxx[1]).astype("int")

        boxes, probs, label = assemble((xboxes, xprobs, xlabel, xmin_range, xmax_range,x_ind), (yboxes, yprobs, ylabel, ymin_range, ymax_range,y_ind), (zboxes, zprobs, zlabel, zmin_range, zmax_range, z_ind))
        
        for roi in range(len(probs)):
            res_tomos[tomo]['bbox'].append({'class': label[roi],'x1': boxes[0][roi], 'x2': boxes[1][roi], 'y1': boxes[2][roi], 'y2': boxes[3][roi],'z1':boxes[4][roi],'z2':boxes[5][roi], 'probs': probs[roi]})
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
    print('line_thres:{}'.format(line_threshold))
    print('All mAP = {}'.format(np.mean(np.mean(np.array(all_aps)))))
