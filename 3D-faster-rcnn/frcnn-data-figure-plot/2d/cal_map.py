import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
from sklearn.metrics import average_precision_score
from pprint import pprint

def read_gt_file(fname='real-3D-annotation-test.txt'):
    """Read gt file, return bboxs in images"""
    f = open(fname, 'r')

    all_imgs = {}

    for line in f:
        line_split = line.strip().split(',')
        (tomo, z1, y1, x1, z2, y2, x2, class_name) = line_split
        # tomo e.g. str '0145'
        if tomo not in all_imgs:
            all_imgs[tomo] = {}
            all_imgs[tomo]['bbox'] = []

        all_imgs[tomo]['bbox'].append(
            {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
             'y2': int(float(y2)), 'z1': int(float(z1)), 'z2': int(float(z2))})

    return all_imgs


def read_pred_file(fname):
    """Read pred 3d file, return all imgs"""
    f = open(fname, 'r')

    all_imgs = {}

    for line in f:
        line_split = line.strip().split(',')
        # (tomo,x1,y1,z1,x2,y2,z2,class_name) = line_split
        (tomo, z1, y1, x1, z2, y2, x2, class_name, probs, fsize) = line_split
        # tomo e.g. str '0145'
        if tomo not in all_imgs:
            all_imgs[tomo] = {}
            all_imgs[tomo]['bbox'] = []

        # TODO: fsize 好像不用加?
        all_imgs[tomo]['bbox'].append(
            {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
             'y2': int(float(y2)), 'z1': int(float(z1)), 'z2': int(float(z2)), 'probs': float(probs)})

    return all_imgs


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
    return w * h * d


def cube_iou(a, b):
    # a and b should be (x1,y1,z1,x2,y2,z2)
    if a[0] >= a[3] or a[1] >= a[4] or a[2] >= a[5] or b[0] >= b[3] or b[1] >= b[4] or b[2] >= b[5]:
        print('WARNING: indice mismatch')
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)


def cube_map(pred, gt, f):
    T = {}
    P = {}
    # fx, fy = f
    fxx, fxy = f['fxx']  # y,z
    fyx, fyy = f['fyy']  # x,z
    fzx, fzy = f['fzz']  # x,y

    ##
    if fxx != fzy or fxy != fyy or fyx != fzx:
        print('Warning: rescale size mismatch..')

    for bbox in gt:
        # print(bbox)
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
            # print(gt_box)
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fyx
            gt_x2 = gt_box['x2'] / fyx
            gt_y1 = gt_box['y1'] / fxx
            gt_y2 = gt_box['y2'] / fxx
            gt_z1 = gt_box['z1'] / fxy
            gt_z2 = gt_box['z2'] / fxy
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = cube_iou((pred_x1, pred_y1, pred_z1, pred_x2, pred_y2, pred_z2),
                           (gt_x1, gt_y1, gt_z1, gt_x2, gt_y2, gt_z2))
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched']:
            # if not gt_box['bbox_matched'] and not gt_box['difficult']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    # import pdb
    # pdb.set_trace()
    return T, P


def cal_map(pred, gt):
    """Calculate mean ap"""
    T = {}
    P = {}
    img2map = []
    class2aps = {}
    for tomo_id in pred:
        if tomo_id not in gt:
            print('WARNING: {} not in test set'.format(tomo_id))
        else:
            # print(pred[tomo_id])
            if 'f' not in pred[tomo_id]:
                pred[tomo_id]['f'] = {'fxx': [0.106, 0.106], 'fyy': [0.106, 0.106], 'fzz': [0.106, 0.106]}

            t, p = cube_map(pred[tomo_id]['bbox'], gt[tomo_id]['bbox'], pred[tomo_id]['f'])
            for key in t.keys():
                if key not in T:
                    T[key] = []
                    P[key] = []
                T[key].extend(t[key])
                P[key].extend(p[key])

            all_aps = []
            for key in T.keys():
                ap = average_precision_score(T[key], P[key])
                if np.isnan(ap):  # why not delete?
                    ap = 1.0
                # print('{} AP: {}'.format(key, ap))
                all_aps.append(ap)
                if key not in class2aps:
                    class2aps[key] = []
                class2aps[key].append(ap)

            img2map.append(np.mean(np.array(all_aps)))

            # print('tomo: {}, mAP = {}'.format(tomo_id, np.mean(np.array(all_aps))))

    # print('line_thres:{}'.format(line_threshold))
    # print(all_aps)
    # print('All mAP = {}'.format(np.mean(np.mean(np.array(all_aps)))))
    print(img2map)

    class2map = {}
    for cls, cls_ap in class2aps.items():
        class2map[cls] = np.mean(np.array(cls_ap))
    pprint(class2map)

    print("All mAp = {}".format(np.mean(np.array(img2map))))


def main():
    pred_file = "3Dresults.txt"
    gt_file = "sim-3d-annotation-test.txt"

    pred_res = read_pred_file(pred_file)
    gt_res = read_gt_file(gt_file)

    cal_map(pred_res, gt_res)


if __name__ == '__main__':
    main()
