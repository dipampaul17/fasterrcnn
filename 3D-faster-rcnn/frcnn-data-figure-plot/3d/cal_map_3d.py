import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
from sklearn.metrics import average_precision_score
from pprint import pprint


def read_gt_file(fname):
    """Read gt file, return bboxs in images"""
    f = open(fname, 'r')

    all_imgs = {}

    for line in f:
        line_split = line.strip().split(',')
        (tomo_file, x1, x2, x3, r, class_name) = line_split
        # tomo e.g. str '0145'
        if tomo_file not in all_imgs:
            all_imgs[tomo_file] = {}
            all_imgs[tomo_file]['bbox'] = []

        all_imgs[tomo_file]['bbox'].append(
            {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'x3': int(float(x3)), 'r': int(float(r))})

    return all_imgs


def read_pred_file(fname):
    """Read pred 3d file, return all imgs"""
    f = open(fname, 'r')

    all_imgs = {}

    for line in f:
        line_split = line.strip().split(',')
        # (tomo,x1,y1,z1,x2,y2,z2,class_name) = line_split
        (tomo_file, x1, x2, x3, r, class_name, probs, fx1, fx2, fx3) = line_split
        # tomo e.g. str '0145'
        if tomo_file not in all_imgs:
            all_imgs[tomo_file] = {}
            all_imgs[tomo_file]['bbox'] = []

        # fsize 目前不用加
        all_imgs[tomo_file]['bbox'].append(
            {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'x3': int(float(x3)), 'r': int(float(r)),
             'prob': float(probs)})

    return all_imgs


def union3d(au, bu, area_intersection):
    area_a = (au[3] - au[0]) * (au[4] - au[1]) * (au[5] - au[2])
    area_b = (bu[3] - bu[0]) * (bu[4] - bu[1]) * (bu[5] - bu[2])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection3d(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    z = max(ai[2], bi[2])
    w = min(ai[3], bi[3]) - x
    h = min(ai[4], bi[4]) - y
    d = min(ai[5], bi[5]) - z
    if w < 0 or h < 0 or d < 0:
        return 0
    return w * h * d


def iou_r(a, b):
    # a and b should be (x1,x2,x3,r)
    a_x1_min = a[0] - a[3]
    a_x1_max = a[0] + a[3]
    a_x2_min = a[1] - a[3]
    a_x2_max = a[1] + a[3]
    a_x3_min = a[2] - a[3]
    a_x3_max = a[2] + a[3]

    x = [a_x1_min, a_x2_min, a_x3_min, a_x1_max, a_x2_max, a_x3_max]

    b_x1_min = b[0] - b[3]
    b_x1_max = b[0] + b[3]
    b_x2_min = b[1] - b[3]
    b_x2_max = b[1] + b[3]
    b_x3_min = b[2] - b[3]
    b_x3_max = b[2] + b[3]

    y = [b_x1_min, b_x2_min, b_x3_min, b_x1_max, b_x2_max, b_x3_max]

    area_i = intersection3d(x, y)
    area_u = union3d(x, y, area_i)

    return float(area_i) / float(area_u + 1e-6)


def cube_map(pred, gt, f):
    T = {}
    P = {}
    fx1, fx2, fx3 = f.values()

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_x3 = pred_box['x3']
        pred_r = pred_box['r']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx1
            gt_x2 = gt_box['x2'] / fx2
            gt_x3 = gt_box['x3'] / fx3
            gt_r = gt_box['r'] / fx1
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = iou_r((pred_x1, pred_x2, pred_x3, pred_r), (gt_x1, gt_x2, gt_x3, gt_r))
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
            #  这个地方?
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
                pred[tomo_id]['f'] = {'fx1': 1, 'fx2': 1, 'fx3': 1}

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
                if ap == 1:
                    print(key, T[key], P[key])

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
    print("image mean ap:")
    print(img2map)

    class2map = {}
    for cls, cls_ap in class2aps.items():
        class2map[cls] = np.mean(np.array(cls_ap))
        # if class2map[cls] == 1:
        #     print(cls_ap)

    print("class mean ap:")
    pprint(class2map)

    print("All mAp = {}".format(np.mean(np.array(img2map))))


def main():
    pred_file = "3Dresults.txt"
    gt_file = "label_test-sim.txt"

    pred_res = read_pred_file(pred_file)
    gt_res = read_gt_file(gt_file)

    cal_map(pred_res, gt_res)


if __name__ == '__main__':
    main()
