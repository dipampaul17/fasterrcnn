# label_test-sim.txt，生成在code代码里
读取：

```
for line in f:
            line_split = line.strip().split(',')
            (filename,x1,x2,x3,r,class_name) = line_split
```

# 3Dresults.txt是识别到的3D结果
格式：

```
'{},{},{},{},{},{},{},{},{},{}\n'.format(filepath, x1, x2, x3, r, label, new_probs, fx1, fx2, fx3)
```

fx是放缩因子



#参考mAP代码(measure_map.py)

```
    t, p = get_map(all_dets, img_data['bboxes'], (fx1, fx2, fx3))
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        #print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    images_map.append(np.mean(np.array(all_aps)))
    #print('mAP = {}'.format(np.mean(np.array(all_aps))))
    #print(T)
    #print(P)
print('All mAP = {}'.format(np.mean(np.array(images_map))))
f_rec.close()

```


```
def get_map(pred, gt, f):
    T = {}
    P = {}
    fx1, fx2, fx3 = f

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
            gt_x1 = gt_box['x1']/fx1
            gt_x2 = gt_box['x2']/fx2
            gt_x3 = gt_box['x3']/fx3
            gt_r = gt_box['r']/fx1
            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue
            iou = data_generators.iou_r((pred_x1, pred_x2, pred_x3, pred_r), (gt_x1, gt_x2, gt_x3, gt_r))
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
```
