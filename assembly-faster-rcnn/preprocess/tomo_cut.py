import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
import json

cut_step = 32
discard_thres = 0.3


def tomo_cut(fname):
    img = np.load(fname + '.npy')
    params = json.load(open(fname + '.json'))['0']['instances']

    num_roi = len(params)
    roi_maker = np.zeros(num_roi)

    print 'image.size:', img.shape

    num_cut = np.zeros(3)
    ind = []
    for i in range(3):
        shape = img.shape[i]
        if (shape & shape - 1) != 0:
            print('Warning:::Only work for exp2')
        elif shape % 2 != 0:
            print('Warning:::Only work for exp2')

        num_cut[i] = len(range(0, img.shape[i], cut_step))
        ind.append(range(0, img.shape[i], cut_step))

    bbox_for_img = [[] for i in range(170000)]
    dis_roi = 0
    dis_rroi = 0
    for roi_no in range(num_roi):
        if params[roi_no]['x'][0] < discard_thres * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][1] < discard_thres * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][2] < discard_thres * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][0] > img.shape[0] + (1 - discard_thres) * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][1] > img.shape[1] + (1 - discard_thres) * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][2] > img.shape[2] + (1 - discard_thres) * params[roi_no]['radius']:
            dis_roi += 1
            continue

        label = params[roi_no]['type']
        bbox = np.array([max(params[roi_no]['x'][0] - params[roi_no]['radius'], 0),
                         max(params[roi_no]['x'][1] - params[roi_no]['radius'], 0),
                         max(params[roi_no]['x'][2] - params[roi_no]['radius'], 0),
                         min(params[roi_no]['x'][0] + params[roi_no]['radius'], img.shape[0] - 1),
                         min(params[roi_no]['x'][1] + params[roi_no]['radius'], img.shape[1] - 1),
                         min(params[roi_no]['x'][2] + params[roi_no]['radius'],
                             img.shape[2] - 1)])  # x1,y1,z1, x2,y2,z2
        r = params[roi_no]['radius']
        xind = int((bbox[3] + bbox[0]) / (2*cut_step))
        yind = int((bbox[4] + bbox[1]) / (2*cut_step))
        zind = int((bbox[5] + bbox[2]) / (2*cut_step))
        ii = xind * cut_step
        jj = yind * cut_step
        kk = zind * cut_step
        new_bbox = np.array([max(params[roi_no]['x'][0] - params[roi_no]['radius'] - ii, 0),
                             max(params[roi_no]['x'][1] - params[roi_no]['radius'] - jj, 0),
                             max(params[roi_no]['x'][2] - params[roi_no]['radius'] - kk, 0),
                             min(params[roi_no]['x'][0] + params[roi_no]['radius'] - ii, (cut_step - 1)),
                             min(params[roi_no]['x'][1] + params[roi_no]['radius'] - jj, (cut_step - 1)),
                             min(params[roi_no]['x'][2] + params[roi_no]['radius'] - kk, (cut_step - 1)),
                             label])  # x1,y1,z1, x2,y2,z2
        if params[roi_no]['x'][0] - ii < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][1] - jj < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][2] - kk < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][0] - ii > (cut_step - 1) + (1 - discard_thres) * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][1] - jj > (cut_step - 1) + (1 - discard_thres) * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][2] - kk > (cut_step - 1) + (1 - discard_thres) * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        bbox_for_img[xind * 10000 + yind * 100 + zind].append(new_bbox)

    print 'Discard RoI due to boundary:', dis_roi
    print 'Discard RoI due to cut:', dis_rroi

    img_out = []

    cc = 0
    for ii in ind[0]:
        for jj in ind[1]:
            for kk in ind[2]:
                ind_i = int(ii / cut_step)
                ind_j = int(jj / cut_step)
                ind_k = int(kk / cut_step)
                if len(bbox_for_img[ind_i * 10000 + ind_j * 100 + ind_k]) == 0:
                    cc += 1
                    continue
                new_vol = img[ii:ii + cut_step, jj:jj + cut_step, kk:kk + cut_step]
                dic = {}
                dic['img'] = new_vol
                dic['bbox'] = bbox_for_img[ind_i * 10000 + ind_j * 100 + ind_k]
                img_out.append(dic)
    print 'Discard vol:', cc
    print 'Valid vol:', len(img_out)

    np.save('sim-data' + fname + '.npy',np.array(img_out))


# name = '0b645491-d8b5-428c-b271-f0c4461ddba3'
# tomo_cut(name)

if __name__ == '__main__':
    ids = ['0b645491-d8b5-428c-b271-f0c4461ddba3', '0dc06161-58bb-4426-a2ab-9729b8de50ff',
           '100c050d-8fd2-48f1-925f-848f0756b3d8', '12f5554c-186f-438a-8ef2-1cff970c4fe7',
           '1696470a-4edc-4db6-89d3-ce620bb2bfc3', '1ce3a81a-3c29-4259-9263-a4200cc2b805',
           '2b0d1915-3744-4421-8310-76085fb467b0', '44c816da-97e0-4f82-9387-0456350fb2a1',
           '6e83b58f-7446-4d94-81da-ab17c093f47d', 'b6e24c7e-cea6-44c4-8bed-d907520f1567']
    for name in ids:
        tomo_cut(name)

'''
Each is a (512, 512, 128) tomogram, contain 1000 roi

b=json.load(open(name+'.json'))
>>> b.keys()
[u'0']
>>> b['0'].keys()
[u'random_seed', u'pack inside box num', u'pack score', u'instances', u'pack temprature', u'id', u'uuid']
>>> len(b['0'][u'instances'])
1000
>>> b['0'][u'instances'][0]
{u'angle': [0.6065705425295175, 0.18082977604021683, 3.9710069711144427], u'uuid': u'769d5d91-685b-4529-9700-8ce620febafa', u'pdb_id': u'1EQR', u'diffusion_rate': 0.5, u'radius': 11.43153532995459, u'x': [457, 105, 45], u'mass': 1.0, u'type': 4, u'id': 0}

'''