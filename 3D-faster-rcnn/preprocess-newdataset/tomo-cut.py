import numpy as np
from skimage import io
import math
import os
import json

cut_step = 64
discard_thres = 0.7
#discard_thres=1 #1 means totally valid

def tomo_cut(fname, iind):
    img = np.load('tomogram/' + fname + '.npy')
    params = json.load(open('json/' + fname + '.json'))['instances']

    num_roi = len(params)
    roi_maker = np.zeros(num_roi)

    print ('image.size:', img.shape)
    print ('roi generate:', num_roi)

    num_cut = np.zeros(3)
    ind = []
    for i in range(3):
        shape = img.shape[i]
        '''
        if (shape & shape - 1) != 0:
            print('Warning:::Only work for exp2')
        elif shape % 2 != 0:
            print('Warning:::Only work for exp2')
        '''
        num_cut[i] = len(range(iind, img.shape[i], cut_step))
        ind.append(range(iind, img.shape[i], cut_step))

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
        elif params[roi_no]['x'][0] > img.shape[0] - discard_thres * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][1] > img.shape[1] - discard_thres * params[roi_no]['radius']:
            dis_roi += 1
            continue
        elif params[roi_no]['x'][2] > img.shape[2] - discard_thres * params[roi_no]['radius']:
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

        if params[roi_no]['x'][0] < iind or params[roi_no]['x'][1] < iind or params[roi_no]['x'][2] < iind:
            dis_rroi += 1
            continue


        xind = int((params[roi_no]['x'][0] - iind) / cut_step)
        yind = int((params[roi_no]['x'][1] - iind) / cut_step)
        zind = int((params[roi_no]['x'][2] - iind) / cut_step)
        ii = xind * cut_step
        jj = yind * cut_step
        kk = zind * cut_step
        new_bbox = np.array([params[roi_no]['x'][0] - ii - iind,
                             params[roi_no]['x'][1] - jj - iind,
                             params[roi_no]['x'][2] - kk - iind,
                             params[roi_no]['radius'],
                             label])  # x1,y1,z1,r
        if   params[roi_no]['x'][0] - ii - iind < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][1] - jj - iind < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][2] - kk - iind < discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][0] - ii - iind > (cut_step - 1) - discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][1] - jj - iind > (cut_step - 1) - discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        elif params[roi_no]['x'][2] - kk - iind > (cut_step - 1) - discard_thres * params[roi_no]['radius']:
            dis_rroi += 1
            continue
        bbox_for_img[xind * 10000 + yind * 100 + zind].append(new_bbox)

    print ('Discard RoI due to boundary:', dis_roi)
    print ('Discard RoI due to cut:', dis_rroi)

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
                if ii + cut_step > img.shape[0] or jj + cut_step > img.shape[1] or kk + cut_step > img.shape[2]:
                    continue
                new_vol = img[ii:ii + cut_step, jj:jj + cut_step, kk:kk + cut_step]
                dic = {}
                dic['img'] = new_vol
                dic['bbox'] = bbox_for_img[ind_i * 10000 + ind_j * 100 + ind_k]
                img_out.append(dic)
    print ('Discard vol:', cc)
    print ('Valid vol:', len(img_out))
    if not(os.path.exists('new-size')):
        os.mkdir('new-size')

    np.save('new-size/sim-data-init-'+str(iind)+'-' + fname + '.npy',np.array(img_out))


# name = '0b645491-d8b5-428c-b271-f0c4461ddba3'
# tomo_cut(name)

if __name__ == '__main__':
    ids = os.listdir('tomogram')[:10]
    for kkind in range(0,64,4):
        for name in ids:
            name = name[:-4]
            tomo_cut(name, iind=kkind)

'''
Each is a (400, 400, 200) tomogram, each contains 15,000 rois, around 11,000 out of boundary, around 1,500 are discard due to cut(cut-step = 100)
radius range from 17 - 30


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
