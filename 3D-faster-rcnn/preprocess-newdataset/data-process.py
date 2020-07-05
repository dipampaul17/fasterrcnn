import numpy as np
from skimage import io
import math
import os

def data_extra(fname,f,iind,model='train'):
    np.random.seed(20)
    file=np.load('new-size/sim-data-init-'+str(iind)+'-'+fname+'.npy')
    num = len(file)

    for file_no in range(num):
        item = file[file_no]
        num_roi = len(item['bbox'])
        img = item['img']
        new_name = 'cut'+str(iind)+'-'+fname+'_'+str(file_no).zfill(4)
        if model == 'train':
            name = 'train-sim-volumes/'+new_name+'.npy'
        elif model=='test':
            name = 'test-sim-volumes/'+new_name+'.npy'
        
        np.save(name, img)

        for roi_no in range(num_roi):
            label = item['bbox'][roi_no][4]
            bbox = item['bbox'][roi_no][:4]
            f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[1], bbox[2], bbox[3], label))


if __name__ == '__main__':
    ids = os.listdir('tomogram')
    f_train=open('label_train-sim.txt','w')
    f_test=open('label_test-sim.txt','w')
    for name in ids[:70]:
        for kkind in range(0,64,4):
            data_extra(name[:-4],f_train, kkind, model='train')
    for name in ids[70:]:
        for kkind in range(0,64,4):
            data_extra(name[:-4],f_test, kkind, model='test')
    f_train.close()
    f_test.close()


