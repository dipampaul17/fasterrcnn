import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math

def data_extra(fname,f,model='train'):

    np.random.seed(20)

    file=np.load('sim-data-'+fname+'.npy')
    num = len(file)

    for file_no in range(num):
        item = file[file_no]
        num_roi = len(item['bbox'])
        img = item['img']
        new_name = fname+'_'+str(file_no).zfill(4)
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
    ids = ['0b645491-d8b5-428c-b271-f0c4461ddba3', '0dc06161-58bb-4426-a2ab-9729b8de50ff',
           '100c050d-8fd2-48f1-925f-848f0756b3d8', '12f5554c-186f-438a-8ef2-1cff970c4fe7',
           '1696470a-4edc-4db6-89d3-ce620bb2bfc3', '1ce3a81a-3c29-4259-9263-a4200cc2b805',
           '2b0d1915-3744-4421-8310-76085fb467b0', '44c816da-97e0-4f82-9387-0456350fb2a1',
           '6e83b58f-7446-4d94-81da-ab17c093f47d', 'b6e24c7e-cea6-44c4-8bed-d907520f1567']
    f_train=open('label_train-sim.txt','w')
    f_test=open('label_test-sim.txt','w')
                                                                                                                             1,1           Top
    for name in ids[:8]:
        data_extra(name,f_train, model='train')
    for name in ids[8:]:
        data_extra(name,f_test, model='test')
    f_train.close()
    f_test.close()
