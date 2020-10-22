import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math
import os
def data_extra(fname,f,model='train'):

    np.random.seed(20)

    file=np.load('3d-frcnn/sim-data-'+fname+'.npy', allow_pickle=True)
    num = len(file)

    for file_no in range(num):
        item = file[file_no]
        num_roi = len(item['bbox'])
        img = item['img']
        new_name = fname+'_'+str(file_no).zfill(4)

        if model == 'train':
            if not(os.path.exists('train-sim-old')):
                os.mkdir('train-sim-old')
            name = 'train-sim-old/'+new_name+'.npy'
        elif model=='test':
            if not(os.path.exists('test-sim-old')):
                os.mkdir('test-sim-old')
            name = 'test-sim-old/'+new_name+'.npy'
        
        np.save(name, img)

        for roi_no in range(num_roi):
            label = item['bbox'][roi_no][4]
            bbox = item['bbox'][roi_no][:4]
            f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[1], bbox[2], bbox[3], label))


if __name__ == '__main__':
    # ids = ['0b645491-d8b5-428c-b271-f0c4461ddba3', '0dc06161-58bb-4426-a2ab-9729b8de50ff',
    #        '100c050d-8fd2-48f1-925f-848f0756b3d8', '12f5554c-186f-438a-8ef2-1cff970c4fe7',
    #        '1696470a-4edc-4db6-89d3-ce620bb2bfc3', '1ce3a81a-3c29-4259-9263-a4200cc2b805',
    #        '2b0d1915-3744-4421-8310-76085fb467b0', '44c816da-97e0-4f82-9387-0456350fb2a1',
    #        '6e83b58f-7446-4d94-81da-ab17c093f47d', 'b6e24c7e-cea6-44c4-8bed-d907520f1567']
    ids = os.listdir('data/tomogram')[:10]
    f_train=open('label_train-old.txt','w')
    f_test=open('label_test-old.txt','w')
    for name in ids[:8]:
        data_extra(name[:-4],f_train, model='train')
    for name in ids[8:]:
        data_extra(name[:-4],f_test, model='test')
    f_train.close()
    f_test.close()


