import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import math

def data_extra(fname,f,f_res,model='train'):

    np.random.seed(20)

    file=np.load('sim-data-'+fname+'.npy')
    num = len(file)

    

    for file_no in range(num):
        item = file[file_no]
        num_roi = len(item['bbox'])
        img = item['img']
        new_name = fname+'_'+str(file_no).zfill(4)
        for roi_no in range(num_roi):
            label = item['bbox'][roi_no][6]
            bbox = item['bbox'][roi_no][:6]
            x_len = int(math.ceil(bbox[3])) - int(bbox[0])
            y_len = int(math.ceil(bbox[4])) - int(bbox[1])
            z_len = int(math.ceil(bbox[5])) - int(bbox[2])
            
            cut_x = int(x_len*0.375)
            cut_y = int(y_len*0.375)
            cut_z = int(z_len*0.375)

            if model=='train':
                for xind in range(int(bbox[0])+cut_x,int(math.ceil(bbox[3]))-cut_x,2):
                    name = 'sim-images/'+new_name+'_roi0x_'+str(xind)+'.png'
                    plt.imsave(name,img[xind,:,:], cmap='gray')            
                    f.write('{},{},{},{},{},{}\n'.format(name, bbox[1], bbox[2], bbox[4], bbox[5], label))    

                for yind in range(int(bbox[1])+cut_y,int(math.ceil(bbox[4]))-cut_y, 2):
                    name = 'sim-images/'+ new_name+'_roi0y_'+str(yind)+'.png'
                    plt.imsave(name,img[:,yind,:], cmap='gray')
                    f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[2], bbox[3], bbox[5], label))    

                for zind in range(int(bbox[2])+cut_z,int(math.ceil(bbox[5]))-cut_z, 2):
                    name = 'sim-images/'+new_name+'_roi0z_'+str(zind)+'.png'
                    plt.imsave(name,img[:,:,zind], cmap='gray')
                    f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[1], bbox[3], bbox[4], label))
            elif model == 'test':
                f_res.write('{},{},{},{},{},{},{},{}\n'.format(new_name, bbox[0], bbox[1], bbox[2], bbox[3],bbox[4], bbox[5], label))

        if model=='test':    

            for xind in range(img.shape[0]):
                name = 'test-sim-images/'+new_name+'_roi0x_'+str(xind)+'.png'
                plt.imsave(name,img[xind,:,:], cmap='gray')               
                f.write('{},{},{},{},{},{}\n'.format(name, bbox[1], bbox[2], bbox[4], bbox[5], label))        

            for yind in range(img.shape[1]):
                name = 'test-sim-images/'+new_name+'_roi0y_'+str(yind)+'.png'
                plt.imsave(name,img[:,yind,:], cmap='gray')
                f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[2], bbox[3], bbox[5], label))        

            for zind in range(img.shape[2]):
                name = 'test-sim-images/'+new_name+'_roi0z_'+str(zind)+'.png'
                plt.imsave(name,img[:,:,zind], cmap='gray')
                f.write('{},{},{},{},{},{}\n'.format(name, bbox[0], bbox[1], bbox[3], bbox[4], label))





if __name__ == '__main__':
    ids = ['0b645491-d8b5-428c-b271-f0c4461ddba3', '0dc06161-58bb-4426-a2ab-9729b8de50ff',
           '100c050d-8fd2-48f1-925f-848f0756b3d8', '12f5554c-186f-438a-8ef2-1cff970c4fe7',
           '1696470a-4edc-4db6-89d3-ce620bb2bfc3', '1ce3a81a-3c29-4259-9263-a4200cc2b805',
           '2b0d1915-3744-4421-8310-76085fb467b0', '44c816da-97e0-4f82-9387-0456350fb2a1',
           '6e83b58f-7446-4d94-81da-ab17c093f47d', 'b6e24c7e-cea6-44c4-8bed-d907520f1567']
    f_train=open('label_train-sim.txt','w')
    f_test=open('label_test-sim.txt','w')
    f_res=open('sim-3D-annotation-test.txt','w')
    for name in ids[:8]:
        data_extra(name,f_train,f_res, model='train')
    for name in ids[8:]:
        data_extra(name,f_test, f_res, model='test')
    f_train.close()
    f_test.close()
    f_res.close()

'''
>>> len(a)
8000
>>> a[0]
{'bbox': array([[16.4693207 , 28.12710828,  5.95247082, 28.4693207 , 40.12710828,        25.95247082,  1.      ],
       [21.4693207 ,  5.12710828, 20.95247082, 33.4693207 , 17.12710828,        40.95247082,  1.        ]]), 'img': array([[[ 0.00460042,  0.00720354,  0.00293247, ...,  0.01813673,          0.01923667, 0.01348566],
                         ....
                    [ 0.00042046, -0.00117394, -0.00955305, ...,  0.00637516,          0.00344128,0.00162736]]], dtype=float32)}


>>> a[0]['bbox']   
array([[16.4693207 , 28.12710828,  5.95247082, 28.4693207 , 40.12710828,
        25.95247082,  1.        ],
       [21.4693207 ,  5.12710828, 20.95247082, 33.4693207 , 17.12710828,
        40.95247082,  1.        ]])
>>> a[1]['bbox']
array([[ 8.65041245, 18.84964251, 10.89288536, 20.65041245, 32.84964251,
        22.89288536, 10.        ],
       [28.65041245, 16.84964251, 20.89288536, 40.65041245, 28.84964251,
        34.89288536,  6.        ],
       [ 6.65041245,  8.84964251, 30.89288536, 18.65041245, 20.84964251,
        44.89288536,  4.        ],
       [21.65041245, 37.84964251, 12.89288536, 33.65041245, 47.84964251,
        24.89288536,  9.        ],
       [31.65041245, 35.84964251, 30.89288536, 43.65041245, 47.84964251,
        44.89288536,  6.        ],
       [ 2.65041245, 32.84964251, 24.89288536, 16.65041245, 44.84964251,
        42.89288536, 11.        ]])
>>> a[1]['img'].shape
(48, 48, 48)
>>> a[2]['img'].shape
(48, 48, 48)
'''