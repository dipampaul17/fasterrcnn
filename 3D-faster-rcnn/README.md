# Updates:
Finished training code

# 3D Notes:

* ['bboxes'] is [x1,x2,x3,r]. **x1, x2, x3 are depth, heighth, width. Be careful about the order.**
* Size:
    - input image size: 48\*48\*48, and 64\*64\*64 (two datasets)
    - resize to 600\*600\*600 in parser input
    - Anchor size 10,15,20,25 resize -> 125, 188, 250, 312 / 94, 140, 188, 235.
        We set anchor scales as [128, 192, 256].  (NOTE: large number of anchors in 3D)
    - 600/4=150. 150\*150\*150 after feature extraction
* **feature extraction network changed to DSRF3D-v2, different from the proposal**
* Anchor info defined in config(To be changed) 
* In data_generators.py:cal_rpn: turn off some of the negative anchor regions, and limit it to 512 regions.
* Allows the object outside the boundary, but the center point must inside. ROIpooling only pool the valid area

# Reqirements

opencv=3.4.2 scipy=1.2.1 scikit-image=0.14.2 scikit-learn=0.20.3 tensorflow-gpu=1.9.0 numpy=1.16.2 matplotlib=2.2.3 keras-gpu=2.2.4
(mrcfile=1.1.2)


# 2D-keras-frcnn Readme
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

USAGE:
- train_frcnn.py can be used to train a model. To train on Pascal VOC data, simply do:
python train_frcnn.py /path/to/pascalvoc/data/
- the Pascal VOC data set (images and annotations for bounding boxes around the classified objects) can be obtained from: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

- simple_parser.py provides an alternative way to input data, using a text file. Simply provide a text file, with each
line containing:
filepath,x1,y1,x2,y2,class_name
For example:
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat

- test_frcnn.py can be used to perform inference, given pretrained weights. Specify a path to the folder containing
images:
python test_frcnn.py /path/to/imgs/

NOTES:
config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].

Example output:

![ex1](http://i.imgur.com/UtGXhtd.jpg)
![ex2](http://i.imgur.com/Szf78o2.jpg)
![ex3](http://i.imgur.com/OjVXTbn.jpg)
![ex4](http://i.imgur.com/9Fbe2Ow.jpg)
