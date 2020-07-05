assembly-faster-rcnn:

基于代码https://github.com/moyiliyi/keras-faster-rcnn

预处理阶段：
preprocess文件夹，先运行tomo_cut.py将大图切成小图，在用data_process-sim.py整理成目标格式

后处理/assemble bbox：
三种方法：直接3D bbox 做NMS：3D-bbox-nms.py
         沿一个方向assemble：One-direction.py
         Assembly Faster RCNN：nms-plane-3d-nms--request3direction.py
result_plot.py画图
2Dresult.txt：格式(filepath, x1, x2, y1, y2, key, new_probs[jk], fx, fy))

