import numpy as np
import matplotlib.pyplot as plt
import cv2
from imageio import imwrite


def get_real_coordinates(x1, y1, r, ratio=1):
    """x1, y1, r -> (x1, y1, x2, y2)"""
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round((x1 + 2 * r) // ratio))
    real_y2 = int(round((y1 + 2 * r) // ratio))
    return real_x1, real_y1, real_x2, real_y2


def get_cut_img(img_npy_path="test-sim-volumes/6e83b58f-7446-4d94-81da-ab17c093f47d_0000.npy"):
    """npy -> 切片"""
    img = np.load(img_npy_path)
    img = img[0, :, :]
    cv2.imwrite("example.png", img)
    # imsave('./example.png', img)
    # img = img[:, :, 0]
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def read_bboxes(gt_fname, pred_fname):
    gt_bboxes = {}
    pred_bboxes = {}

    with open(gt_fname) as gt_f:
        for line in gt_f:
            tomo_file, x1, x2, x3, r, class_name = line.strip().split(',')
            if tomo_file not in gt_bboxes:
                gt_bboxes[tomo_file] = []
            gt_bboxes[tomo_file].append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(r)), class_name])

    with open(pred_fname) as pred_f:
        for line in pred_f:
            tomo_file, x1, x2, x3, r, class_name, probs, fx1, fx2, fx3 = line.strip().split(',')
            # tomo_file = tomo_file.strip('.npy')
            if tomo_file not in pred_bboxes:
                pred_bboxes[tomo_file] = []
            pred_bboxes[tomo_file].append([int(float(x1)), int(float(x2)), int(float(x3)), int(float(r)), class_name])

    return gt_bboxes, pred_bboxes


# import pysnooper
# @pysnooper.snoop()
def main():
    direction = 'z'
    # img = get_cut_img()
    # cv2.imwrite("example.png", img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    gt_bboxes_path = "./label_test-sim.txt"
    pred_bboxes_path = "./3Dresults.txt"
    tomo_plot_example = "6e83b58f-7446-4d94-81da-ab17c093f47d_0000.npy"

    gt_bboxes, pred_bboxes = read_bboxes(gt_bboxes_path, pred_bboxes_path)
    if tomo_plot_example not in gt_bboxes:
        print("Error: tomo {} not in gt bboxes".format(tomo_plot_example))
        return
    # 没有检测到的就没必要画了?
    if tomo_plot_example not in pred_bboxes:
        print("Warning: tomo {} not in pred bboxes".format(tomo_plot_example))
        return
    plot_gt_bboxes, plot_pred_bboxes = gt_bboxes[tomo_plot_example], pred_bboxes[tomo_plot_example]

    img = cv2.imread("example.png")
    # plt.imshow(img, cmap='gray')
    # plt.show()
    for x1, x2, x3, r, class_name in plot_gt_bboxes:
        gtx1, gty1, gtx2, gty2 = get_real_coordinates(x2, x3, r)
        cv2.rectangle(img, (int(round(gty1)), int(round(gtx1))), (int(round(gty2)), int(round(gtx2))), (0, 0, 255), 1)
    for x1, x2, x3, r, class_name in plot_pred_bboxes:
        real_x1, real_y1, real_x2, real_y2 = get_real_coordinates(x2, x3, r)
        cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (255, 0, 0), 1)

    # # plt.imshow(img[int(round(gtx1)):int(round(gtx2)), int(round(gtz1)):int(round(gtz2))],cmap='gray')
    # # plt.imshow(img[int(round(gtz1)):int(round(gtz2)), int(round(gtx1)):int(round(gtx2))],cmap='gray')
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # plt.savefig("example-tagged.png")
    # cv2.imwrite("./example-tagged.png")
    imwrite('./example-tagged.png', img)


if __name__ == '__main__':
    main()
