import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import convolve
from skimage.transform import resize
from cv2 import circle
from skimage.filters import sobel_h, sobel_v
import os
from util import *

def Q1_1():
    depth_images_path = ["Part1/1/depthImage.png", "Part1/2/depthImage.png", "Part1/3/depthImage.png"]
    rgb_images_path = ["Part1/1/rgbImage.jpg", "Part1/2/rgbImage.jpg", "Part1/3/rgbImage.jpg"]
    intrinsic_path = ["Part1/1/intrinsics.txt", "Part1/2/intrinsics.txt", "Part1/2/intrinsics.txt"]
    extrinsic_path = ["Part1/1/extrinsic.txt", "Part1/2/extrinsic.txt", "Part1/3/extrinsic.txt"]
    output_name = ["Part1/output/output_0.txt", "Part1/output/output_1.txt", "Part1/output/output_2.txt"]
    imgs = []
    matrixes = []
    output = []
    for i in range(0, 3):
        img_depth = image.imread(depth_images_path[i])
        img_rgb = image.imread(rgb_images_path[i])
        matrixes.append([get_matrix_from_text(intrinsic_path[i]), get_matrix_from_text(extrinsic_path[i])])
        imgs.append([img_depth, img_rgb])
    for i in range(0, 3):
        output_i = []
        K = matrixes[i][0]
        R_It = matrixes[i][1]
        depth_img = imgs[i][0]
        rgb_img = imgs[i][1]
        K_inv = np.linalg.inv(K)
        R_It_full = np.concatenate((R_It, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0)
        R_It_full_inv = np.linalg.inv(R_It_full)
        # print(trans @ trans_inv)

        for x in range(0, rgb_img.shape[1]):
            for y in range(0, rgb_img.shape[0]):
                if depth_img[y, x] != 0:
                    q = np.array([x, y ,1]).reshape((3,1))
                    Q_over_w = K_inv @ q # compute scaled camera coordinate
                    w = depth_img[y, x]/Q_over_w[2,0] # obtain w using depth information
                    q = q*w # factor q by
                    Q_c = K_inv @ q # camera coordinate
                    Q_c_4 = np.append(Q_c, np.array([1])) # append 1 at the end
                    Q_w = (R_It_full_inv @ Q_c_4)[:3] # world coordinate
                    point = [Q_w[0], Q_w[1], Q_w[2], rgb_img[y, x, 0], rgb_img[y, x, 1], rgb_img[y, x, 2]]
                    output_i.append(point)
        print("file " + str(i) + " completed!")
        output.append(output_i)
    for i in range(0, 3):
        write_matrix_to_txt_file(output_name[i], output[i])
def Q1_2():
    rotation_in_rad = 0.25 * np.pi

    depth_images_path = ["Part1/1/depthImage.png", "Part1/2/depthImage.png", "Part1/3/depthImage.png"]
    rgb_images_path = ["Part1/1/rgbImage.jpg", "Part1/2/rgbImage.jpg", "Part1/3/rgbImage.jpg"]
    intrinsic_path = ["Part1/1/intrinsics.txt", "Part1/2/intrinsics.txt", "Part1/2/intrinsics.txt"]
    extrinsic_path = ["Part1/1/extrinsic.txt", "Part1/2/extrinsic.txt", "Part1/3/extrinsic.txt"]
    output_name = ["Part1/output/output_0.txt", "Part1/output/output_1.txt", "Part1/output/output_2.txt"]
    imgs = []
    matrixes = []
    for i in range(0, 3):
        img_depth = image.imread(depth_images_path[i])
        img_rgb = image.imread(rgb_images_path[i])
        matrixes.append([get_matrix_from_text(intrinsic_path[i]), get_matrix_from_text(extrinsic_path[i])])
        imgs.append([img_depth, img_rgb])
    for i in range(0, 3):
        rot = get_rotation_matrixes('z', rotation_in_rad)
        rot_inv = np.linalg.inv(rot)
        K = matrixes[i][0]
        R_It = matrixes[i][1]
        depth_img = imgs[i][0]
        rgb_img = imgs[i][1]
        K_inv = np.linalg.inv(K)
        R_It_full = np.concatenate((R_It, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0)
        R_It_full_inv = np.linalg.inv(R_It_full)
        padded_img = np.zeros((rgb_img.shape[0] * 5, rgb_img.shape[1] * 5, rgb_img.shape[2]))
        padded_img_depth = np.zeros((rgb_img.shape[0] * 5, rgb_img.shape[1] * 5))
        for x in range(0, rgb_img.shape[1]):
            for y in range(0, rgb_img.shape[0]):
                if depth_img[y, x] != 0:
                    q = np.array([x, y, 1]).reshape((3, 1))
                    Q_over_w = K_inv @ q  # compute scaled camera coordinate
                    w = depth_img[y, x] / Q_over_w[2, 0]  # obtain w using depth information
                    q = q * w  # factor q by
                    q_2 = K @ rot_inv @ K_inv @ q
                    padded_img[
                        int(q_2[1] / q_2[2]) + 2 * rgb_img.shape[0], int(q_2[0] / q_2[2]) + 2 * rgb_img.shape[1]] = \
                    rgb_img[y, x] / 255
                    padded_img_depth[int(q_2[1]/q_2[2])+2*rgb_img.shape[0], int(q_2[0]/q_2[2])+2*rgb_img.shape[1]] = depth_img[y, x]/255
def Q1_3():
    total_rotation_in_rad = 0.22 * np.pi
    count = 20
    dir = 'z' # <========================================== Change this
    rotations_in_rad = [0]
    for i in range(0, count):
        rotations_in_rad.append(total_rotation_in_rad/count*(i+1))
    depth_images_path = ["Part1/1/depthImage.png", "Part1/2/depthImage.png", "Part1/3/depthImage.png"]
    rgb_images_path = ["Part1/1/rgbImage.jpg", "Part1/2/rgbImage.jpg", "Part1/3/rgbImage.jpg"]
    intrinsic_path = ["Part1/1/intrinsics.txt", "Part1/2/intrinsics.txt", "Part1/2/intrinsics.txt"]
    extrinsic_path = ["Part1/1/extrinsic.txt", "Part1/2/extrinsic.txt", "Part1/3/extrinsic.txt"]
    output_path = "Part1/img_output3/frame{}.jpeg" # <========================================== Change this
    imgs = []
    matrixes = []
    output = []
    for i in range(0, 1):
        img_depth = image.imread(depth_images_path[i])
        img_rgb = image.imread(rgb_images_path[i])
        matrixes.append([get_matrix_from_text(intrinsic_path[i]), get_matrix_from_text(extrinsic_path[i])])
        imgs.append([img_depth, img_rgb])
    for i in range(0, 1):
        output_i = []
        K = matrixes[i][0]
        R_It = matrixes[i][1]
        depth_img = imgs[i][0]
        rgb_img = imgs[i][1]
        K_inv = np.linalg.inv(K)
        R_It_full = np.concatenate((R_It, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0)
        R_It_full_inv = np.linalg.inv(R_It_full)
        padded_img = np.zeros((count+1, rgb_img.shape[0] * 5, rgb_img.shape[1] * 5, rgb_img.shape[2]))
        for x in range(0, rgb_img.shape[1]):
            for y in range(0, rgb_img.shape[0]):
                if depth_img[y, x] != 0:
                    j = 0
                    for deg in rotations_in_rad:
                        q = np.array([x, y, 1]).reshape((3, 1))
                        Q_over_w = K_inv @ q  # compute scaled camera coordinate
                        w = depth_img[y, x] / Q_over_w[2, 0]  # obtain w using depth information
                        q = q * w  # factor q by
                        rot = get_rotation_matrixes(dir, deg)
                        rot_inv = np.linalg.inv(rot)
                        q_2 = K @ rot_inv @ K_inv @ q
                        padded_img[j,
                            int(q_2[1] / q_2[2]) + 2 * rgb_img.shape[0], int(q_2[0] / q_2[2]) + 2 * rgb_img.shape[1]] = rgb_img[y, x] / 255
                        j = j + 1
    for j in range(count+1):
        plt.imshow(padded_img[j])
        plt.savefig(output_path.format(j))
    generate_gif("Part1/img_output3/frame{}.jpeg", count, "Part1/img_output3") # <========================================== Change this

if __name__ == "__main__":
    pass






