import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import convolve
from skimage.transform import resize
from cv2 import circle
from skimage.filters import sobel_h, sobel_v
import os
import open3d as o3d
from mpl_toolkits import mplot3d
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
    pair = 1
    # selet_points_and_save_as_pkl(pair=pair, num_points=12)
    # use_sift_to_find_points(pair=pair)
    if pair == 1:
        points = obtain_pairs_from_pickle("./Part2/first_pair/pair_points.pkl")
        img_l_gray = image.imread("Part2/first_pair/p11.jpg")
        img_r_gray = image.imread("Part2/first_pair/p12.jpg")
    else:
        points = obtain_pairs_from_pickle("./Part2/second_pair/pair_points.pkl")
        img_l_gray = image.imread("Part2/second_pair/p21.jpg")
        img_r_gray = image.imread("Part2/second_pair/p22.jpg")
    img_width = img_r_gray.shape[1]
    img_height = img_r_gray.shape[0]
    f = 35.0/1000
    sx = 0.00002611
    sy = 0.00002611
    M = np.array([[-f/sx, 0, 0], [0, -f/sy, 0], [0, 0, 1]])
    # M = np.array([[-f/sx, 0, 23.5/1000/2], [0, -f/sy, 15.6/1000/2], [0, 0, 1]])
    # approximate fundamental matrix swith 8 point algorithm
    F = eight_point(img_l_gray, img_r_gray, k=-1)
    # F = compute_F(img_l_gray, img_r_gray)
    print(F)
    # recover Essential matrix from Fundamental matrix
    E = M.T @ F @ M
    # solve for T
    EtE = E.T @ E
    eye = np.eye((3))
    # compute normalizing constant
    T_mag = np.sqrt(np.sum(eye*EtE)/2)
    # normalize EtE
    EtE_hat = (E.T/T_mag) @ (E/T_mag)
    # recovevr T
    T_x_hat = np.sqrt(1 - np.minimum(EtE_hat[0, 0], 1))
    T_y_hat = np.sqrt(1 - np.minimum(EtE_hat[1, 1], 1))
    T_z_hat = np.sqrt(1 - np.minimum(EtE_hat[2, 2], 1))
    T_hat = np.array([T_x_hat, T_y_hat, T_z_hat])
    # recover R
    w_0 = np.cross(E[0]/T_mag, T_hat)
    w_1 = np.cross(E[1]/T_mag, T_hat)
    w_2 = np.cross(E[2]/T_mag, T_hat)
    R = np.zeros((3, 3))
    R[0] = w_0 + np.cross(w_1, w_2)
    R[1] = w_1 + np.cross(w_2, w_0)
    R[2] = w_2 + np.cross(w_0, w_1)
    R, R2, T_hat = cv2.decomposeEssentialMat(E)
    T_hat = T_hat.reshape((3,))
    # begin computing for each point
    coords = []
    for i in range(0, len(points[0])):
        # solve for a0, b0, c0 using TRIANG
        p_l = points[0][i]
        p_r = points[1][i]
        p_l = np.array([p_l[0]-img_width/2, p_l[1]-img_height/2, f])
        p_r = np.array([p_r[0]-img_width/2, p_r[1]-img_height/2, f])
        # p_l = np.array([p_l[0], p_l[1], f])
        # p_r = np.array([p_r[0], p_r[1], f])
        p_l = p_l * np.array((23.5/1000/img_width, 15.6/1000/img_height, 1))
        p_r = p_r * np.array((23.5/1000/img_width, 15.6/1000/img_height, 1))
        system = np.zeros((3, 3))
        system[:, 0] = p_l
        system[:, 1] = -R.T @ p_r
        system[:, 2] = np.cross(p_l, R.T @ p_r)
        sol = np.linalg.inv(system) @ T_hat
        mid_point = (sol[0]*p_l + T_hat + sol[1]*R.T @ p_r)/2
        coords.append(mid_point)
    # plotting 3 rectangles
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(coords))  # now they are point cloud objects
    # point_cloud.colors = o3d.utility.Vector3dVector(matrix[:, 3:] / 255)
    o3d.visualization.draw_geometries([point_cloud])
    # for i in range(0, 3):
    #     rec_i = []
    #     for pt in range(0, 4):
    #         rec_i.append(coords[i + pt])
    #     rec_i = np.array(rec_i)








