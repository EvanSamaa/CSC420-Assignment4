import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib import image
from scipy.ndimage import convolve
from skimage.transform import resize
from cv2 import circle
import cv2
from skimage.filters import sobel_h, sobel_v
import os
import imageio

def read_img_gs(fname):
    # this function reads a colored image with file name fname and returns
    # a grayscale image as an numpy ndarray with intensity of [0...1]
    img = image.imread(fname)
    img = img[:,:,0]*0.2989 + img[:,:,1]*0.5870 + img[:,:,2]*0.1140
    return img

def get_matrix_from_text(f_name):
    file = open(f_name, "r")
    text = file.read()
    rows = text.split("\n")
    matrix = []
    for row in rows:
        if row != "":
            matrix_row = []
            text_num = row.split()
            for num in text_num:
                matrix_row.append(float(num))
            matrix.append(matrix_row)
    output = np.array(matrix)
    return output
def write_matrix_to_txt_file(f_name, matrix):
    file = open(f_name, "w")
    for row in matrix:
        line = ""
        for i in range(len(row)-1):
            line = line + str(row[i]) + " "
        line = line + str(row[len(row)-1])
        file.write(line + "\n")
def display_3d_point_cloud(txt_file_name):
    matrix = get_matrix_from_text(txt_file_name)
    points = matrix[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points) # now they are point cloud objects
    point_cloud.colors = o3d.utility.Vector3dVector(matrix[:, 3:]/255)
    o3d.visualization.draw_geometries([point_cloud])

def get_rotation_matrixes(dir, rad):
    if dir == "y":
        matrix = [[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]
    elif dir == "x":
        matrix = [[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]
    elif dir == "z":
        matrix = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    return np.array(matrix)
def generate_gif(partial_name, count, output_dir):
    images = []
    for i in range(count):
        images.append(imageio.imread(partial_name.format(i)))
    for i in range(count, 0, -1):
        images.append(imageio.imread(partial_name.format(i)))
    imageio.mimsave(output_dir + "{}movie.gif".format(count), images)
def eight_point(img1, img2, k = 8):
    src_pts, dst_pts = generate_n_correspondance(img1, img2)
    if src_pts.shape[0] < 8:
        return False
    if k >= 0:
        index = np.random.choice(src_pts.shape[0], k, replace=False)
        src_pts_selected = src_pts[index, :]
        dst_pts_selected = dst_pts[index, :]
    else:
        k = src_pts.shape[0]
        src_pts_selected = src_pts
        dst_pts_selected = dst_pts
    A = (dst_pts_selected[:, 1] * src_pts_selected[:, 1]).reshape((k, 1))
    A = np.concatenate((A, (dst_pts_selected[:, 1] * src_pts_selected[:, 0]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (dst_pts_selected[:, 1]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (dst_pts_selected[:, 0] * src_pts_selected[:, 1]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (dst_pts_selected[:, 0] * src_pts_selected[:, 0]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (dst_pts_selected[:, 0]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (src_pts_selected[:, 1]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, (src_pts_selected[:, 0]).reshape((k, 1))), axis=1)
    A = np.concatenate((A, np.ones((k, 1))), axis=1)
    u, s, vh = np.linalg.svd(A)
    F = vh.T[:, -1]
    F = F.reshape((3, 3))
    u_f, s_f, vh_f = np.linalg.svd(F)
    s_mat_f = np.diag(s_f)
    s_mat_f[-1, -1] = 0
    F_prime = u_f @ s_mat_f @ vh_f
    return F_prime
def epipole_colatino(img1, img2):
    F = eight_point(img1, img2)
    u, s, vh = np.linalg.svd(F)
    e_l = u[:, -1]
    e_r = vh.T[:, -1]
    return e_l, e_r
def compute_F(img1, img2):
    #
    # cv_img1 = img1.reshape((img1.shape[0], img1.shape[1], 1))
    # cv_img2 = img2.reshape((img2.shape[0], img2.shape[1], 1))
    cv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    MIN_MATCH_COUNT = 10
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # find the keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(cv_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(cv_img2, None)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # ============================================================
    pts2 = []
    pts1 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(keypoints_2[m.trainIdx].pt)
            pts1.append(keypoints_1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    return F
def generate_n_correspondance(img1, img2):
    #
    # cv_img1 = img1.reshape((img1.shape[0], img1.shape[1], 1))
    # cv_img2 = img2.reshape((img2.shape[0], img2.shape[1], 1))
    cv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    MIN_MATCH_COUNT = 10
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # find the keypoints and descriptors with SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(cv_img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(cv_img2, None)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # ============================================================
    # pts2 = []
    # pts1 = []
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < 0.8 * n.distance:
    #         good.append(m)
    #         pts2.append(keypoints_2[m.trainIdx].pt)
    #         pts1.append(keypoints_1[m.queryIdx].pt)
    # pts1 = np.int32(pts1)
    # pts2 = np.int32(pts2)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # print(F)
    # ============================================================
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    src_pts = np.concatenate((np.array(src_pts[:, 1]).reshape(-1, 1), np.array(src_pts[:, 0]).reshape(-1, 1)), axis=1)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.concatenate((np.array(dst_pts[:, 1]).reshape(-1, 1), np.array(dst_pts[:, 0]).reshape(-1, 1)), axis=1)
    # these are in (y, x)
    return src_pts, dst_pts
def use_sift_to_find_points(pair=1):
    if pair == 1:
        img_l = image.imread("Part2/first_pair/p11.jpg")
        img_r = image.imread("Part2/first_pair/p12.jpg")
    else:
        img_l = image.imread("Part2/second_pair/p21.jpg")
        img_r = image.imread("Part2/second_pair/p22.jpg")
    src_pts, dst_pts = generate_n_correspondance(img_l, img_r)
    p_l = []
    p_r = []
    for i in range(src_pts.shape[0]):
        p_l.append([src_pts[i, 1], src_pts[i, 0]])
        p_r.append([dst_pts[i, 1], dst_pts[i, 0]])
    pairs = [p_l, p_r]
    import pickle
    if pair == 1:
        with open('./Part2/first_pair/pair_points.pkl', 'wb') as f:
            pickle.dump(pairs, f)
    else:
        with open('./Part2/second_pair/pair_points.pkl', 'wb') as f:
            pickle.dump(pairs, f)
def manually_obtain_points(img_l, count = 5):
    def draw_circle_l(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img_l, (x, y), 5, (0, 255, 0), -1)
            points_l.append([x, y])
    points_l = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle_l)
    while (1):
        cv2.imshow('image', img_l)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(points_l) >= count:
            break
    return points_l
def selet_points_and_save_as_pkl(pair=1, num_points=5):
    if pair == 1:
        img_l = image.imread("Part2/first_pair/p11.jpg")
        img_r = image.imread("Part2/first_pair/p12.jpg")
    else:
        img_l = image.imread("Part2/second_pair/p21.jpg")
        img_r = image.imread("Part2/second_pair/p22.jpg")
    p_l = manually_obtain_points(img_l, num_points)
    p_r = manually_obtain_points(img_r, num_points)
    pairs = [p_l, p_r]
    import pickle
    if pair == 1:
        with open('./Part2/first_pair/pair_points.pkl', 'wb') as f:
            pickle.dump(pairs, f)
    else:
        with open('./Part2/second_pair/pair_points.pkl', 'wb') as f:
            pickle.dump(pairs, f)
def obtain_pairs_from_pickle(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist
if __name__ == "__main__":
    img_l = image.imread("Part2/first_pair/p11.jpg")
    img_r = image.imread("Part2/first_pair/p12.jpg")
    use_sift_to_find_points(img_l, img_r)
    # selet_points_and_save_as_pkl(1, 12)
    # obtain_pairs_from_pickle("./Part2/first_pair/pair_points.pkl")
