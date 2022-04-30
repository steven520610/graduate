# %%
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
# read the image file & output_image the color & gray image


def read_img(path):
    # opencv read image in "BGR" color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img_rgb

# the dtype of img must be "uint8" to avoid the error of SIFT detector


def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ", img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def SIFT(img):
    SIFT_detector = cv2.SIFT_create()
    kp, des = SIFT_detector.detectAndCompute(img, None)
    return kp, des


def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()  # deep copy
    img = cv2.drawKeypoints(
        gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


def K_NN(kp1, des1, kp2, des2, k):
    k_nearest_neighbor = np.zeros((len(kp1), k), dtype=np.uint16)
    for i in range(len(kp1)):
        distance = np.zeros(len(kp2))
        source_feature = des1[i]
        for j in range(len(kp2)):
            target_feature = des2[j]
            distance[j] = np.linalg.norm(source_feature - target_feature)
        minimum_distance_index = np.argsort(distance)
        k_nearest_neighbor[i] = minimum_distance_index[:k]
    return k_nearest_neighbor


def Ratio_Test(kp1, des1, kp2, des2, nearest_neighbor, threshold):
    matches = []
    for i in range(len(kp1)):
        d1 = np.linalg.norm(des1[0] - des2[nearest_neighbor[i][0]])
        d2 = np.linalg.norm(des1[0] - des2[nearest_neighbor[i][1]])
        if d1 < threshold * d2:
            matches.append(list(kp1[i].pt + kp2[nearest_neighbor[i][0]].pt))
    matches = np.array(matches)
    return matches


"""
    def plot_matches(matches, concatenate_image):
        match_img = concatenate_image.copy()
        offset = concatenate_image.shape[1]/2
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(np.array(match_img).astype('uint8'),
                cmap="gray")  # 　RGB is integer type

        ax.plot(matches[:, 0], matches[:, 1], 'xr')
        ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')

        ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
                'r', linewidth=0.5)

        plt.show()
"""

"""
    # p2 = Hp1
    # (wx2, wy2 , w) = H(x1, y1, 1)
    # H = [
    #      [A11, A12, A13],
    #      [A21, A22, A23],
    #      [A31, A32, A33]
    #     ]
    # w = x1A31 + y1A32 + A33
    # wx2 = x2(x1A31 + y1A32 + A33) = x1A11 + y1A12 + A13
        # >> x1A11 + y1A12 + A13 - x2(x1A31 + y1A32 + A33) = 0
    # wy2 = y2(x1A31 + y1A32 + A33) = x1A21 + y1A22 + A23
        # >> x1A21 + y1A22 + A23 - y2(x1A31 + y1A32 + A33) = 0
"""


def homography(samples):
    a_matrix = []
    for sample in samples:
        p1 = np.append(sample[:2], 1)  # x1, y1, 1
        p2 = np.append(sample[2:], 1)  # x2, y2, 1
        a1 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]]
        a2 = [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]]
        a_matrix.append(a1)
        a_matrix.append(a2)
    a_matrix = np.array(a_matrix)
    # corresponding smallest eigenvalue of eigenvector
    _, _, vh = np.linalg.svd(a_matrix)
    h = vh[-1].reshape(3, 3)
    h = h / h[2][2]
    return h


def Ransac(matches, threshold, iterations):
    best_h_matrix = []
    best_inliers = []
    for iter in range(iterations):
        samples_index = random.sample(range(len(matches)), 4)
        samples = [matches[index] for index in samples_index]
        h_matrix = homography(samples)
        all_p1 = np.concatenate(
            (matches[:, :2], np.ones((len(matches), 1))), axis=1)
        all_p2 = matches[:, 2:]
        estimate_p2 = np.matmul(h_matrix, all_p1.T).T
        # print(estimate_p2.shape)
        for i in range(len(matches)):
            estimate_p2[i] = estimate_p2[i] / estimate_p2[i][2]
        estimate_p2 = estimate_p2[:, 0:2]
        errors = np.linalg.norm(all_p2 - estimate_p2, axis=1)**2
        inliers_index = np.where(errors < threshold)[0]
        inliers = matches[inliers_index]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_h_matrix = h_matrix
    return best_inliers, best_h_matrix


def translate(image1, image2, h_matrix):
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    # 以image2中的四個corner的座標當作基準去計算
    corners = np.array(
        [[0, 0, 1], [w2 - 1, 0, 1], [w2 - 1, h2 - 1, 1], [0, h2 - 1, 1]])
    # 把image2四個corner的座標經過homography matrix轉換後
    # 得到在image1相對應的座標
    transform_corners = np.matmul(h_matrix, corners.T).T
    for i in range(len(corners)):
        transform_corners[i] = transform_corners[i] / transform_corners[i][2]

    transform_corners = transform_corners[:, :2]
    x_min = min(min(transform_corners[:, 0]), 0)
    y_min = min(min(transform_corners[:, 1]), 0)
    """
    print(transform_corners) 
                            [[-170.06491952  -49.23902745] image2中(0, 0)對應到image1的座標
                            [ 836.57361736   38.94680412]  image2中(w2 - 1, 0)對應到image1的座標
                            [ 820.07621228  745.49125985]    ''    (w2 - 1, h2 - 1)    ''
                            [-205.34991699  763.95041104]]   ''    (0, h2 - 1)    ''
    """
    size = (int(round(w2 + abs(x_min))), int(round(h2 + abs(y_min))))

    # h_matrix是使用image2轉換成image1
    # 所以A是把image1轉換到與image2的translation_matrix
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    A = np.matmul(translation_matrix, h_matrix)
    """
        print(size) >> (1213, 805)
    """
    # 把image1用合併後的size來表示的話，
    # 是對image1和translation_matrix A 做運算
    warped_1 = cv2.warpPerspective(src=image1, M=A, dsize=size)

    # 因為我們是以image2為基準，
    # 所以要把image2以合併後的size表示的話
    # 只需要做affine translation就好

    print(h_matrix)
    print(translation_matrix)
    print(A)

    warped_2 = cv2.warpPerspective(
        src=image2, M=translation_matrix, dsize=size)
    return warped_1, warped_2


def isblack(pixel):
    black_pixel = np.array([0, 0, 0])
    return np.array_equal(pixel, black_pixel)


def stitch_image(warped_1, warped_2):
    rows, cols, channels = warped_1.shape
    output_image = np.zeros((rows, cols, channels), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (not isblack(warped_1[i][j])) and isblack(warped_2[i][j]):
                output_image[i][j][:] = warped_1[i][j][:]
            elif isblack(warped_1[i][j]) and (not isblack(warped_2[i][j])):
                output_image[i][j][:] = warped_2[i][j][:]
            elif (not isblack(warped_1[i][j])) and (not isblack(warped_2[i][j])):
                output_image[i][j][:] = (
                    warped_1[i][j][:] / 2) + (warped_2[i][j][:] / 2)
            else:
                pass
    return output_image
# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main


def create_im_window(window_name, img):
    cv2.imshow(window_name, img)

# show the all window you call before im_show()
# and press any key to close all windows


def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image1_gray, image1_rgb = read_img("test/m3.jpg")
    image2_gray, image2_rgb = read_img("test/m4.jpg")
    """
        # concat_rgb = np.concatenate([image1_rgb, image2_rgb], axis=1)
        # concat_gray = np.concatenate([image1_gray, image2_gray], axis=1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(concat_rgb)
        # plt.subplot(1, 2, 2)
        # plt.imshow(concat_gray, cmap="gray")
        # plt.show()
    """
    kp1, des1 = SIFT(image1_gray)
    kp2, des2 = SIFT(image2_gray)

    """ Plot the keypoints in concatenate images
        # image1_sift = plot_sift(image1_gray, image1_rgb, kp1)
        # image2_sift = plot_sift(image2_gray, image2_rgb, kp2)
        # concat_SIFT = np.concatenate([image1_sift, image2_sift], axis=1)
        # plt.imshow(concat_SIFT)
    """
    # Show the coordinate of keypoint
    # keypoints are in the form of x-y coordinate, not the common seen row-column form.
    # coordinate = cv2.KeyPoint_convert(kp1)

    two_nn = K_NN(kp1, des1, kp2, des2, 2)
    matches = Ratio_Test(kp1, des1, kp2, des2, two_nn, 0.95)
    # plot_matches(matches, concat_gray)

    best_inliers, best_h_matrix = Ransac(matches, 5, 2000)
    # plot_matches(best_inliers, concat_gray)

    warped_1, warped_2 = translate(image1_rgb, image2_rgb, best_h_matrix)
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 1, 1)
    plt.imshow(warped_1)
    plt.subplot(3, 1, 2)
    plt.imshow(warped_2)
    output_image = stitch_image(warped_1, warped_2)
    plt.subplot(3, 1, 3)
    plt.imshow(output_image)
    plt.show()

    # the example of image window
    # create_im_window("Result", test)
    # im_show()

    # you can use this function to store the result
    cv2.imwrite("result.jpg", output_image)

# %%
