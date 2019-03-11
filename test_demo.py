# import tensorflow as tf
# img = tf.constant(value=[[[[1], [2], [3], [4]],
#                           [[1], [2], [3], [4]],
#                           [[1], [2], [3], [4]],
#                           [[1], [2], [3], [4]]]], dtype=tf.float32)
# img = tf.concat(values=[img, img], axis=3)
#
# filter = tf.constant(value=1, shape=[3, 3, 2, 5], dtype=tf.float32)
# out_img1 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='SAME')
# out_img2 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='VALID')
# out_img3 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='SAME')
# reduce_min = tf.reduce_min(img, axis=-1)
# # error
# # out_img4 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='VALID')
# with tf.Session() as sess:
#     print('rate=1, SAME mode result:')
#     print(sess.run(out_img1))
#     print('rate=1, VALID mode result:')
#     print(sess.run(out_img2))
#     print('rate=2, SAME mode result:')
#     print(sess.run(out_img3))  # error #print 'rate=2, VALID mode result:' #print(sess.run(out_img4))
#     print("reduce_min")
#     print(sess.run(reduce_min))
import numpy as np
# arr =   np.array([[[[1], [2], [3]],
#                       [[1], [2], [3]],
#                       [[1], [2], [3]],
#                       [[1], [2], [3]]]])
# sum = np.sum(arr, axis=1)
# print(sum)

# x = np.arange(0, 80, 1, float)
# y = x[:, np.newaxis]
# print(x.shape)
# print(y.shape)

print(chr(ord("a") ^ 0))

import cv2
import matplotlib.pyplot as plt
# img = cv2.imread("./im0001.jpg")
# img = cv2.circle(img, (20, 20), 2, (255,0,0))
# img = cv2.resize(img, (12, 512))
# print(img.shape)
# img = cv2.imread("./test.jpg")
# img = cv2.circle(img, (393, 160), 2, (255,0,0))
# plt.figure("Image") # 图像窗口名称
# plt.imshow(img)
# plt.show()
# img_h, img_w = img.shape[0], img.shape[1]
# cv2.imshow("before", img)
# cv2.waitKey(0)
# img_flip = cv2.flip(img, 1)
# cv2.imshow("flip", img_flip)
# cv2.waitKey(0)
# matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), 20, 1)
# img_rorate = cv2.warpAffine(img, matrix, (img_w, img_h), flags=cv2.INTER_CUBIC)
# new_point = np.matmul((20, 20, 1), matrix.T)
# print(new_point)
# img = cv2.circle(img_rorate, tuple(new_point), 2, (255,0,0))
# cv2.resize(interpolation=cv2.INTER_)
# # print(img_rorate)
# cv2.imshow("rorate2", img)
# cv2.waitKey(0)

# import tensorflow as tf
# def rorate(img):
#     print(img)
#     img_h, img_w = img.shape[0], img.shape[1]
# #     print(img_h)
#     matrix = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), 20, 1)
# #     print(matrix)
#     img_rorate = cv2.warpAffine(img, matrix, (img_w, img_h), flags=cv2.INTER_CUBIC)
#
# #     print("rorate:",img_rorate)
#     return img_rorate
# # # print(img)
# # cvt_img = tf.convert_to_tensor(img,dtype=tf.float32)
# # print(type(cvt_img))
# # img = np.array(img, dtype=np.float32)
# img_input = tf.placeholder(tf.uint8,[None, 160, 70, 3],name = 'array1')
# #
# # print(img)
# resize = tf.image.resize_bilinear(img_input,(160, 160))
# print_img = tf.Print(resize, [resize], message='Debug:')
# tf_rorate = tf.py_func(rorate, [img_input], tf.uint8)
# #
# with tf.Session() as sess:
#     # img = sess.run(cvt_img,feed_dict={img:img})
#     # print(type(img))
#     img = np.expand_dims(img, axis=0)
#     print(img.shape)
#     sess.run(tf.global_variables_initializer())
#     # img_res = sess.run(tf_rorate, feed_dict={img_input:img})
#     sess.run(print_img, feed_dict={img_input:img})
# cv2.imshow("rorate2", img_res)
# cv2.waitKey(0)

def np_draw_labelmap(pt, heatmap_sigma, heatmap_size, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    if pt[0] < 1 or pt[1] < 1:
        return (img, 0)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * heatmap_sigma), int(pt[1] - 3 * heatmap_sigma)]
    br = [int(pt[0] + 3 * heatmap_sigma + 1), int(pt[1] + 3 * heatmap_sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return (img, 0)

    # Generate gaussian
    size = 6 * heatmap_sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))
    elif type == 'Cauchy':
        g = heatmap_sigma / (((x - x0) ** 2 + (y - y0) ** 2 + heatmap_sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return (img, 1)
path = "/media/roots/data/keypoints_detection/Datasets/test/Images/blouse/0026a3bf368ceac5ea8106f207a9129a.jpg"
# res = np_draw_labelmap([25, 25], 6, 128)
heatmap_size = 128
# img = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
# res = cv2.GaussianBlur(img, (5, 5), 5, (25, 25))
# cv2.imshow('Guassian', res)
# cv2.waitKey(0)
img = cv2.imread(path,1)
cv2.imshow('img', img)
cv2.waitKey(0)