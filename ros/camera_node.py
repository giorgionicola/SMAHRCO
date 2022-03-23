import rospy
import pyk4a
from pyk4a import PyK4A, Config
import numpy as np
from skimage.transform import resize_local_mean
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import apriltag
import cv2
import matplotlib.pyplot as plt
import time

depth_threshold = 1000.0
crop_region = ((0, 1080), (210, 1750))
final_shape = (128, 128)

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)


def grayConversion(image):
    grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


def preprocess_hsv(depth: np.array, color: np.array):
    new_depth = depth[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
    depth_mask = new_depth < depth_threshold

    color = color[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1],:]

    hsv_img = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    hue_img = hsv_img[:, :, 0] / 255
    # saturation_img = hsv_img[:, :, 1] / 255
    # value_img = hsv_img[:, :, 2] / 255

    # hue_img = hue_img[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
    # saturation_img = saturation_img[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
    # value_img = value_img[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]

    hue_threshold_low = 0.43
    hue_threshold_high = 0.55
    hue_mask = hue_threshold_low < hue_img
    hue_mask *= hue_img < hue_threshold_high

    # value_threshold_high = 1
    # value_mask = value_img < value_threshold_high

    depth_mask = depth_mask * hue_mask #* value_mask

    new_depth = new_depth * depth_mask
    new_depth = cv2.resize(new_depth, final_shape)

    depth_rviz = new_depth * (2 ** 16 - 1)
    new_depth = new_depth / depth_threshold

    return new_depth, depth_rviz.astype(np.uint16)


def preprocess_apriltag(depth: np.array, color: np.array):
    new_depth = depth[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
    depth_mask = (new_depth > 0) * (new_depth < depth_threshold)

    gray_image = grayConversion(color)

    tags_detected = detector.detect(gray_image)

    if len(tags_detected) == 2:
        tag1_corners = tags_detected[0].corners
        tag2_corners = tags_detected[1].corners

        bottom_point1 = tag1_corners[tag1_corners[:, 1].argmax()]
        bottom_point2 = tag2_corners[tag2_corners[:, 1].argmax()]

        a = (bottom_point1[1] - bottom_point2[1]) / (bottom_point1[0] - bottom_point2[0])
        b = bottom_point1[1] - a * bottom_point1[0]
    elif len(tags_detected) == 1:
        tag1_corners = tags_detected[0].corners
        a = 0
        b = tag1_corners[tag1_corners[:, 1].argmax(), 1]
    elif len(tags_detected) > 2:
        print(
            f'WTF!!! More than 2 tags!!!! tags detected: {len(tags_detected)}, current method not likely to work with more than 3 tags')
        while len(tags_detected) > 2:
            avg_y_center = np.mean([tags_detected[i].center[1] for i in range(len(tags_detected))])
            delta_y = [tags_detected[i].center[1] - avg_y_center for i in range(len(tags_detected))]
            tags_detected.pop(np.argmax([abs(d) for d in delta_y]))

        tag1_corners = tags_detected[0].corners
        tag2_corners = tags_detected[1].corners

        bottom_point1 = tag1_corners[tag1_corners[:, 1].argmax()]
        bottom_point2 = tag2_corners[tag2_corners[:, 1].argmax()]

        a = (bottom_point1[1] - bottom_point2[1]) / (bottom_point1[0] - bottom_point2[0])
        b = bottom_point1[1] - a * bottom_point1[0]

    if len(tags_detected) > 0:
        bottom_line = lambda x: a * x + b
        for i in range(depth_mask.shape[1]):
            depth_mask[:int(bottom_line(i) + 100), i] = False
    else:
        rospy.logwarn('No tag visible!!!!')

    new_depth = new_depth * depth_mask
    new_depth = cv2.resize(new_depth, final_shape)

    depth_rviz = new_depth * (2 ** 16 - 1)
    new_depth = new_depth / depth_threshold

    return new_depth, depth_rviz.astype(np.uint16)


def main():
    azure = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1080P,
                         color_format=pyk4a.ImageFormat.COLOR_BGRA32,
                         depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                         camera_fps=pyk4a.FPS.FPS_30,
                         synchronized_images_only=True,
                         depth_delay_off_color_usec=0,
                         wired_sync_mode=pyk4a.WiredSyncMode.STANDALONE,
                         subordinate_delay_off_master_usec=0,
                         disable_streaming_indicator=False))
    azure.start()

    rospy.init_node(name='Azure_kinect')
    rate = rospy.Rate(30)

    n_images = 3

    img_pub = rospy.Publisher('/preprocessed_image', Image, queue_size=0)
    img_pub_rviz = rospy.Publisher('/preprocessed_image_rviz', Image, queue_size=0)

    br = CvBridge()
    i = 1
    filtered_depth = np.zeros(final_shape)
    filtered_depth_rviz = np.zeros(final_shape, dtype=np.uint16)
    while not rospy.is_shutdown():

        capture = azure.get_capture()
        if np.any(capture.depth) and np.any(capture.color):
            t0 = time.time()
            depth = pyk4a.depth_image_to_color_camera(capture.depth, capture._calibration, capture.thread_safe)
            new_depth, depth_rviz = preprocess_hsv(depth, capture.color)
            # new_depth, depth_rviz = preprocess_apriltag(depth, capture.color)


            filtered_depth += new_depth
            filtered_depth_rviz += depth_rviz

            if i == n_images:
                filtered_depth /= n_images
                msg: Image = br.cv2_to_imgmsg(filtered_depth)
                msg.header.stamp = rospy.Time.now()
                img_pub.publish(msg)

                filtered_depth_rviz = filtered_depth_rviz // n_images
                msg_2: Image = br.cv2_to_imgmsg(filtered_depth_rviz, )
                msg_2.header.stamp = rospy.Time.now()
                img_pub_rviz.publish(msg_2)

                filtered_depth = np.zeros(final_shape)
                filtered_depth_rviz = np.zeros(final_shape, dtype=np.uint16)
                i = 1
            else:
                i += 1

        rate.sleep()


if __name__ == '__main__':
    main()
