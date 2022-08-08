import cv2
import numpy as np
import core.utils as utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
# from ImageEnhance import retinex
import json

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path      = "D:/ObjectDetection/Yolov3_using_Tensorflow/VOC2012/Images/342.jpg"
num_classes     = 1
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with open('D:/ObjectDetection/Yolov3_using_Tensorflow/ImageEnhance/config.json', 'r') as f:
    config = json.load(f)

# original_image = cv2.imread(image_path)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# original_image_size = original_image.shape[:2]
# image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
# image_data = image_data[np.newaxis, ...]
# return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

def ocrDetection():
    print("OCR start...")

with tf.Session(graph=graph) as sess:
    original_image = cv2.imread(image_path)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    
    # img_msrcr = retinex.MSRCR(
    #     original_image,
    #     config['sigma_list'],
    #     config['G'],
    #     config['b'],
    #     config['alpha'],
    #     config['beta'],
    #     config['low_clip'],
    #     config['high_clip']
    # )
   
    # img_amsrcr = retinex.automatedMSRCR(
    #     original_image,
    #     config['sigma_list']
    # )

    # img_msrcp = retinex.MSRCP(
    #     original_image,
    #     config['sigma_list'],
    #     config['low_clip'],
    #     config['high_clip']        
    # )
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.2, method='nms')
    image1 = utils.draw_bbox(original_image, bboxes)
    # image2 = utils.draw_bbox(img_msrcr, bboxes)
    # image3 = utils.draw_bbox(img_amsrcr, bboxes)
    # image4 = utils.draw_bbox(img_msrcp, bboxes)
    # image = Image.fromarray(image)
    result1 = np.asarray(image1)
    # result1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

    # result2 = np.asarray(image2)
    # result2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # result3 = np.asarray(image3)
    # result3 = cv2.cvtColor(image3, cv2.COLOR_RGB2BGR)

    # result4 = np.asarray(image4)
    # result4 = cv2.cvtColor(image4, cv2.COLOR_RGB2BGR)

    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('Image', result1)
    print(result1)
    # cv2.imshow('MSRCR', result2)
    # cv2.imshow('AutoMSRCR', result3)
    # cv2.imshow('MSRCP', result4)
    # cv2.imshow('Retinex', img_msrcr)
    # cv2.imshow('Automated Retinex', img_amsrcr)
    # cv2.imshow('MSRCP', img_msrcp)
    cv2.waitKey(0)

cv2.destroyAllWindows()



