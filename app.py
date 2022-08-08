from flask import Flask, request, Response, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import core.utils as utils
from core.config import cfg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import json
import time
import pytesseract

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
num_classes     = 1
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)
output_path = './detections/'

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

@app.route('/')
def home():
    return "License Plate Recognition"


def CharacterNumberDetect(img):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    gray = cv2.imread(img, 0)
    gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.medianBlur(gray, 3)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # cv2.imshow("Otsu", thresh)
    # cv2.waitKey(0)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # create copy of image
    im2 = gray.copy()

    plate_num = ""
    # loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        
        # if height of box is not a quarter of total height then skip
        if height / float(h) > 6: continue
        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue
        area = h * w
        # if width is not more than 25 pixels skip
        if width / float(w) > 15: continue
        # if area is less than 100 pixels skip
        if area < 100: continue
        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        #cv2.imshow("ROI", roi)
        #cv2.waitKey(0)
        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        print("text " + text)
        plate_num += text
    print(plate_num)
    return plate_num


def image_detection(img):
    response = []
    print(img)
     # start detection
    with tf.Session(graph=graph) as sess:
        original_image = cv2.imread(img)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.2, method='nms')
        imageResponse = utils.draw_bbox(original_image, bboxes)

        return imageResponse


@app.route('/LicensePlateDetection', methods=['POST'])
def get_License_Plate_Detections():
    raw_images = []
    images = request.files['images']
    filename = secure_filename(images.filename) # save file 
    filepath = os.path.join('./PlateImage', filename)
    images.save(filepath)
    print(filepath)
    cv2.imread(filepath)
    print (images)
    image_names = []

    # for image in images:
    image_name = images.filename
    image_names.append(image_name)
    images.save(os.path.join(os.getcwd(), image_name))
    img_raw = tf.image.decode_image(
        open(image_name, 'rb').read(), channels=3)

    print("raw")
    raw_images.append(img_raw)
    print(raw_images)
        
    num = 0
    
    # create list for final response
    response = []

    for j in range(len(raw_images)):
        responses = []
        raw_img = raw_images[j]
        print(raw_images[j])
        num+=1
        img = tf.expand_dims(raw_img, 0)
        print("here")
        print(img)

        t1 = time.time()
        response = image_detection(filepath)
        print(response)
        str(response)
        if response is not None: 
            print("License Plate Start Recognize...")
            carPlateNumber = CharacterNumberDetect(filepath)
            response.append({"PlateNumber": carPlateNumber.replace('\n','')})
            print(response)

        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        #remove temporary images
        for name in image_names:
            os.remove(name)
        try:
            return jsonify({"response":response}), 200
        except FileNotFoundError:
            abort(404)


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)