import cv2
import numpy as np
import base64

class converter:
    @staticmethod
    def numpy_to_base64(img_data):
        success, img_data = cv2.imencode(".jpg", img_data)
        byte_list = np.array([data[0] for data in img_data]).tobytes()
        byte_list = base64.b64encode(byte_list)
        return str(byte_list, 'utf-8')

    @staticmethod
    def base64_to_numpy(img_data):
        im_bytes = base64.b64decode(img_data)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        return img

class stitcher:
    def __init__(self):
        self.stitch_object = cv2.Stitcher.create() # just for now, once you get your stitcher working then replace this 

    def stitch_images(self, image_list):
        (status, result) = self.stitch_object.stitch(image_list)
        return result if status == cv2.STITCHER_OK else None