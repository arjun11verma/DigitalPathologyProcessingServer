import cv2
import numpy as np
import base64
import os

def crop(img_arr, top, left, bottom, right):
    return img_arr[top:bottom, left:right]

class ImageProcessor:
    """Image processing class for converting image formats, stitching images together, applying filters and removing black space from microscope images"""

    def __init__(self):
        self.stitcher = cv2.Stitcher.create()
        """Intializer for the class"""

    def combineImages(self, image_list):
        return np.concatenate(image_list, axis=0)

    def displayImage(self, img_url):
        """Displays an image from either a filepath or an Numpy array"""
        if(isinstance(img_url, str)):
            slide_image = cv2.imread(img_url)
        else:
            slide_image = img_url

        cv2.imshow("Slide Image", slide_image)
        cv2.waitKey(0)

    def recordImage(self, img_url, img_name, save_image):
        """Converts an image to Numpy and saves it if the option is selected."""
        if(isinstance(img_url, str)):
            slide_image = cv2.imread(img_url)
        else:
            slide_image = img_url

        self.num_images += 1
        img_name = img_name + str(self.num_images) + ".jpg"

        path = 'C:\VSCode Projects\DigitalPathology\ImageProcessingServer\RecordedImages'
        if(save_image): cv2.imwrite(os.path.join(path, img_name), slide_image)

        return slide_image
    
    def twoDimConvolution(self, slide_image, kernel):
        """Performs a 2D Convlution on the image"""
        return cv2.filter2D(slide_image, -1, kernel)
    
    def removeNoise(self, img_data):
        pass
    
    def sharpenImage(self, img_data, factor, increase):
        """Applies a typical medical image processing sharpening kernel to the image"""
        kernel_data = []
        factor = 1/factor

        dim = 3
        for i in range(dim*dim):
            kernel_data.append(-1/factor)

        kernel_data[int(dim*dim/2)] = (factor-1)/(factor) + increase

        sharpening_kernel = np.array(kernel_data).reshape((dim, dim))
    
        print(sharpening_kernel)
        
        return self.twoDimConvolution(img_data, sharpening_kernel)

    def stitchImages(self, slides):
        """Stitches together an array of images using the OpenCV Panorama stitcher"""
        (status, result) = self.stitcher.stitch(slides)

        if(status == cv2.STITCHER_OK):
            return result
        else:
            return np.array([])

    def base64ToArray(self, img_data):
        """Converts an image from Base64 to a Numpy array"""
        im_bytes = base64.b64decode(img_data)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        return img
    
    def arrayToJPEG(self, img_data):
        """Converts an image from a Numpy array to a JPEG buffer"""
        result, jpeg_data = cv2.imencode('.jpg', img_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jpeg_data if result else None
    
    def arrayToBase64(self, img_data):
        """Converts an image from a Numpy array to Base64"""
        success, img_data = cv2.imencode(".jpg", img_data)

        byte_list = []
        for i in range(len(img_data)):
            byte_list.append((img_data[i])[0])
        
        byte_list = np.array(byte_list)
        byte_list = byte_list.tobytes()
        byte_list = base64.b64encode(byte_list)

        return str(byte_list, 'utf-8')
