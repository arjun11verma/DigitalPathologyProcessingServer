from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask_ngrok import run_with_ngrok

from ast import literal_eval
from datetime import datetime

from ImageProcessor import converter, stitcher
from Databases import MongoDatabase, S3Database

app = Flask(__name__)

@app.route('/upload_image', methods = ['POST'])
def uploadImage():
   slide_image_data = (literal_eval(request.data.decode('utf8')))
   s3_path = f'{S3Database.generate_slide_image_path(slide_image_data)}/{slide_image_data["timestamp"]}'
   success = None 

   if (slide_image_data['status'] == "Y"):
      mongo_instance = MongoDatabase(app)
      success = mongo_instance.upload_slide_image(slide_image_data)
   else:
      s3_instance = S3Database()
      s3_instance.s3_delete(s3_path)
   
   return {'response': "Y"} if success != None else {'response': "N"}

@app.route('/stitch_images', methods = ['POST'])
def acceptImages():
   image_stitcher = stitcher()
   s3_instance = S3Database()
   timestamp = (datetime.now()).strftime("%m:%d:%Y-%H:%M:%S")
   slide_image_data = (literal_eval(request.data.decode('utf8')))

   slide_images = [slide_image_data[str(i)] for i in range(len(slide_image_data) - 4)]
   slide_images = [converter.base64_to_numpy(slide_image) for slide_image in slide_images]
   stitched_image = image_stitcher.stitch_images(slide_images)
   stitched_image = converter.numpy_to_base64(stitched_image)

   success = s3_instance.upload_slide_image(stitched_image, slide_image_data, timestamp) if stitched_image != None else False
   return {'response': "Y", 'imageData': stitched_image} if success else {'response': "N", 'imageData': None}

def main():
   cors = CORS(app)
   run_with_ngrok(app)
   app.run()

if __name__ == "__main__":
   main()

   
   