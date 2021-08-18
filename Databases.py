from flask_pymongo import PyMongo
from bson import ObjectId
import json 
import boto3
from APIKEYS import MONGODB_KEY, S3_ACCESS_KEY, S3_SECRET_KEY

class MongoDatabase:
    def __init__(self, app):
        self.mongo = PyMongo(app, uri = MONGODB_KEY)
        self.images = self.mongo.db.ImageSet
    
    def mongo_upload(self, data):
        return self.images.insert_one(data)
    
    def upload_slide_image(self, slide_image_data):
        username = slide_image_data['username']
        slide_id = slide_image_data['slide_id']
        slide_type = slide_image_data['slide']
        cancer_type = slide_image_data['cancer']
        timestamp = slide_image_data['timestamp']
        upload_document = {'username': username, 'slide_id': slide_id, 'slide': slide_type, 'cancer': cancer_type, 'timestamp': timestamp, 'diagnosis': "N"}
        return self.mongo_upload(upload_document)

# https://aws.amazon.com/premiumsupport/knowledge-center/decrypt-kms-encrypted-objects-s3/
# https://boto3.amazonaws.com/v1/documentation/api/latest/index.html look at credentials section 

class S3Database:
    def __init__(self):
        self.s3_resource = boto3.resource(
            's3',
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY
        )
        self.s3_bucket = self.s3_resource.Bucket("digitalpath")

    @staticmethod
    def generate_slide_image_path(slide_image_data):
        return f'/{slide_image_data["username"]}/{slide_image_data["slide_id"]}'
    
    def s3_upload(self, path, data):
        return self.s3_bucket.put_object(Key=path, Body=data)
    
    def s3_delete(self, path):
        return self.s3_bucket.delete_objects(Delete={
            'Objects': [
                {
                    'Key': path
                }
            ]
        })
    
    def upload_slide_image(self, slide_image, slide_image_data, timestamp):
        return self.s3_upload(f'{self.generate_slide_image_path(slide_image_data)}/{timestamp}', slide_image)
    
    def delete_slide_image(self, slide_image_data):
        return self.s3_delete(f'{self.generate_slide_image_path(slide_image_data)}/{slide_image_data["timestamp"]}')
