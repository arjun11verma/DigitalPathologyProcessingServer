from Databases import S3Database
import json

s3_instance = S3Database()

s3_instance.s3_upload('/test/data', json.dumps({'test': "test"}))