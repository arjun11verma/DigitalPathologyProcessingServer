# Image Stitching Server

The Image Stitching Server accepts slide images sent from the app over HTTP and combines them to form a fully stitched whole slide image.

## Installation/Usage

Pip install requirements.txt. Python3 and Pip3 required. Running this server will create a ngrok tunnel that can be accessed from outside of the local host - use the generated ngrok url in the app so the app knows where to send the images. 