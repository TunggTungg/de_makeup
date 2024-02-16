import cv2
import numpy as np
import grpc
from ovmsclient import make_grpc_client
from PIL import Image
import base64


def convert_image(encoded_img):

    if isinstance(encoded_img, str):
        b64_decoded_image = base64.b64decode(encoded_img)
    else:
        b64_decoded_image = encoded_img

    img_arr = np.fromstring(b64_decoded_image, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img,axis=0).astype(np.float32)
    return img

def grpc_infer(img):
    """serving with gRPC
    """
    client = make_grpc_client("10.5.0.5:9000")
    img = convert_image(img)
    inputs = {"input_1": img}
    results = client.predict(inputs=inputs, model_name="demakeup")
    return results

if __name__ == '__main__':
    client = make_grpc_client("localhost:9000")
    array_zeros = np.zeros((224, 244, 3), dtype =np.float32)
    array_zeros = np.expand_dims(array_zeros,0)
    inputs = {"input_1": array_zeros}
    results = client.predict(inputs=inputs, model_name="demakeup")
    print(results.shape)
    img = Image.fromarray((results[0] * 255).astype(np.uint8))
    img = img.save("geeks.jpg")
