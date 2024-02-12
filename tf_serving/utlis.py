import cv2
import numpy as np
import grpc
from protos.tensorflow_serving.apis import predict_pb2
from protos.tensorflow_serving.apis import prediction_service_pb2_grpc 
from protos.tensorflow.core.framework import tensor_pb2  
from protos.tensorflow.core.framework import tensor_shape_pb2  
from protos.tensorflow.core.framework import types_pb2  

def convert_image(encoded_img):

    if isinstance(encoded_img, str):
        b64_decoded_image = base64.b64decode(encoded_img)
    else:
        b64_decoded_image = encoded_img

    img_arr = np.fromstring(b64_decoded_image, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    img = np.expand_dims(img,axis=0).astype(np.float32)
    mean = [103.939, 116.779, 123.68]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    return img

def grpc_infer(img):
    """serving with gRPC
    """
    channel = grpc.insecure_channel("10.5.0.5:8500") # localhost:8500 for non-docker
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    # model_name
    request.model_spec.name = "makeup-serving"
    # signature name, default is `serving_default`
    request.model_spec.signature_name = "reid-predict"

    imgs = convert_image(img)
    
    # protos
    tensor_shape = imgs.shape
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape] 
    tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)  
    tensor = tensor_pb2.TensorProto(  
                  dtype=types_pb2.DT_FLOAT,
                  tensor_shape=tensor_shape,
                  float_val=imgs.reshape(-1))
    request.inputs['input_image'].CopyFrom(tensor)
    
    try:
        result = stub.Predict(request, 10.0)
        result = np.clip(result.outputs["output_image"].float_val, 0.0, 1.0)
        result = result.reshape((224,224, 3))
        return result
    except Exception as e:
        print(e)
        return None

