from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from model import de_Makeup_Model
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
if tf.executing_eagerly():
   tf.compat.v1.disable_eager_execution()

model = de_Makeup_Model().build_model('model.h5')

builder = saved_model_builder.SavedModelBuilder('models/serving/1')

signature = predict_signature_def(
    inputs={
        'input_image': model.inputs[0],
    },
    outputs={
        'output_image': model.outputs[0]
    }
)

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'reid-predict': signature},
        # or
        # signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
    )
    builder.save()
