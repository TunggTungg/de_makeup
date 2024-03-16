from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import keras

class de_Makeup_Model:
    def __init__(self, transfer_learning=False):
        self.transfer_learning = transfer_learning

    def ResNet50(self, input) -> Model:
        encoder_blocks_name = ["conv1_relu", "conv2_block3_out", "conv3_block4_out",
                                    "conv4_block6_out"]
        model = tf.keras.applications.ResNet50(include_top=False, input_tensor=input,
                                        weights='imagenet')
        model.trainble = False
        return encoder_blocks_name, Model(inputs=model.input, outputs=model.output, name='encoder')

    def conv_block(self, inputs, num_filters):
        x = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=num_filters, kernel_size=(3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x

    def upsample_concate_block(self, inputs, skip_connection, num_filters, strides=2, kernel_size=2):
        x = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, output_padding = 0)(inputs)
        x = layers.Concatenate()([skip_connection, x])
        x = self.conv_block(x, num_filters)
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self, pre_trained=None):

        inputs = keras.Input(shape=(224, 224) + (3,))
        ### [First half of the network: downsampling inputs] ###
        encoder_blocks_name, backbone  = self.ResNet50(inputs)
        ### [First half of the network: downsampling inputs] ###
        encoder_blocks = []
        for i in range(len(encoder_blocks_name)):
          encoder_blocks.append(backbone.get_layer(name=encoder_blocks_name[i]).output)
          # print(encoder_blocks[i])

        # bridge
        br = backbone.output

        # decoder
        db5 = self.upsample_concate_block(inputs=br, skip_connection=encoder_blocks[-1], num_filters=256)
        db4 = self.upsample_concate_block(inputs=db5, skip_connection=encoder_blocks[-2], num_filters=256)
        db3 = self.upsample_concate_block(inputs=db4, skip_connection=encoder_blocks[-3], num_filters=128)
        db2 = self.upsample_concate_block(inputs=db3, skip_connection=encoder_blocks[-4], num_filters=64)

        # final output
        first_feature = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
        final_feature = self.upsample_concate_block(inputs=db2, skip_connection=first_feature, num_filters=64)
        outputs = layers.Conv2D(filters=3, kernel_size=(1,1), activation='tanh')(final_feature)

        # Define the model
        model = keras.Model(backbone.input, outputs, name='de-makeup')

        if pre_trained != None:
          model.load_weights(pre_trained)
          print(f"Load {pre_trained} successfully")
        return model

if __name__ == '__main__':
    model = de_Makeup_Model().build_model('model/model.h5')
