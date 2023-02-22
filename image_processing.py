import tensorflow as tf

IMG_HEGIHT  = 224
IMG_WIDHT   = 224
IMG_CHANNEL = 3

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=IMG_CHANNEL)
    
    input_image  = tf.image.resize(image, (IMG_HEGIHT, IMG_WIDHT))
    
    # Convert both images to float32 tensors
    input_image  = tf.cast(input_image, tf.float32)/255.0
    
    return input_image

def load_image_val(image_file):
    input_image = load(image_file)
    return input_image