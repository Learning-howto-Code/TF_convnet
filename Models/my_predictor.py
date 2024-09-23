import tensorflow as tf
import numpy as np


def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dytype=tf.float32)
    image = tf.image.resize(image, [60,60]) #(60,60, 3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    predictions = model.predict(image) # [0.005, 0.0003, 0.99, 0.00 ....]
    predictions = np.argmax(predictions) # 2

    return predictions

if __name__=="__main__":

    img_path = "/Users/jakehopkins/Downloads/German Traffic Signs Dataset/Test/2/00124.png"
    model = tf.keras.layers.TFSMLayer('./Models', call_endpoint='serving_default')


    prediction = predict_with_model(model, img_path)

    print(f"predictio = {prediction}")