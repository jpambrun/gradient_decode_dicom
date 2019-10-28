import tensorflow as tf
from gradient_decode_dicom import decode_dicom_image

def test_tf1():
    g = tf.Graph()
    with g.as_default():
        filename_t = tf.placeholder(tf.string)
        dcm_data = tf.io.read_file(filename_t)
        op = decode_dicom_image(dcm_data)

    with tf.Session(graph=g) as sess:
        for filename in tf.io.gfile.glob('*.dcm'):
            im_np = sess.run(op, feed_dict={filename_t: filename})
            print(im_np.shape)

def test_tf2():
    for filename in tf.io.gfile.glob('*.dcm'):
      dcm_data = tf.io.read_file(filename)
      im_float = decode_dicom_image(dcm_data, on_error='strict', dtype=tf.float32, color_dim=False).numpy()
      im_int16 = decode_dicom_image(dcm_data, on_error='strict', dtype=tf.int16, color_dim=False).numpy()
      im_int32 = decode_dicom_image(dcm_data, on_error='strict', dtype=tf.int32, color_dim=False).numpy()
      im_int64 = decode_dicom_image(dcm_data, on_error='strict', dtype=tf.int64, color_dim=False).numpy()
      print(filename, im_float.shape, im_float, im_float[0,225:227,220:230])
      print(filename, im_int16.shape, im_int16, im_int16[0,225:227,220:230])
      print(filename, im_int32.shape, im_int32, im_int32[0,225:227,220:230])
      print(filename, im_int64.shape, im_int64, im_int64[0,225:227,220:230])

print('TEST: TF version=' + tf.__version__)

if tf.__version__[0] == "1":
    test_tf1()
elif tf.__version__[0] == "2":
    test_tf2()
