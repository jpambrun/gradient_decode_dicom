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
      im_np = decode_dicom_image(dcm_data, color_dim=False,
        on_error='strict',
        scale='preserve',
        dtype=tf.float32,
        name=None).numpy()
      print(filename, im_np.shape, im_np)

print('TEST: TF version=' + tf.__version__)

if tf.__version__[0] == "1":
    test_tf1()
elif tf.__version__[0] == "2":
    test_tf2()
