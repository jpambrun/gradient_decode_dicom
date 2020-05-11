import tensorflow as tf
from gradient_decode_dicom import decode_dicom_image, decode_dicom_data, tags

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
    @tf.function
    def preprocess(dcm_content):
        dcm_tags = decode_dicom_data(
            dcm_content, tags=[tags.ImageOrientationPatient, tags.PixelSpacing]
        )
        iop = tf.strings.to_number(tf.strings.split([dcm_tags[0]], sep="\\")[0])
        image = decode_dicom_image(dcm_content, dtype=tf.float32)
        image.set_shape([1, None, None, 1])
        if iop[0] < 0:
            image = tf.image.flip_up_down(image)
        if iop[4] < 0:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_pad(image, 224, 224, antialias=True)
        i_min = tf.reduce_min(image)
        image_masked = tf.boolean_mask(image, image > i_min + 2)
        i_mean = tf.math.reduce_mean(image_masked)
        i_std = tf.math.reduce_std(image_masked)
        image = tf.clip_by_value(image, i_mean - 4 * i_std, i_mean + 4 * i_std)
        image = (image - i_mean) / (2 * i_std)
        image = tf.image.grayscale_to_rgb(image)
        image = tf.reshape(image, [224, 224, 3])
        return image

    dataset = (
        tf.data.Dataset.list_files("*.dcm")
        .map(tf.io.read_file)
        .repeat()
        .map(preprocess, num_parallel_calls=4)
        .take(3000)
    )

    for i, item in enumerate(dataset):
        if i % 100 == 0:
            print(tf.shape(item), i)

print('TEST: TF version=' + tf.__version__)

if tf.__version__[0] == "1":
    test_tf1()
elif tf.__version__[0] == "2":
    test_tf2()
