import tensorflow as tf
import glob

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

class Bboxes_tfrecord:
    def write(self, tf_fname, samples):
        with tf.python_io.TFRecordWriter(tf_fname) as tf_writer:
            tf.logging.info('BboxesClsLoc num_samples:%d' % len(samples))
            for sample in samples:
                bboxes = sample['bboxes'].flatten().tolist()
                features = tf.train.Features(feature={'bboxes': float_feature(bboxes)})
                example = tf.train.Example(features=features)
                tf_writer.write(example.SerializeToString())

    def read(self, tf_fname):
        fnames = glob.glob(tf_fname)
        if not fnames:
            raise Exception('%s do not exist' % tf_fname)
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=4)
        dataset = dataset.map(self.parser, num_parallel_calls=4)
        return dataset

    def parser(self, record):
        keys_to_features = {'bboxes': tf.VarLenFeature(dtype=tf.float32)}
        features = tf.parse_single_example(record, features=keys_to_features)
        bboxes = tf.sparse_tensor_to_dense(features['bboxes'])
        bboxes = tf.reshape(bboxes, [-1, 4])
        return {'bboxes': bboxes}
