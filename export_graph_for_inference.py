import tensorflow as tf

from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('version', 1,
                        'model version number, default: 1')


def export_graph():
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size=FLAGS.image_size)

        # [N, N, 3] 크기의 float32 행렬을 받는다.
        # input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3], name='input_image')

        # 이미지를 base64 인코딩한 scalar 값을 받는다.
        input_bytes = tf.placeholder(tf.string, shape=[])
        # base64 이미지를 int Tensor 타입으로 변형한다.
        input_image = tf.image.decode_jpeg(input_bytes, channels=3)
        # 이미지를 리사이징한다.
        input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
        # 텐서를 float 타입으로 바꾼다.
        input_image = utils.convert2float(input_image)
        # 이미지 shape 변환
        input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

        cycle_gan.model()
        output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))

        output_bytes = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)

        # SavedModel을 저장할 경로
        export_dir = "savedmodel/{}".format(FLAGS.version)
        fitting_signatures = {
            # 이 시그니처 이름은 gRPC 또는 REST 요청시 적어주어야 합니다.
            "fitting": tf.saved_model.signature_def_utils.predict_signature_def(
                # _bytes 접미어를 가지고 있는 텐서는 바이너리 값을 가지고 있는 것으로 간주합니다.
                inputs={"input_bytes": input_bytes},
                outputs={"output_bytes": output_bytes})
        }

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        # SavedModel에 그래프 추가
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=fitting_signatures)
        builder.save()


def main(unused_argv):
    print('Export model...')
    export_graph()


if __name__ == '__main__':
    # flags 파싱 및 main 함수 호출
    tf.app.run()
