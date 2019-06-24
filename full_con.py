import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.app.flags.DEFINE_integer('train_or_test', 0, '0则为训练，非0则为测试')
FLAGS = tf.app.flags.FLAGS


def random_initial(shape):
    """
    :param shape: 输入形状
    :return: 返回标准差为1，均值为0的一组随机值 
    """
    return tf.random_normal(shape)


def full_con():
    # 准备数据
    mnist = input_data.read_data_sets("MNIST_data/mnist/", one_hot=True)
    with tf.variable_scope('placholder'):
        x = tf.placeholder(tf.float32, [None, 28*28], name='x_placeholder')
        y = tf.placeholder(tf.float32, [None, 10], name='y_placeholder')

    with tf.variable_scope('variable'):
        w = tf.Variable(random_initial([784, 10]), name='weight')
        bias = tf.Variable(random_initial([10]), name='bias')

    with tf.variable_scope('loss'):
        y_predict = tf.matmul(x, w) + bias
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.variable_scope('accuracy'):
        # 计算每步的识别准确率;tf.equal()返回equal_list,
        equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))
        # 必须转化为list of allowed values: float32, float64, int32, uint8, int16, int8,
        # complex64,int64, qint8, quint8, qint32, bfloat16, uint16,
        # complex128, float16, uint32, uint64中的一个才能输入reduce_mean()求平均值
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 存在变量需要初始化变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver(var_list=[w, bias], max_to_keep=2)
        if FLAGS.train_or_test == 0:
            for i in range(2000):
                train_x, train_y = mnist.train.next_batch(100)
                sess.run(train_op, feed_dict={x: train_x, y: train_y})
                acc = sess.run(accuracy, feed_dict={x: train_x, y: train_y})
                print('第{0}步，准确率为{1}'.format(i, acc))
                # 每训练1000次，保存模型
                if i % 1000 == 0 and i != 0:
                    saver.save(sess, 'model/full_con.ckpt')
        else:
            # 判断是否存在full_con.ckpt;文件名并非为这个，而是full_con.ckpt.data-00000-of-00001
            # if os.path.exists('./model/full_con.ckpt'):
            file_list = os.listdir('./model')
            for file in file_list:
                if file.startswith('full_con'):
                    saver.restore(sess, 'model/full_con.ckpt')
                    # 用训练的结果做预测
                    for i in range(20):
                        x_test, y_test = mnist.test.next_batch(1)
                        y_pre = sess.run(y_predict, feed_dict={x: x_test, y: y_test})
                        print('实际为{0}，预测为{1}'.format(tf.argmax(y_test, 1).eval(),
                                                     tf.argmax(y_pre, 1).eval()))
                    break
            else:
                print('please train first!')

if __name__ == '__main__':
    full_con()


