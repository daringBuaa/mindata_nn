import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def random_initial(shape):
    """
    :param shape: 输入形状
    :return: 返回标准差为1，均值为0的一组随机值 
    """
    return tf.random_normal(shape)


def cnn_recognition():
    # 1.准备数据：minist：特殊的读取方式batch = mnist.train.next_batch (batch_size)
    # mnist 为元祖
    mnist = input_data.read_data_sets("MNIST_data/mnist/", one_hot=True)
    with tf.variable_scope('variable'):
        # 准备测试数据集28*28*1,特征值数量为784;None表示不确定样本个数
        x_train = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_train')
        # 手写数字为0-9共10个数字
        y_true = tf.placeholder(tf.float32, [None, 10], name='label')

    with tf.variable_scope('convb'):
        # 卷积1[None, 28, 28, 1] ---> [None, 28, 28,32]
        # 设置filter：为一组随机权重，[5, 5, 1, 32]:height,width,in_channel,out_channel
        filter1 = tf.Variable(random_initial([5, 5, 1, 32]))
        bias_conv1 = tf.Variable(random_initial([32]))
        # 此时padding=‘SAME’表示，输入输出的height、width相同
        x_filter1 = tf.nn.conv2d(x_train, filter=filter1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1
        # 使用relu激活函数：因为使用sigmoid激活函数，计算量大，且在backward propagation容易梯度爆炸
        x_relu1 = tf.nn.relu(x_filter1)
        # 使用max_pool池化：[2,2]，步长为2：[None, 28, 28, 32] -->[None, 14, 14, 32]
        # 此时的padding=‘SAME’表示是否在不够的时候继续读取
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 卷积2 [None, 14, 14,32] --->[None, 14, 14, 64]
        filter2 = tf.Variable(random_initial([5, 5, 32, 64]))
        bias_convb2 = tf.Variable(random_initial([64]))
        x_filter2 = tf.nn.conv2d(x_pool1, filter=filter2, strides=[1, 1, 1, 1], padding='SAME') + bias_convb2
        # 激活函数relu
        x_relu2 = tf.nn.relu(x_filter2)
        # 池化 [None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('full_connected'):
        # 初始化全连接层变量 weight 维度应为[7*7*64,10]
        value_initial = random_initial([7*7*64, 10])
        weight = tf.Variable(value_initial, dtype=tf.float32, name='w')
        bias = tf.Variable(random_initial([10]), dtype=tf.float32, name='bias')
        # 开始计算
        x_input = tf.reshape(x_pool2, [-1, 7*7*64])
        y_predict = tf.matmul(x_input, weight) + bias

    with tf.variable_scope('optimizer'):
        # soft_max激活函数，并计算交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                              (labels=y_true, logits=y_predict))
        # 使用梯度下降优化
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.variable_scope('accuracy'):
        # 计算每步的识别准确率;tf.equal()返回equal_list,
        equal_list = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_true, 1))
        # 必须转化为list of allowed values: float32, float64, int32, uint8, int16, int8,
        # complex64,int64, qint8, quint8, qint32, bfloat16, uint16,
        # complex128, float16, uint32, uint64中的一个才能输入reduce_mean()求平均值
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    tf.summary.scalar('losses', loss)
    tf.summary.scalar('acc', accuracy)

    tf.summary.histogram('weight', weight)
    tf.summary.histogram('bias', bias)
    # 存在变量需要初始化变量
    init_op = tf.global_variables_initializer()
    # 合并
    merge = tf.summary.merge_all()
    # 保存模型结果
    saver = tf.train.Saver(var_list=[weight, bias], max_to_keep=3)

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        file = tf.summary.FileWriter('D:/人工智能/神经网络/tensorboard/hand_writting', sess.graph)
        for i in range(1000):
            # x为一行784列[None,784]；train_lable:[None, 10]
            x, train_label = mnist.train.next_batch(80)
            # 开始训练,此时x_train 要求为[None, 28 ,28 , 1]
            train_x = x.reshape([-1, 28, 28, 1])
            sess.run(train_op, feed_dict={x_train: train_x, y_true: train_label})
            acc = sess.run(accuracy, feed_dict={x_train: train_x, y_true: train_label})
            summary = sess.run(merge, feed_dict={x_train: train_x, y_true: train_label})
            # 打印每步的训练准确率
            file.add_summary(summary, i)
            print('第{0}步，识别准确率为{1}'.format(i, acc))
            # 每1000步保存一次模型
            if i % 300 == 0:
                saver.save(sess, 'model/cnn_handwriter.ckpt')
                print("保存成功！")


if __name__ == '__main__':
    cnn_recognition()




