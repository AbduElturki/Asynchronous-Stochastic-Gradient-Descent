import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def CNN_model(inputs,graph):
    with graph.as_default():

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.reshape(tf.get_variable('weights',
                                                shape=[5* 5* 3* 48],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                collections=["W_global"]),[5,5,3,48])

            conv = tf.nn.conv2d(inputs,
                                kernel,
                                [1,1,1,1],
                                padding='SAME')

            biases = tf.get_variable('biases',
                                     [48],
                                     initializer=tf.constant_initializer(0.1),
                                     collections=["W_global"])

            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
        
        # pool1
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.reshape(tf.get_variable('weights',
                                                shape=[3* 3* 48* 128],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                collections=["W_global"]),[3,3,48,128])

            conv = tf.nn.conv2d(pool1,
                                kernel,
                                [1,1,1,1],
                                padding='SAME')

            biases = tf.get_variable('biases',
                                     [128],
                                     initializer=tf.constant_initializer(0.1),
                                     collections=["W_global"])

            bias = tf.nn.bias_add(conv,biases)
            conv2 = tf.nn.relu(bias, name=scope.name)

        # pool2
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        with tf.variable_scope('conv3') as scope:
            kernel = tf.reshape(tf.get_variable('weights',
                                                shape=[3* 3* 128* 256],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                collections=["W_global"]),[3,3,128,256])

            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias)

        with tf.variable_scope('conv4') as scope:
            kernel = tf.reshape(tf.get_variable('weights',
                                                shape=[2* 2* 256* 128],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                collections=["W_global"]),[2,2,256,128])

            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)

        # pool3
        pool3 = tf.nn.max_pool(conv4,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool3')

        # Fully connected layer
        with tf.variable_scope('fc') as scope:
            reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
            weights = tf.reshape(tf.get_variable('weights',
                                                 shape=[4*4*128*256],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 collections=["W_global"]),[4*4*128,256])

            biases = tf.get_variable('biases', [256],
                                     initializer=tf.constant_initializer(0.1),
                                     collections=["W_global"])
            FullyCon = tf.nn.relu(tf.matmul(reshape,weights) + biases,name=scope.name)


        #Soft max
        with tf.variable_scope('fc2') as scope:
            weights = tf.reshape(tf.get_variable('weights',
                                                 [256*256],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 collections=["W_global"]),[256, 256])
            biases = tf.get_variable('biases',
                                     [256],
                                     initializer=tf.truncated_normal_initializer(0.1),
                                     collections=["W_global"])
            FullyCon2 = tf.add(tf.matmul(FullyCon, weights), biases,name=scope.name)

        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.reshape(tf.get_variable('weights',
                                                 [256*FLAGS.nb_classes],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                 collections=["W_global"]),[256, FLAGS.nb_classes])
            biases = tf.get_variable('biases',
                                     [FLAGS.nb_classes],
                                     initializer=tf.truncated_normal_initializer(0.1),
                                     collections=["W_global"])
            softmax_linear = tf.nn.softmax(tf.add(tf.matmul(FullyCon2, weights), biases),name=scope.name)


    return softmax_linear
