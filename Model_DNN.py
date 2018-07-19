import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def CNN_model(inputs,graph):
    with graph.as_default():

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.reshape(tf.get_variable('weights', shape=[3* 3* 1* 32],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[3,3,1,32])
            
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [32],initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            
        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')


        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = tf.reshape(tf.get_variable('weights', shape=[3* 3* 32* 64],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[3,3,32,64])
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            
        
        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('fc3') as scope:
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            weights = tf.reshape(tf.get_variable('weights', shape=[7*7*64*128],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[7*7*64,128])
            biases = tf.get_variable('biases', [128], initializer=tf.constant_initializer(0.1),collections=["W_global"])
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            


        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.reshape(tf.get_variable('weights', [128*FLAGS.nb_classes],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1),collections=["W_global"]),[128,FLAGS.nb_classes])
            biases = tf.get_variable('biases', [FLAGS.nb_classes],initializer=tf.constant_initializer(0.1),collections=["W_global"])
            softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
            

    return softmax_linear

