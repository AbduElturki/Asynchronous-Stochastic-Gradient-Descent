import tensorflow as tf
import Tf_op as df
import socket as sck
import numpy as np
import Model_DNN as mdnn
import Communication as com
import threading

FLAGS = tf.app.flags.FLAGS

# Parameter Server routine
def PS():
    with tf.Graph().as_default() as graph:
        # Get input and labels for learning from D
        inputs, labels = tf.placeholder(tf.float32,shape=[None,FLAGS.image_size,FLAGS.image_size,FLAGS.image_depth]), tf.placeholder(tf.float32,shape=[None,FLAGS.nb_classes])
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = mdnn.CNN_model(inputs,graph)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        init_glob = tf.variables_initializer(tf.get_collection("W_global"))
        update_op = df.apply_sparse_update(graph,"W_global")
        with tf.Session(graph=graph) as sess:
            # Initialize the Deep Neural Network
            sess.run([init,init_glob])

            # Configure socket
            tcpsock = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
            tcpsock.setsockopt(sck.SOL_SOCKET, sck.SO_REUSEADDR, 1)
            tcpsock.bind(("",FLAGS.port))
            tcpsock.settimeout(None)

            history = []
            iteration = 0

            while iteration <FLAGS.iter_max+FLAGS.nb_workers-1:
                while 1:
                    #print "waiting to listen"
                    tcpsock.listen(1)
                    #print wsocket
                    threading.Thread(target=listenToWorker,
                                     args=(tcpsock,sess,graph,iteration,update_op)).run()
                    #print "done listening"
    print "PS is closed"

def listenToWorker(tcpsock,sess,graph,iteration,update_op):
    (wsocket, (ip, port)) = tcpsock.accept()
    cmd,data = com.recv_msg(wsocket)
    if cmd == "GET_W":
        #Encode parameter
        parameters = com.encode_variables(sess,"W_global",iteration,compression=1)
        com.send_msg(wsocket,parameters,"PARAM")
        wsocket.close()
    elif cmd =="PUSH":
        old_iter,gradients = com.decode_variables(data)
        new_gradients = {}
        # for each trainable variable of the model
        for k in gradients.keys():
            #Multiple with learning rate
            new_gradients[k] = np.asarray([FLAGS.learning_rate*gradients[k][i]
                                           for i in range(len(gradients[k]))])

        # Update paramters
        feed_dict = {}
        for k in gradients.keys():
            feed_dict[k[:-5]+"_delta:0"]=new_gradients[k]

        sess.run(update_op,feed_dict)
        # Add update to history
        iteration+=1
        cmd,data = com.recv_msg(wsocket)
        if cmd == "GET_W":
            #Encode parameter
            parameters = com.encode_variables(sess,"W_global",iteration,compression=1)
            com.send_msg(wsocket,parameters,"PARAM")
            wsocket.close()
        else:
            wsocket.close()
