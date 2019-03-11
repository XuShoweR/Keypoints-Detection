import tensorflow as tf

tf_initial_ones = tf.ones((1, 4, 4, 1), dtype=tf.float64)
result = tf.layers.conv2d_transpose(tf_initial_ones, 4, 3, (3, 3), padding='same')
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result_final = sess.run(result)
print(result_final, result_final.shape)