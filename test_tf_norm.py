
import config
import tensorflow as tf


lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_lvalues, dtype=tf.int64)), 0)
rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_rvalues, dtype=tf.int64)), 1)
class_id = tf.convert_to_tensor(0, dtype=tf.int64)
norm_gather_ind = tf.stack([lnorm_table.lookup(class_id), rnorm_table.lookup(class_id)], axis=-1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.tables_initializer())
# print(sess.run(rnorm_table))
print(sess.run(norm_gather_ind))