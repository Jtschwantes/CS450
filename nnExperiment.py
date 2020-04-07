import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu))
model.add(tf.keras.layers.Dense(60, activation=tf.nn.leaky_relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 3)
# model.save('mod.model')
# new_model = tf.karas.models.load_model('mod.model)
predictions = model.predict([x_test])
print(np.argmax(predictions[0]))
plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()