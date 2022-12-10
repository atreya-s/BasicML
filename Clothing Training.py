import tensorflow as tf

# Loading the fashion MNIST dataset through pre-existing dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalizing the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training the model
model.fit(x_train, y_train, epochs=5)

# Evaluating the model on the test set
model.evaluate(x_test, y_test)
