import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense

# Load the training and test data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
)

# Define the base VGG16 model
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Create a new model by specifying the input and output layers
model = tf.keras.Sequential()
model.add(base_model)  # Add the VGG16 base model

# Add a Flatten layer
model.add(Flatten())

# Add a Dense classification layer
model.add(Dense(24, activation="softmax"))

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(train_data, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data)

# Print the test accuracy
print("Test accuracy:", test_accuracy)
