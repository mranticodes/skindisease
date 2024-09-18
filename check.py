import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('final_model_62.h5')

# Load and preprocess the image
def preprocess_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image to match the input shape required by the model
    image = image.resize((224, 224))  # Assuming your model expects 224x224 images
    
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    
    # Normalize the pixel values (optional, depending on how your model was trained)
    image_array = image_array / 255.0
    
    # Add an extra dimension for batch size (the model expects input shape [batch_size, height, width, channels])
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Test the image
image_path = 'acne.jpeg'  # Replace with the path to your test image
processed_image = preprocess_image(image_path)

# Make the prediction
prediction = model.predict(processed_image)

# Class names (ensure these match the order used during model training)
class_names = [
    "Acne / Rosacea",
    "Eczema",
    "Normal Skin",
    "Psoriasis/Lichen Planus",
    "Fungal Infections",
    "Vitiligo"
]

# Get the index of the predicted class
predicted_class_index = np.argmax(prediction)

# Get the predicted class label
predicted_class = class_names[predicted_class_index]

# Print the prediction
print(f"Predicted class: {predicted_class}")
