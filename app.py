import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


# Load the model
# python -m pip install "tensorflow<2.11"

# Load the CNN model
# model = tf.keras.models.load_model("trained_model.h5")
#defining image_to_base64
import base64
from io import BytesIO
model = tf.keras.models.load_model('final_model_62.h5')
model.save('final_model_saved')  # This saves the model in the SavedModel format
model = tf.keras.models.load_model('final_model_saved')

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Showing the logo
with open("logo_1.png", "rb") as icon_image_file:
    icon_image_data = icon_image_file.read()

st.image(icon_image_data, use_column_width=True)

# Custom CSS styling
st.markdown(
    """

    <style>
    /* Set the background color of the entire app */
    body {
        background-color: #E6E6FA;
        font-family: Arial, sans-serif;
    }

    /* Style the sidebar */
    .sidebar .markdown-text-container {
        background-color: #F5F5F5; /* Background color for the sidebar */
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
    }

    .sidebar h2 {
        color: #5045F2; /* Header text color */
        font-size: 20px; /* Header text size */
    }

    .sidebar p {
        color: #333; /* Text color */
        font-size: 14px; /* Text size */
    }

    /* Style the main content area */
    .main-content {
        margin-left: 20px; /* Create some space between sidebar and content */
    }

    /* Style the header */
    .header {
        background-color: #5045F2; /* Header background color */
        color: white; /* Header text color */
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }

    /* Style buttons */
    .stButton {
        background-color: #5045F2; /* Button background color */
        color: white; /* Button text color */
        border-radius: 5px;
        cursor: pointer;
    }

    /* Style buttons on hover */
    .stButton:hover {
        background-color: #322EDF; /* Hover background color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Description
st.markdown("<p class='header' style='font-size:20px;'><strong>Welcome to Dermsage!</strong></p>", unsafe_allow_html=True)

# Get user's name
st.markdown("<br>", unsafe_allow_html=True)
user_name = st.text_input("Enter your name:")

# Upload Image
uploaded_image = st.file_uploader("Drag or upload image here", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_image is not None:
    # Convert RGBA image to RGB
    image = Image.open(uploaded_image).convert("RGB")
    # Resize the image to the desired dimensions
    image_for_prediction = image.resize((224, 224))
    # Convert to NumPy array
    image_for_prediction = np.asarray(image_for_prediction)
    image_for_prediction = image_for_prediction / 255.0  # Normalize the image data
    image_for_prediction = np.expand_dims(image_for_prediction, axis=0)

    # Make predictions
    prediction = model.predict(image_for_prediction)
    prediction = prediction.tolist()
    # Display the prediction
    st.markdown(
        f"<div style='display: flex; justify-content: center;'>"
        f"<img src='data:image/png;base64,{image_to_base64(image)}' alt='Uploaded Image' style='max-width: 200px;'/>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown(
    "<div style='text-align: center;'>"
    "<p style='font-size: 18px; color: green;'>Prediction Complete!</p>"
    "</div>",
    unsafe_allow_html=True
)
    class_names = [
        "Acne / Rosacea",
        "Eczema",
        "Normal Skin",
        "Psoriasis/Lichen Planus",
        "Fungal Infections",
        "Vitiligo",
    ]

    # Display the prediction
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    anchor=""
    
    #conditions for anchor tag
    if predicted_class=="Vitiligo":
        anchor="https://www.mayoclinic.org/diseases-conditions/vitiligo/symptoms-causes/syc-20355912"
    elif predicted_class=="Psoriasis/Lichen Planus":
        anchor="https://www.mayoclinic.org/diseases-conditions/psoriasis/symptoms-causes/syc-20355840#:~:text=Psoriasis%20is%20a%20skin%20disease,make%20it%20hard%20to%20concentrate"
    elif predicted_class=="Acne / Rosacea":
        anchor="https://www.mayoclinic.org/diseases-conditions/acne/symptoms-causes/syc-20368047#:~:text=Overview,but%20acne%20can%20be%20persistent"
    elif predicted_class=="Eczema":
        anchor="https://my.clevelandclinic.org/health/diseases/9998-eczema"
    elif predicted_class=="Fungal Infections":
        anchor="https://www.mayoclinic.org/diseases-conditions/ringworm-body/symptoms-causes/syc-20353780"
    else:
        anchor="#"
    
    st.markdown(
        f"<p style='font-size: 24px; color: #5045F2; text-align: center;'><strong>Hi {user_name}, Thanks for visiting us!</strong></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size: 24px; color: #5045F2; text-align: center;'><b>Here is your predicted skin condition: </b><strong style='color:black;'><u><a href={anchor}>{predicted_class}</a></u></strong></p></p>",
        unsafe_allow_html=True,
    )

    # Brief explanation about the predicted class
    class_descriptions = [
        " Acne is a common skin condition that causes pimples and other blemishes. Rosacea is a chronic skin condition that causes redness and visible blood vessels on the face.",
        " Eczema is a condition that causes the skin to become red, itchy, and inflamed. It often appears as dry, scaly patches on the skin.",
        " This image appears to be of normal skin with no signs of any specific skin condition.",
        " Psoriasis is a chronic skin condition that causes cells to build up rapidly on the surface of the skin. Lichen Planus is an inflammatory skin condition. Both can cause rashes and skin lesions.",
        " These are fungal skin infections that can cause itching, redness, and rashes on the skin.",
        " Vitiligo is a long-term skin condition characterized by patches of the skin losing their pigment. This results in the appearance of white patches on the skin.",
    ]

    # Show description of the predicted class
    st.markdown(
        f"<p style='font-size: 18px; color: #333; text-align: left;'><b>Description:</b>{class_descriptions[predicted_class_index]}</p>",
        unsafe_allow_html=True,
    )
    # Note
    st.markdown(
        "<div style='font-size: 6px; border: 2px solid black; padding: 5px; margin-top: 15px;'><p><strong>Note:</strong> The AI model used in this application is under development. Please use the results as general information and consult a healthcare professional for accurate diagnosis and treatment.</p></div>",
        unsafe_allow_html=True,
    )

# About Dermsage
st.sidebar.markdown("<h2 class='header'>About</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<p>Dermsage is an AI-based tool for preliminary diagnosis of dermatological manifestations. We aim to provide quick skin disease diagnosis.</p>",
    unsafe_allow_html=True,
)

# Benefits
st.sidebar.markdown("<h2 class='header' >Benefits</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<ul><li>Fast and reliable skin disease detection.</li><li>Accessible from anywhere.</li><li>Support for multiple skin conditions.</li></ul>", unsafe_allow_html=True)

# Contact Us
st.sidebar.markdown("<h2 class='header'>Contact Us</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p>If you have any questions or feedback, please email us at dermsage@gmail.com.</p>", unsafe_allow_html=True)

# Privacy Policy
st.sidebar.markdown("<h2 class='header'>Privacy</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<ul><li>We don't save anything.</li><li>We assure your privacy.</li></ul>", unsafe_allow_html=True)

# Additional CSS styles for overall appearance
st.markdown(
    """
    <style>
    body {
        background-color: #E6E6FA;
    }
    .stApp {
        background-color: transparent !important;
    }
    .sidebar .markdown-text-container {
        background-color: #FFF8DC;
    }
    .sidebar h2 {
        color: #FF69B4;
    }
    .sidebar p {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
