# importing necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import urllib.request

def resize_image(image_path):
  '''
  Function to resize an image into 128 *128
  '''
  resized_image = cv2.resize(image_path, (128,128), interpolation = cv2.INTER_AREA) # cv2.INTER_AREA: This is used when we need to shrink an image.

  return resized_image

def predict(image_path):
    # Loading model
    import urllib.request

def predict(image_path):
    # Load the model file from GitHub
    #model_url = "https://raw.githubusercontent.com/princebari/Malaria-Disease-Detection-Using-Transfer-Learning/main/InceptionV3.h5"
    #model_path = "InceptionV3.h5"
    #urllib.request.urlretrieve(model_url, model_path)

    # Loading model
    model = load_model("InceptionV3.h5")

    # Rest of your code...

    # Make prediction
    pred = model.predict(input_image)

    # Convert the prediction to a label
    if pred[0] > 0.5:
        label = 'Parasitized Cell'
    else:
        label = 'Uninfected Cell'
    return label
    model = load_model('InceptionV3.h5')

    # Read image
    image = imageio.imread(image_path)

    # Resize image into 128x128
    resized_image = resize_image(image)

    # Convert into numpy array
    image_array = np.array(resized_image)

    # Rescale image (0-1)
    rescaled_image = image_array.astype(np.float32) / 255.0

    # Expanding dimensions
    input_image = np.expand_dims(rescaled_image, axis=0)

    # Make prediction
    pred = model.predict(input_image)

    # Convert the prediction to a label
    if pred[0] > 0.5:
        label = 'Parasitized Cell'
    else:
        label = 'Uninfected Cell'
    return label


def main():
    # Set background image
    page_bg_img = '''
        <style>
        body {
            background-image: url("https://cdn4.vectorstock.com/i/1000x1000/73/38/malaria-on-orange-background-style-vector-14187338.jpg");
            background-size: cover;
            opacity: 0.95;
        }
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Malaria Disease Detection")
    st.write("Welcome to the Malaria Disease Detection! Please upload an image of a cell, and we will predict if it is parasitized or uninfected. Once the image is uploaded, it will be displayed on the screen. Our model will then make a prediction, and the result will be shown below the image.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display uploaded image
        image = imageio.imread(uploaded_file)

        # making image smaller to show
        smaller_image = cv2.resize(image, (400, 400))  # Resize the image to a smaller size
        st.image(smaller_image, caption='Uploaded Cell Image')

        # Make prediction
        prediction = predict(uploaded_file)
        prediction = "<h3 style='font-family: Arial;'>Prediction: " + prediction + "</h3>"
        st.write(prediction, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
