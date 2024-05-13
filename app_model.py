############################################################################################
# Imports (for Streamlit app & model prediction)
############################################################################################
import streamlit as st # https://docs.streamlit.io/develop/api-reference
import os
import re
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from itertools import cycle

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_m, efficientnet_v2_s

model_types = {
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_s": efficientnet_v2_s,
}

############################################################################################
# Functions/variables to be used in the Streamlit app
############################################################################################
# Define the theme control function
def set_theme(theme):
    if theme == "dark":
        dark = '''
        <style>
            .stApp {
            background-color: black;
            }
        </style>
        '''
        st.markdown(dark, unsafe_allow_html=True)
    else:
        light = '''
        <style>
            .stApp {
            background-color: white;
            }
        </style>
        '''
        st.markdown(light, unsafe_allow_html=True)

# Define the model loading function
def load_model(model_type, model_path):
    # Load the model checkpoint (remove map_location if you have a GPU)
    loaded_cpt = torch.load(model_path, map_location=torch.device('cpu')) 
    # Define the EfficientNet_V2_M model (by default, no pre-trained weights are used)
    model = model_types[model_type]()
    # Modify the classifier to match the number of classes in the dataset
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    # Load the state_dict in order to load the trained parameters 
    model.load_state_dict(loaded_cpt)
    # Set the model to evaluation mode
    model.eval()
    return model

# Define the image prediction function
def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted

# Function to generate and display the graph of detected objects
def display_distribution_plot(class_counts, sns_palette="pastel"):
    # Generate seaborn color palette
    seaborn_palette = sns.color_palette(sns_palette)
    # Convert seaborn colors to Plotly-compatible RGBA format
    plotly_colors = ['rgba' + str(tuple(int(255 * c) for c in color[:3]) + (1,)) for color in seaborn_palette]
    # Create a Plotly figure
    fig = go.Figure(data=[go.Bar(y=list(class_counts.values()), x=list(class_counts.keys()), orientation='v', marker_color=plotly_colors)])
    fig.update_layout(
        title='Distribution of Detected Objects',
        title_font=dict(size=20),
        xaxis_title='Object',
        yaxis_title='Count',
        xaxis=dict(
            title_font=dict(color='black', size=18),
            tickfont=dict(color='black'),
            showline=True
        ),
        yaxis=dict(
            title_font=dict(color='black', size=18),
            tickfont=dict(color='black'),
            showline=True
        ),
        height=600,
        width=800,
        paper_bgcolor="lightgray",
        margin=dict(pad=0, r=20, t=50, b=60, l=60)
    )
    st.plotly_chart(fig, use_container_width=True)

############################################################################################
# Body of the Streamlit app
############################################################################################
def main():
    # Set the title of the Streamlit app
    st.title("Microorganism Classification")

    # Set the default theme to light
    st.markdown(
        """
        <style>
        .stPlotlyChart {{
            outline: 10px solid #FFFFFF;
            border-radius: 5px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.20), 0 6px 20px 0 rgba(0, 0, 0, 0.30);
        }}
        </style>
        """, unsafe_allow_html=True
    )

    # Create a toggle button
    toggle = st.sidebar.button("Toggle theme", key="theme_toggle")

    # Use a global variable to store the current theme
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    # Change the theme based on the button state
    if toggle:
        if st.session_state.theme == "light":
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"

    # Apply the theme to the app
    set_theme(st.session_state.theme)

    # File uploader for image selection
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg"], accept_multiple_files=True) #"png"

    # List of available AI models
    available_models = os.listdir("models")
    selected_model = st.selectbox("Select a model", available_models)

    # Extract the input image dimensions from the model name
    pattern = r'(\d{3,4})x(\d{3,4})'
    image_size = int(re.search(pattern, selected_model).group().split("x")[0])

    # Load the selected model in pytorch
    model = load_model(
        os.getenv("TORCHVISION_MODEL_TYPE", "efficientnet_v2_m"),
        os.path.join("models", selected_model),
    )

    # Load the class labels
    class_labels = ["Acantharia", "Calanoida", "Neoceratium_petersii", "Ptychodiscus_noctiluca", "Undella"]

    # List to store predicted class labels
    predicted_class_labels = []

    # Display a message indicating images classification part
    text = "<span style='font-size: 14px;'>Classify images</span>"
    st.markdown(text, unsafe_allow_html=True)

    # Create a grid layout of images
    cols = cycle(st.columns(4)) # Ref: https://discuss.streamlit.io/t/grid-of-images-with-the-same-height/10668/8

    if uploaded_files is not None:
        # Iterate over uploaded images and predict their classes
        for uploaded_file in uploaded_files:
            # Read the uploaded image
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            
            # Convert the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Define image transformations
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

            # Perform image classification
            predicted_class_index = predict_image(uploaded_file, model, transform)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_labels.append(predicted_class_label)
            
            # Display the uploaded image with the predicted class
            next(cols).image(image, width=150, caption=f"{uploaded_file.name} ({predicted_class_label})", use_column_width=True)

        # Determine the number of detected objects
        num_objects = len(predicted_class_labels)

        # Display the number of detected objects
        st.write(f"Number of detected objects: {num_objects}")

        # Count the occurrences of each class label
        class_counts = {label: predicted_class_labels.count(label) for label in class_labels}
        
        # Plot the distribution of detected objects
        if num_objects > 0:
            display_distribution_plot(class_counts)

############################################################################################
# Entry point of the Streamlit app
############################################################################################
if __name__ == "__main__":
    main()
