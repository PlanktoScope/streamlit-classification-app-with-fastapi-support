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
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

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
def load_model(architecture, model_path, num_classes=5):
    # Load the model checkpoint (remove map_location if you have a GPU)
    loaded_cpt = torch.load(model_path, map_location=torch.device('cpu')) 
    # Define the model according to the architecture and modify the number of output classes (by default, no pre-trained weights are used)
    if architecture == "EfficientNet_V2_S":
        model = models.efficientnet_v2_s()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "EfficientNet_V2_M":
        model = models.efficientnet_v2_m()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "EfficientNet_B7":
        model = models.efficientnet_b7()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "ResNet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "DenseNet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif architecture == "VGG16":
        model = models.vgg16()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError("Unsupported architecture")
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
    classif_scores = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    return predicted, classif_scores

# Define the function to save the probabilities to a file
def save_probabilities(probas, filename='classification_scores.txt'):
    with open(filename, 'w') as f:
        for prob in probas:
            f.write(f'{prob[0]}: {prob[1].tolist()}\n')

# Function to generate and display the graph of detected objects
def display_distribution_plot(class_counts, sns_palette="pastel"):
    # Generate seaborn color palette
    seaborn_palette = sns.color_palette(sns_palette)
    # Convert seaborn colors to Plotly-compatible RGBA format
    plotly_colors = ['rgba' + str(tuple(int(255 * c) for c in color[:3]) + (1,)) for color in seaborn_palette]
    # Create a Plotly figure
    fig = go.Figure(data=[go.Bar(y=list(class_counts.values()), x=list(class_counts.keys()), orientation='v', marker_color=plotly_colors, text=list(class_counts.values()), textposition='auto')])
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
    # Set the page configuration
    st.set_page_config(
        page_title="Microorganism Classification",
        page_icon="fairscope_favicon.png",
        layout = 'wide',
    )

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

    # Initialize an empty list to store probabilities
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = []

    with st.sidebar:
        # Add text and link to the sidebar
        st.markdown("""
        ### :rocket: Try this easy-to-follow [notebook](https://colab.research.google.com/drive/1iyoA4jVSI0dErl7N3N-rPlx2mrBrV1ad?usp=drive_link) to train your task-specific classifier
        """)
        # Load the class labels
        class_labels = st.text_input("Enter class labels (comma-separated in alphabetical order)", value="d_veliger, pedi_veliger, umbo_veliger").split(", ")
        # Select the model architecture
        architecture = st.selectbox(
        "Select model architecture",
        ("EfficientNet_V2_M", "EfficientNet_V2_S", "EfficientNet_B7", "ResNet50", "DenseNet121", "VGG16")
        )

    # Select a model to use for image classification
    selected_model = st.file_uploader("Upload a model", type=["pth", "pt", "pb"])

    # Wait for the user to select a model
    if selected_model is not None:
        # Extract the input image dimensions from the model name
        pattern = r'(\d{3,4})x(\d{3,4})'
        image_size = int(re.search(pattern, selected_model.name).group().split("x")[0])
        
        # Load the selected model in pytorch
        model = load_model(architecture, os.path.join("models", selected_model.name), num_classes=len(class_labels))
        st.success('Model loaded successfully!')

    # File uploader for image selection
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg"], accept_multiple_files=True)

    # List to store predicted class labels
    predicted_class_labels = []

    # Display a message indicating images classification part
    text = "<span style='font-size: 14px;'>Classify images</span>"
    st.markdown(text, unsafe_allow_html=True)

    # Create a grid layout of images
    cols = cycle(st.columns(4)) # Ref: https://discuss.streamlit.io/t/grid-of-images-with-the-same-height/10668/8

    if uploaded_files is not None:
        # Iterate over uploaded images and predict their classes
        for (i, uploaded_file) in enumerate(uploaded_files):
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
            predicted_class_index, predicted_classif_scores = predict_image(uploaded_file, model, transform)
            file_name = f"{i}. {uploaded_file.name}"
            st.session_state.probabilities.append((file_name, dict(zip(class_labels,predicted_classif_scores.tolist()[0]))))
            print(predicted_classif_scores)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_labels.append(predicted_class_label)
            
            # Display the uploaded image with the predicted class
            next(cols).image(image, width=150, caption=f"{i}. {predicted_class_label} ({torch.max(predicted_classif_scores):.4f})", use_column_width=True)

        # Save the updated probabilities to a text file
        #save_probabilities(st.session_state.probabilities)

        # Determine the number of detected objects
        num_objects = len(predicted_class_labels)

        # Display the number of detected objects
        st.write(f"Number of detected objects: {num_objects}")

        # Count the occurrences of each class label
        class_counts = {label: predicted_class_labels.count(label) for label in class_labels}

        # Convert probabilities to string format
        probabilities_str = '\n'.join([f"{name}: {scores}" for name, scores in st.session_state.probabilities])
        
        # Plot the distribution of detected objects
        if num_objects > 0:

            # Download the classification scores file with streamlit
            with st.sidebar:
                st.download_button(label="Download classification scores", data=probabilities_str, file_name="classification_scores.txt", mime="text/plain")
            
            # Display the distribution of detected objects
            display_distribution_plot(class_counts)

############################################################################################
# Entry point of the Streamlit app
############################################################################################
if __name__ == "__main__":
    main()
