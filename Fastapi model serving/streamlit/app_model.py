############################################################################################
# Imports (for Streamlit app & model prediction)
############################################################################################
import streamlit as st # https://docs.streamlit.io/develop/api-reference
import requests
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from itertools import cycle

############################################################################################
# Functions to be used in the Streamlit app
############################################################################################
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
        page_title="Larval Stage Classification",
        page_icon="fairscope_favicon.png",
        layout = 'wide',
    )

    # Set the title of the Streamlit app
    st.title("Classification of Mussel & Oyster Larval Stages")

    # Set the default theme of the plotly chart to light
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

    # Initialize an empty list to store predicted probabilities
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = []

    # Define the class labels
    class_labels = ["Oyster - DV", "Oyster - PV", "Oyster - UV", "Mussel - DV", "Mussel - PV", "Mussel - UV"]

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
            image = Image.open(uploaded_file)

            # Convert image to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format=image.format)
            img_bytes = img_bytes.getvalue()

            # Make API request to perform image classification
            response = requests.post("http://localhost:8000/predict", files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)})
            predicted_class_index = response.json()["prediction"]
            predicted_classif_scores = response.json()["scores"]

            # Perform image classification
            file_name = f"{i}. {uploaded_file.name}"
            st.session_state.probabilities.append((file_name, dict(zip(class_labels, predicted_classif_scores))))
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_labels.append(predicted_class_label)
            
            # Display the uploaded image with the predicted class
            next(cols).image(image, width=150, caption=f"{i}. {predicted_class_label} ({max(predicted_classif_scores):.4f})", use_column_width=True)

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
