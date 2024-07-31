############################################################################################
# Imports (for creating a REST API for the model)
############################################################################################
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uvicorn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

############################################################################################
# Functions to be used by the FastAPI server
############################################################################################
# Define the model loading function
def load_model(model_path, num_classes=6):
    # Load the model checkpoint (remove map_location if you have a GPU)
    loaded_cpt = torch.load(model_path, map_location=torch.device('cpu'))
    # Define the model architecture and modify the number of output classes (by default, no pre-trained weights are used)
    model = models.efficientnet_v2_s()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    # Load the state_dict in order to load the trained parameters 
    model.load_state_dict(loaded_cpt)
    # Set the model to evaluation mode
    model.eval()
    return model

# Define the image reading function
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

# Define the image transformation function
def transform_image(image, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    # Apply image transformations and add batch dimension
    image = transform(image).unsqueeze(0)
    return image

# Define the image prediction function
def predict_image(model, image: Image.Image):
    image_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
    classif_scores = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, 1)
    return predicted, classif_scores

############################################################################################
# Entry point of the FastAPI server
############################################################################################
# Define the FastAPI app object and configure it with the required routes and models
api_server = FastAPI(title='Mussel and Oyster Larvae Classification API',
              description="Obtain model predictions for mussel and oyster larvae images.",
              version='1.0')

# Load the pre-trained model
model_path = "C:/Users/Wassim/Downloads/FairScope/Projet de fin d'études/Dashboard/streamlit classification app with fastapi/models/Effv2s_DA2+fill_256x256_cosan_mussel_oyster.pth"
model = load_model(model_path)

# Define the response model of the API for each image
class Prediction(BaseModel):
    filename: str
    prediction: int
    scores: list

@api_server.post("/predict", response_model=Prediction)
async def get_prediction(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    predicted_class_index, predicted_classif_scores = predict_image(model, image)
    predicted_classif_scores = predicted_classif_scores.tolist()[0]  # Convert tensor to list for response
    return {"filename": file.filename, "prediction": predicted_class_index, "scores": predicted_classif_scores}

if __name__ == "__main__":
    uvicorn.run(api_server, host="0.0.0.0", port=8000)
