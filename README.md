# Streamlit web application for microorganism image classification

## Introduction

This project hosts a Streamlit web application designed to classify objects in images using a pre-trained model. Users can upload images, and the application will display the images along with their predicted classes, and a visual distribution of detection results.

Features:

- Dark and light theme toggling for the application interface (preliminary version).
- Image classification using a pretrained Pytorch model.
- Visualization of the prediction results through an integrated Plotly graph:

<center><img src="plotly_graph.png" alt="Distribution of detected microorganisms" title="Distribution of detected microorganisms" width="400" height="400"/></center>

## Usage

### Prerequisites

Before running this application, make sure you have Docker installed on your system. If you do not have Docker, you can download it from the [official Docker website](https://docs.docker.com/get-docker/).

### Local Docker image build for local testing

#### Cloning the Repository

To clone the repository and navigate into the project directory, run:

    git clone https://github.com/PlanktoScope/streamlit-classification-app.git
    cd streamlit-classification-app
    
#### Creating a folder for pretrained models

To avoid errors later in running the docker container, please import your pretrained models (be sure to include the input image size in its name like sizexsize) and create a folder as follows:

    models/<model_name>


#### Building the Docker Image

Build the Docker image using the following command (it takes a considerable time):

    docker build -t <image_name>:<tag> .

#### Running the Docker Container

Run the Docker container with:

    docker run -p 8501:8501 <image_name>:<tag>

The Streamlit app will be served on port 8501.

#### Accessing the application

Once the application is running, click on the link displayed in your terminal or open your web browser and navigate to:

    http://localhost:8501

From there, you can use the web interface to:

- Upload images for classification.
- View model predictions and the distribution of detected objects.

### Deployment/Testing with Forklift

You can use Forklift to easily deploy the Docker container provided by this repository.

#### Prerequisites

You will need to have the Docker Engine installed on your computer. Installation instructions are
available [here](https://docs.docker.com/engine/install/).

Then, you will need to set up the [`forklift`](https://github.com/PlanktoScope/forklift) tool on
your computer. Setup instructions are available
[here](https://github.com/PlanktoScope/forklift?tab=readme-ov-file#downloadinstall-forklift). Note
that currently `forklift` is only tested for Linux computers, and that Forklift is still an
experimental prototype.

#### First-time deployment

You can clone, stage, and apply the latest commit of this Forklift pallet to your computer, by
using the `forklift` tool:
```
sudo -E forklift pallet switch --apply github.com/PlanktoScope/streamlit-classification-app@main
```

Warning: this will replace all Docker containers on your Docker host with the package deployments
specified by this pallet and delete any Docker containers not specified by this pallet's package
deployments.

If your user is [in the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)
(so that you don't need to use `sudo` when running `docker` commands), then you don't need to use
`sudo` with `forklift`:

```
forklift pallet switch --apply github.com/PlanktoScope/streamlit-classification-app@main
```

When this pallet is updated on GitHub and you want to apply the latest changes on your computer, you
can use the same command as above (either
`forklift pallet switch --apply github.com/PlanktoScope/streamlit-classification-app@main` or that command with
`sudo -E`) to clone, stage, and apply the updated version of the pallet.

#### Subsequent deployment

Because the `forklift` tool uses [Docker Compose](https://docs.docker.com/compose/) to manage the
Docker containers specified by this pallet, the containers will not be running after you restart
your computer (this is true each time you restart your computer); you will need to run a command to
start the containers again:

```
sudo -E forklift stage apply
```

Or if your user is in the `docker` group:

```
forklift stage apply
```

#### Operation

After you have applied the pallet so that the streamlit demo app's container is running, you can
access the streamlit demo app from your web browser at <http://localhost/ps/streamlit-demo>.

Before you can use the streamlit demo app, you will need to download a classification model file
(e.g. <https://github.com/PlanktoScope/streamlit-classification-app/releases/download/models%2Fdemo-1/effv2s_no_norm_DA+sh_20patience_256x256_50ep_loss.pth>)
into `~/.local/share/planktoscope/models`.

## License
This project is licensed under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

Copyright [Wassim Chakroun](http://www.linkedin.com/in/wassim-chakroun/) and PlanktoScope project contributors.
