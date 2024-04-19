# Use an official Python runtime as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the app files from the server into the container
COPY . .

# Install the required dependencies
RUN pip3 install -r requirements.txt

# Make port 8501 available to anyone outside this container
EXPOSE 8501

# Add a healthcheck to the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the command to run the Streamlit app (what the image will do when it starts as a container)
#CMD ["streamlit", "run", "app_model.py"]

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app_model.py", "--server.port=8501", "--server.address=0.0.0.0"]