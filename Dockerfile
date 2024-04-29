# Use an official Python runtime as the base image
# We use the same base image as what the PlanktoScope segmenter's container image uses:
FROM docker.io/library/python:3.9.18-slim-bullseye

# Install curl for container healthcheck
RUN \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get -y install --no-install-recommends curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  rm -f /tmp/apt-packages

# Set the working directory in the container
WORKDIR /app

# Copy the app files from the server into the container
COPY . .

# Install the required dependencies
RUN \
  pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && \
  pip3 cache purge && \
  rm -rf /root/.cache/pip

# Make port 8501 available to anyone outside this container
EXPOSE 8501

# Add a healthcheck to the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app_model.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
