# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_full.txt

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .

# Run the app. Gunicorn is a production-ready server.
# It runs on port 7860, which is the port Hugging Face Spaces expects.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:server"]