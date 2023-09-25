FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc

# Sets the working directory for the Dockerfile
WORKDIR /app

# Copies the contents of the current directory into the Docker image. 
COPY . /app

# Creates a virtual environment called venv. This is a separate Python environment that is isolated from the rest of the Docker image
RUN python3 -m venv venv

# Activates the virtual environment and installs the latest version of pip
RUN . venv/bin/activate \
 && pip3 install --upgrade pip

# Installs the Python dependencies from the current directory
RUN pip3 install --no-cache-dir .

# Run the script
CMD ["python", "src/llm_vm/server/main.py"]

