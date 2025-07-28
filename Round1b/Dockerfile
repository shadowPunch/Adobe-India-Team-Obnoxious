FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install build tools and git, which are necessary for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the optimization script into the container
COPY optimize_model.py .

# Run the optimization script to download and quantize the model
# The final artifacts will be saved to the ./model directory inside the image.
RUN python optimize_model.py

# Copy the main application script
COPY main.py .

# Download the NLTK 'punkt' tokenizer data
RUN python -c "import nltk; nltk.download('punkt')"

# Define the command to run the application when the container starts
# It expects 'input' and 'output' directories to be mounted from the host.
CMD ["python", "main.py"]
