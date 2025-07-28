FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_1a.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_1a.txt

# Copy the main script into the container
COPY main.py .

# Define the default command to run when the container starts.
# It will execute the Python script to process PDFs from the mounted
# /app/input directory and save results to the /app/output directory.
CMD ["python", "main_1a.py"]
