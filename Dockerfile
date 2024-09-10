# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Run application
CMD ["python", "scripts/satisfaction_analysis.py"]
