# Base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port for Jupyter Notebook 
EXPOSE 8888


CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
