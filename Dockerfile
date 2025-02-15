# Use an official Python runtime as the base image
FROM python:3.12.9-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model files
COPY churn_model.pkl .
COPY scaler.pkl .
COPY boruta_features.pkl .


# Copy the application code
COPY main.py .
COPY prediction_pipeline.py .
COPY data_preparation_.py .
COPY feature_selection.py .
COPY quantile_bins.pkl .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]