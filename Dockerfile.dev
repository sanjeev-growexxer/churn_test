# Use an official Python runtime as the base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the credit card.csv file into the container at /app
COPY Churn_raw_data.csv .

# Copy the churn_test.py script into the container at /app
COPY churn_pred.py .

EXPOSE 80

# Run the churn_test.py script when the container starts
CMD ["python", "churn_pred.py"]