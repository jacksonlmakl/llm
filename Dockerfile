# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Command to run the application
CMD ["python", "api.py"]

#Manual Docker CLI Build Commands
# docker build -t jacksonmakl/api:latest . --no-cache && docker push jacksonmakl/api:latest
# docker run -p 5000:5000 api
