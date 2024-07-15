# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install uWSGI and other dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    build-essential \
    python3-dev \
    libpcre3 \
    libpcre3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to the Docker image and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Remove default Nginx configuration file
RUN rm /etc/nginx/sites-enabled/default

# Copy Nginx configuration file
COPY ./nginx.conf /etc/nginx/nginx.conf

# Copy application code to /app
COPY . /app

# Create directories and set permissions
RUN mkdir -p /var/log/uwsgi
RUN chown -R www-data:www-data /var/log/uwsgi
RUN chmod -R 755 /app

# Expose ports
EXPOSE 80

# Start Gunicorn and Nginx
CMD service nginx start && gunicorn -w 2 -b unix:/app/app/app.sock app.app:app