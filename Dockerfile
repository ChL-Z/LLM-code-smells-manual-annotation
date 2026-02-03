# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your repo into the container
COPY . .

# Run the server using the shell form so $PORT is expanded
CMD python -m http.server $PORT