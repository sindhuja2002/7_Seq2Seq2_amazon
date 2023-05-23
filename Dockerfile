# Use a Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the chatbot script and model files to the container
COPY chatbot.py /app/chatbot.py
COPY seq2seq_model.h5 /app/seq2seq_model.h5
COPY amazon_data.csv /app/amazon_data.csv

# Install the required dependencies
RUN pip install tensorflow==2.7.0 pandas numpy

# Run the chatbot script when the container starts
CMD [ "python", "/app/chatbot.py" ]
