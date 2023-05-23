import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


# Load the trained seq2seq model
model = load_model('seq2seq_model.h5')


# Load the scraped data from the CSV file
data = pd.read_csv('amazon_data.csv')

# Prepare the input and target sequences
input_texts = data['Title'].values
target_texts = data['Price'].values

# Tokenize the input and target sequences
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(input_texts)
input_sequences = tokenizer_input.texts_to_sequences(input_texts)
input_vocab_size = len(tokenizer_input.word_index) + 1

tokenizer_target = Tokenizer()
tokenizer_target.fit_on_texts(target_texts)
target_sequences = tokenizer_target.texts_to_sequences(target_texts)
target_vocab_size = len(tokenizer_target.word_index) + 1

# Load the tokenizer used for training
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(input_texts)

tokenizer_target = Tokenizer()
tokenizer_target.fit_on_texts(target_texts)

# Define the maximum sequence length
max_sequence_length = max(len(seq) for seq in input_sequences + target_sequences)

# Function to generate a response from the input query
def generate_response(query):
    # Convert the query to a sequence of integers
    query_sequence = tokenizer_input.texts_to_sequences([query])
    query_sequence = pad_sequences(query_sequence, maxlen=max_sequence_length, padding='post')

    # Generate the initial decoder input
    decoder_input = np.zeros((1, max_sequence_length))
    decoder_input[0, 0] = tokenizer_target.word_index['start']

    # Generate the response word by word
    response = ''
    while True:
        # Predict the next word
        predictions = model.predict([query_sequence, decoder_input])
        next_word_index = np.argmax(predictions[0, -1, :])
        next_word = tokenizer_target.index_word[next_word_index]

        # Append the predicted word to the response
        if next_word == 'end' or len(response.split()) >= max_sequence_length:
            break
        response += next_word + ' '

        # Update the decoder input for the next iteration
        decoder_input[0, len(response.split())] = next_word_index

    return response

# Test the chatbot
while True:
    user_input = input('User: ')
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print('ChatBot:', response)
