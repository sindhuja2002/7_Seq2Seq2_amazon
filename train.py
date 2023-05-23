import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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

# Pad the input and target sequences to a fixed length
max_sequence_length = max(len(seq) for seq in input_sequences + target_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding='post')

# Split the data into training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_sequences, target_sequences, test_size=0.2)

# Define the seq2seq model architecture
def seq2seq_model(input_vocab_size, output_vocab_size, hidden_units):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, hidden_units)(encoder_inputs)
    _, state_h, state_c = LSTM(hidden_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, hidden_units)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# Define hyperparameters
hidden_units = 256
batch_size = 64
epochs = 10

# Create the seq2seq model
model = seq2seq_model(input_vocab_size, target_vocab_size, hidden_units)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit([input_train, target_train[:, :-1]], target_train[:, 1:],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([input_val, target_val[:, :-1]], target_val[:, 1:]))

# Save the trained model
model.save('seq2seq_model.h5')
