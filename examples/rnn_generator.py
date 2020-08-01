from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
# from google.colab import files
# https://colab.research.google.com/drive/1ysEKrw_LE2jMndo1snrZUh5w87LQsCxk#forceEdit=true&sandboxMode=true
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# path_to_file = list(files.upload().keys())[0] # Select own file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') # get text
vocab = sorted(set(text)) # get all unique characters as an array in ascending order
char2idx = {u:i for i, u in enumerate(vocab)} # create a dictionary with the character as key and index as value, assigns a char to an int

def text_to_int(text):
  return np.array([char2idx[c] for c in text]) # converts text into an array of ints

idx2char = np.array(vocab) # reverse of char2idx
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

text_as_int = text_to_int(text)
print(int_to_text(text_as_int[:13]))


seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # create dataset from integer values of characters
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) # create batches from the dataset size 101, drop excess characters if needed

def split_input_target(chunk):  # for the example: hello
  input_text = chunk[:-1]  # hell
  target_text = chunk[1:]  # ello
  return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry
# split_input_target shifts the entire sequence by one character, providing a target from an input

BATCH_SIZE = 64 # how much data is fed at a time
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256 # size of embedding layer vector
RNN_UNITS = 1024 # amount of information retained from previous steps

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]), # embedding layer
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True, # Include previous steps, not just most recent
                        stateful=True, # 
                        recurrent_initializer='glorot_uniform'), # LTSM, tensor default.
    tf.keras.layers.Dense(vocab_size) # Choose the next character from pool of all previously used characters
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # using logits means it's not a probability distribution

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( # save weights of model
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(data, epochs=35, callbacks=[checkpoint_callback])

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1) # rebuild model to be able to take a single sequenct of any length
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None])) # change output to be singular rather than a batch of 64

# checkpoint_num = 10 # allows you to choose a specific checkpoint
# model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
# model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 800

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states() # get rid of model's "memory"
  for i in range(num_generate):
    predictions = model(input_eval)
    
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))