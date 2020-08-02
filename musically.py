import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import random

'''processing the dataset'''

# load songs from disk
songs_joined = open("dataset.txt").read()

# find all unique characters in the string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset\n")

'''vectorizing the dataset'''

# dictionary that has characters as keys and corresponding indices as values
#lets us access a character's index
char2idx = {u:i for i, u in enumerate(vocab)}

#array that lets us access characters by their indices
idx2char = np.array(vocab)

#convert each character in input to a numerical value
def vectorize(string):
    return np.array([char2idx[char] for char in string])

'''tokenization'''

songs = songs_joined.split("\n\n")  #separate songs
song_keys = ([vectorize( "\n".join(song.split("\n")[6:]) ) for song in songs if song])   #discard metadata (first 6 lines) then vectorize each song. discard empty songs as well
#song_keys = ([vectorize(song) for song in songs if song])   #vectorize each song after discarding empty songs

#return a list of input and output sequences
#suppose the text is "Hello", the input sequence is "Hell" and the output sequence is "ello", 
#since the rnn is trying to predict the next character at each timestep
def get_batch(seq_length, batch_size = len(song_keys)):
    if batch_size > len(song_keys):
        print("get_batch(): batch_size exceeds the number of songs. There are only " + str(len(song_keys)) + " songs")
        batch_size = len(song_keys)
    min_seq_length = min([song.shape[0] for song in song_keys])
    if seq_length > min_seq_length:
        print("get_batch(): seq_length is too large. Shortest sequence is " + str(min_seq_length) + " characters long")
        seq_length = min_seq_length

    sequences = [sequence[:seq_length] for sequence in song_keys]   #trim sequences
    batch_indices = np.random.choice(len(song_keys), batch_size) # randomly pick the indices that will form the batch
    batch = [sequences[index] for index in batch_indices]   #create the batch
    input_batch = [song[:-1] for song in batch] #slice one key from the end of each song
    input_batch = np.vstack(input_batch)    #make list of songs a numpy matrix with seq_length-1 rows and batch_size elements in each row
    output_batch = [song[1:] for song in batch] #slice one key from the beginning of each song
    output_batch = np.vstack(output_batch)    #make list of songs a numpy matrix with seq_length-1 rows and batch_size elements in each row
    #print(output_batch.shape)
    return input_batch, output_batch

'''defining the rnn model'''

#we'll use the sequential api from keras to build the model
#the model has the following layers;
    #tf.keras.layers.Embedding: This is the input layer, consisting of a trainable lookup table that maps the numbers of each character to a vector with embedding_dim dimensions.
    #tf.keras.layers.LSTM: Our LSTM network, with size units=rnn_units.
    #tf.keras.layers.Dense: The output layer, with vocab_size outputs.

def build_rnn(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units. 
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', recurrent_activation='sigmoid', stateful=True),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output into the vocabulary size. 
        tf.keras.layers.Dense(vocab_size)
    ])

'''defining the loss function'''

#To train our model on this classification task, we can use a form of the crossentropy loss (negative log likelihood loss). 
#Specifically, we will use the sparse_categorical_crossentropy loss, as it utilizes integer targets for categorical classification tasks. 
#We will want to compute the loss using the true targets -- the labels -- and the predicted targets -- the logits
def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

'''hyperparameters for training the model'''

# Optimization parameters:
epochs = 2000  # Increase this to train longer
#epochs = int(input("Enter the number of iterations to train for: "))  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 300  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

'''defining optimizer and training operation'''

model = build_rnn(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate) #Adam optimizer

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        #feed the current input into the model and generate predictions
        y_hat = model(x)
    
        #compute the loss. labels - y, logits - y_hat
        loss = compute_loss(y, y_hat)

    #compute gradients 
    #we want the gradient of the loss w.r.t of the model parameters
    #model.trainable_variables returns a list of all model parameters
    grads = tape.gradient(loss, model.trainable_variables)
    
    #apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

'''training the model😁'''

def train():
    loss_history = []
    prev_time = time.time()
    time_elapsed = 0
    for epoch in range(epochs):
        #grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(seq_length, batch_size)
        loss = train_step(x_batch, y_batch)

        #update the loss history
        loss_history.append(loss.numpy().mean())

        #update the model with the changed weights after every 100 epochs
        if epoch % 100 == 0:     
            model.save_weights(checkpoint_prefix)

        time_elapsed += time.time() - prev_time
        prev_time = time.time()
        print("Epoch {} of {}. Loss: {}. Time elapsed: {} seconds.".format(epoch+1, epochs, loss.numpy().mean(), time_elapsed))
        
    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)

    #plot a graph that will show how our loss varied with time
    plt.plot(range(epochs), loss_history)
    plt.title(__file__)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

#train()

'''restoring the last checkpoint'''

#here we load the weights previously written to disk into the model
#this type of model accepts a fixed batch size once it is built
#we use a batch size of 1 to generate one song at a time
#we'll have to rebuild the model to accept inputs of batch_size = 1

batch_size = 1
model = build_rnn(vocab_size, embedding_dim, rnn_units, batch_size) #rebuild model

#restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#model.summary()

'''time to generate songs 😃'''

#procedure:
    # Initialize a "seed" start string and the RNN state, and set the number of characters we want to generate.
    # Use the start string and the RNN state to obtain the probability distribution over the next predicted character.
    # Sample from multinomial distribution to calculate the index of the predicted character. This predicted character is then used as the next input to the model.
    # At each time step, the updated RNN state is fed back into the model, so that it now has more context in making the next prediction. 
    # After predicting the next character, the updated RNN states are again fed back into the model, which is how it learns sequence dependencies in the data, as it gets more information from the previous predictions.

def generate_text(model, start_string, generation_length=1000):
    input_eval = vectorize(start_string)
    input_eval = tf.expand_dims(input_eval, 0)    #add extra dimension for the batch
    text_generated = []   #empty string to store our results
    model.reset_states()  #clear hidden states

    for i in range(generation_length):
        predictions = model(input_eval)   #predict output sequence
        predictions = tf.squeeze(predictions, 0)  #remove the first dimension
        predictions = tf.random.categorical(predictions, num_samples=1)   #from each output vector, returns index of neuron with highest activation
        predictions = tf.squeeze(predictions, axis=-1).numpy()    #remove the last dimension
        predicted_id = predictions[-1]  #get the id of the last character from the sequence - the one the rnn has predicted

        #pass the prediction along with the previous hidden state as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        
        #add the predicted character to the generated text
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

#parameters for song generation
params = [
    {
        'metadata': "X:127\nT:The Twist\nZ: id:dc-reel-117\nM:C\nL:1/8\nK:D Mixolydian\n",
        'start_string': "A,|DEFD EFGB|",
        'generation_length': 244
    },
    {
        'metadata': "X:140\nT:Babylon\nZ: id:dc-reel-129\nM:C\nL:1/8\nK:E Dorian\n",
        'start_string': "BA|G3A BEE2|",
        'generation_length': 116
    },
    {
        'metadata': "X:2\nT:London Bridge\nZ: id:dc-hornpipe-2\nM:C|\nL:1/8\nK:G Major\n",
        'start_string': "GF|DGGB d2GB|",
        'generation_length': 167
    },
    {
        'metadata': "X:160\nT:Lindier's Favourite\nZ: id:dc-reel-147\nM:C\nL:1/8\nK:D Major\n",
        'start_string': "FE|DFFd AFFA|",
        'generation_length': 381
    },
    {
        'metadata': "X:161\nT:The Crusaders\nZ: id:dc-reel-148\nM:C\nL:1/8\nK:F# Minor\n",
        'start_string': "f2ae fece|f2ae fece|",
        'generation_length': 157
    },
    {
        'metadata': "X:150\nT:A Phantom\nZ: id:dc-reel-138\nM:C\nL:1/8\nK:D Dorian\n",
        'start_string': "Adde f2ed|cAGc AcGc|",
        'generation_length': 117
    },
    {
        'metadata': "X:10\nT:Dorylaeum\nZ: id:dc-hornpipe-9\nM:C|\nL:1/8\nK:A Major\n",
        'start_string': "(3EFG|AGAB cBAG|",
        'generation_length': 182
    },
    {
        'metadata': "X:50\nT:Sherlock\nZ: id:dc-hornpipe-45\nM:C|\nL:1/8\nK:E Minor\n",
        'start_string': "B,|E3F G3A|",
        'generation_length': 157
    },
    {
        'metadata': "X:163\nT:Spaceman\nZ: id:dc-jig-140\nM:6/8\nL:1/8\nK:B Minor\n",
        'start_string': "A|FBB BAB|",
        'generation_length': 123
    },
    {
        'metadata': "X:7\nT:The Shogun\nZ: id:dc-ocarolan-19\nM:C\nL:1/8\nK:F Major\n",
        'start_string': "eg|f2c2 fedc|",
        'generation_length': 327
    }
]

def create_song(name):
    param = params[random.randint(0, len(params) - 1)]
    file = open(name + ".abc", "w")
    file.write(param['metadata'] + generate_text(model, param['start_string'], param['generation_length']))
    file.close()
    os.system("abc2midi {0}.abc -o {0}.mid".format(name))

create_song(input("Enter the name of the song to be generated: "))
