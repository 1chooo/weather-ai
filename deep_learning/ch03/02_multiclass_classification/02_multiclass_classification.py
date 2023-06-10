# %% [markdown]
# # Classifying newswires: a multiclass classification example

# %%
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data( num_words=10000)

# %%
print('length of training data:', len(train_data))
print('length of testing data:', len(test_data))

# %%
train_data[10]

# %%
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# %%
import numpy as np

def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension)) 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. 
    
    return results
    
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# %% [markdown]
# One-hot encoding is a widely used format for categorical data, also called categorical encoding.

# %%
def to_one_hot(labels, dimension=46):
    
    results = np.zeros((len(labels), dimension)) 
    for i, label in enumerate(labels):
        results[i, label] = 1. 
    return results

one_hot_train_labels = to_one_hot(train_labels) 
one_hot_test_labels = to_one_hot(test_labels)

# %%
from keras.utils.np_utils import to_categorical 

one_hot_train_labels = to_categorical(train_labels) 
one_hot_test_labels = to_categorical(test_labels)

# %% [markdown]
# ### Building the network
# 
# softmax: means the network will output a probability distribution over the 46 different output classes—for every input sample

# %%
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(46, activation='softmax'))

# %%
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# ### Split the data.

# %%
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000] 
partial_y_train = one_hot_train_labels[1000:]

# %%
history = model.fit(
    partial_x_train, 
    partial_y_train,
    epochs=20,      
    batch_size=512, 
    validation_data=(x_val, y_val)
)

# %%
history_dict = history.history
history_dict.keys()

# %%
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %% [markdown]
# ### Retrain the data.

# %%
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(46, activation='softmax')) 
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)


# %%
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# %%
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array)) / len(test_labels)

# %%
predictions = model.predict(x_test)

# %%
print('shape of predictions:', predictions[0].shape)
print('the coefficients in this vector sum:', np.sum(predictions[0]))
print('the class with the highest probability:', np.argmax(predictions[0]))

# %%
y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(
    optimizer='rmsprop', 
    loss='sparse_categorical_crossentropy', 
    metrics=['acc']
)

# %%
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(4, activation='relu')) 
model.add(layers.Dense(46, activation='softmax')) 
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
) 
model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val)
)

# %%


# %% [markdown]
# # Further experiments
# 
# - Try using larger or smaller layers: 32 units, 128 units, and so on.
# - You used two hidden layers. Now try using a single hidden layer, or three hidden layers.
