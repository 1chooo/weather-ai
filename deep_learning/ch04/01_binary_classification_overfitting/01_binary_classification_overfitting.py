# %% [markdown]
# # Two-class Classification
# 
# Deal with overfitting

# %% [markdown]
# ### Import the IMDB dataset

# %%
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %%
print(train_data[0])
print(train_labels[0])

# %%
# max([max(sequence) for sequence in train_data])

# %%
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
)

# %%
# word_index.items()

# %% [markdown]
# ### Preparing the data.

# %%
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. 
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# %%
y_train = np.asarray(train_labels).astype('float32') 
y_test = np.asarray(test_labels).astype('float32')

# %% [markdown]
# ### Building the Network
# 
# relu: meant to zero out negative values
# 
# sigmoid: “squashes” arbitrary values into the [0, 1] interval

# %%
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

# %% [markdown]
# ### Reduce to the smaller size.

# %%
model.add(layers.Dense(4, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(4, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

# %% [markdown]
# #### Three methods to compile the model.

# %%
model.compile(
    optimizer='rmsprop', 
    loss='binary_crossentropy', 
    metrics=['accuracy'],
)

# %%
from keras import optimizers 

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# %%
from keras import losses
from keras import metrics 

model.compile(
    optimizer=optimizers.RMSprop(lr=0.001), 
    loss=losses.binary_crossentropy, 
    metrics=[metrics.binary_accuracy]
)

# %%
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# %%
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
) 
history = model.fit(
    partial_x_train, 
    partial_y_train,
    epochs=20,
    batch_size=512, 
    validation_data=(x_val, y_val)
)

# %% [markdown]
# ### Visualize the training progress.

# %%
history_dict = history.history
history_dict.keys()

# %%
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Train History")
plt.ylabel("loss")
plt.xlabel("Epoch")

plt.legend(["loss", "val_loss"], loc = "upper left")
plt.show()

# %%
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 

plt.show()

# %%
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc'] 
plt.plot(epochs, acc_values, 'bo', label='Training acc') 
plt.plot(epochs, val_acc_values, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% [markdown]
# ### Retrain a model

# %%
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid')) 
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    x_train, 
    y_train, 
    epochs=4, 
    batch_size=512
)

# %%
results = model.evaluate(x_test, y_test)
print(results)

# %%
model.predict(x_test)

# %% [markdown]
# # Further experiments
# 
# The following experiments will help convince you that the architecture choices you’ve made are all fairly reasonable, although there’s still room for improvement:
# - You used two hidden layers. Try using one or three hidden layers, and see how doing so affects validation and test accuracy.
# - Try using layers with more hidden units or fewer hidden units: 32 units, 64 units, and so on.
# - Try using the mse loss function instead of binary_crossentropy.
# - Try using the tanh activation (an activation that was popular in the early days of neural networks) instead of relu.
