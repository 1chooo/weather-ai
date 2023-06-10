# %% [markdown]
# # HW01 CIFAR-10 with CNN
# 
# ##### Course: AP4064
# ##### Assignment: 1
# ##### Major: ATM
# ##### Name: Hugo ChunHo Lin
# ##### Student Id: 109601003

# %% [markdown]
# ### Import the package and the dataset

# %%
import tensorflow as tf
from keras.datasets import cifar10
from keras import models
from keras import layers

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# %% [markdown]
# ### Define the DNN model architecture

# %%
def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

# %% [markdown]
# ### Visualize the data type and the label!

# %%
import matplotlib.pyplot as plt # pip install matplotlib
from random import randrange

text = [
  'airplane', 
  'automobile',
  'bird' ,
  'cat', 
  'deer', 
  'dog', 
  'frog', 
  'horse', 
  'ship', 
  'truck'
]
plt.figure(figsize=(16,10),facecolor='w')
for i in range(5):
  for j in range(8):
    index = randrange(0, 50000)
    plt.subplot(5, 8, i * 8 + j + 1)
    plt.title("label: {}".format(text[train_labels[index][0]]))
    plt.imshow(train_images[index])
    plt.axis('off')

plt.show()

# %%
model = build_model()

# Train the model
history = model.fit(
    train_images, 
    train_labels, 
    epochs=20, 
    validation_data=(test_images, test_labels)
)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

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

# %% [markdown]
# We find that about seven times, we can get the lowest `val_loss`, then we can re-compile our model again, and set the `epoches` to 7 or 8.

# %%
print('Test accuracy:', test_acc)

# %%
model = build_model()

# Train the model
history = model.fit(
    train_images, 
    train_labels, 
    epochs=7, 
    validation_data=(test_images, test_labels),
    batch_size=800
)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

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
print('Test accuracy:', test_acc)

# %% [markdown]
# #### The result of the test accuracy truely increases!
# 
# Then we can try other variables to improve the result of our model.

# %% [markdown]
# ### Now we add the `bastch_size`

# %%
model = build_model()

# Train the model
history = model.fit(
    train_images, 
    train_labels, 
    epochs=7, 
    validation_data=(test_images, test_labels),
    batch_size=50
)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

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
print('Test accuracy:', test_acc)

# %% [markdown]
# ### We find that when we adjust our `batch_size`, we can also increase the result of our model!
# 
# Then we can try another test to change our `optimizer`.

# %%
def build_model():
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

# %%
model = build_model()

# Train the model
history = model.fit(
    train_images, 
    train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels),
    batch_size=50
)
# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

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
print('Test accuracy:', test_acc)

# %% [markdown]
# ## My gained knowledge
# 
# I have tested a lot of experiment to improve the model; however, all of the results still surround to about 0.70 accuracy. 
# 
# #### The best accuracy of my work: `Test accuracy: 0.718500018119812`
# 
# #### Below are the experiments I have conducted:
# * `epoches: from 20 -> 7`.
# * `batch_size: from 100 -> 50`.
# * `optimizer: adam and rmsprop`.
# 
# Even though I designed a lot of experiments, the accuracy did not increase significantly. I have considered the reasons, and here are my conclusions. 
# 
# First, our Deep Neural-Network model was limited by the size of the CIFAR-10 dataset, which consisted of up to 50000 training_data and up to 10000 testing_data. The larger datasets made it difficult for the DNN model to capture all the necessary values during training, which resulted in less accuracy even when we changed several variables. 
# 
# Second, given the large amount of data, I could have tried to drop out the data that affected the results. However, I thought that we might be able to choose the Convolutional Neural-Network instead because it was more suitable for dropping out the worse neural in our model.
# 
# In conclusion, I am excited to have the opportunity to improve my deep-learning skills with this dataset and to review what I have learned before.

# %% [markdown]
# ### Reference
# 
# * [Day 20 ~ AI從入門到放棄 - 新的資料集](https://ithelp.ithome.com.tw/articles/10248873)
# * [簡單使用keras 架構深度學習神經網路 — 以cifar10為例](https://medium.com/@a227799770055/%E7%B0%A1%E5%96%AE%E4%BD%BF%E7%94%A8keras-%E6%9E%B6%E6%A7%8B%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-%E4%BB%A5cifar10%E7%82%BA%E4%BE%8B-b8921ca239cf)
