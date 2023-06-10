# %% [markdown]
# # What is a tensor?

# %% [markdown]
# ### 0-D tensors

# %%
import numpy as np

x = np.array(12)
x

# %%
x.ndim

# %% [markdown]
# ### 1-D tensors.

# %%
x = np.array([12, 3, 6, 4])
x

# %%
x.ndim

# %% [markdown]
# ### 2-D tensors.

# %%
x = np.array([[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]
)
x

# %%
x.ndim

# %% [markdown]
# ### 3-D tensors.

# %%
x = np.array([[[5, 78, 2, 34, 0], 
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
   
             [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1], 
              [7, 80, 4, 36, 2]], 
   
             [[5, 78, 2, 34, 0], 
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]]
)
x
# x.shape (3, 3, 5)

# %%
x.ndim

# %% [markdown]
# A tensor is defined by three key attributes:
# - Number of axes (rank)—For instance, a 3D tensor has three axes, and a matrix has two axes. This is also called the tensor’s ndim in Python libraries such as Numpy.
# - Shape—This is a tuple of integers that describes how many dimensions the tensor has along each axis. For instance, the previous matrix example has shape (3, 5), and the 3D tensor example has shape (3, 3, 5). A vector has a shape with a single element, such as (5,), whereas a scalar has an empty shape, ().
# - Data type (usually called dtype in Python libraries)—This is the type of the data contained in the tensor; for instance, a tensor’s type could be float32, uint8, float64, and so on.

# %%
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

# %% [markdown]
# ### Display the forth digit in this 3-D tensor.

# %%
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# %% [markdown]
# ### Manipulating tensor.

# %%
# pick up the digits 10 to 99 -> include 90 digits.
my_slice = train_images[10: 100]
print(my_slice.shape)

# %%
my_slice = train_images[10:100, :, :]
print(my_slice.shape)

# %%
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

# %% [markdown]
# ### Slice the feature data.

# %%
# bottom-right
my_slice = train_images[:, 14:, 14:]
# center
my_slice = train_images[:, 7:-7, 7:-7]

# %% [markdown]
# ### We don't process an entire dataset at once; rather, we will break the data into small batches.

# %%
batch = train_images[:128]
# the next batch
batch = train_images[128:256]

# %% [markdown]
# #### The real-world example data.
# 
# - Vector data—2D tensors of shape (samples, features)
# - Timeseries data or sequence data—3D tensors of shape (samples, timesteps, features)
# - Images—4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
# - Video—5Dtensorsofshape(samples,frames,height,width, channels) or (samples, frames, channels, height, width)


