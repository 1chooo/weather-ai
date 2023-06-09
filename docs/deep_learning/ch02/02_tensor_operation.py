# %% [markdown]
# # The gear of neural network.

# %%
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
import keras
from keras import models
from keras import layers

# %%
keras.layers.Dense(512, activation='relu')

# %%
# output = relu(dot(W, input) + b)

# %%
def naive_relu(x):
    """_keep the number greater than zero._

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(x.shape) == 2
    x = x.copy()
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
          x[i, j] = max(x[i, j], 0)

    return x

# %%
def naive_add(x, y):

    """_x+y_

    Returns:
        _type_: _description_
    """
    
    assert len(x.shape) == 2 
    assert x.shape == y.shape
    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]): x[i, j] += y[i, j]

    return x

# %%
import numpy as np 

x = np.random.random((64,  3)) 
y = np.random.random((32, 10))
z = x + y
z = np.maximum(z, 0.)

# %% [markdown]
# ### Broadcasting
# let small tensors match the larger tensor.

# %%
def naive_add_matrix_and_vector(x, y): 
    
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]): x[i, j] += y[j]
    
    return x

# %%
import numpy as np

x = np.random.random((64,  3, 32, 10)) 
y = np.random.random((32, 10))
z = np.maximum(x, y)

# %% [markdown]
# ### Dot.

# %%
import numpy as np 

z = np.dot(x, y)

# %%
def naive_vector_dot(x, y): 

    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0] 
    z = 0.
    
    for i in range(x.shape[0]): z += x[i] * y[i]

    return z

# %%
import numpy as np

def naive_matrix_vector_dot(x, y):

    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0] 
    z = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]

    return z

# %%
def naive_matrix_vector_dot(x, y):
    
    z = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y) 
    
    return z

# %%
def naive_matrix_dot(x, y):
    
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1])) 
    
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

# %% [markdown]
# ### Reshaping

# %%
import numpy as np

x = np.array([[0., 1.], 
              [2., 3.],
              [4., 5.]])
print(x)

# %%
x = x.reshape((6, 1))
x

# %%
x = x.reshape((2, 3))
x

# %%
x = np.zeros((300, 20))
x = np.transpose(x)

print(x.shape)

# %%



