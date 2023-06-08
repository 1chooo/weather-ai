# Usage

Python 3.7.6
```shell
pip install --upgrade pip
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install keras
pip install tensorflow
pip install torch
pip install opencv-python
pip install nltk

pip install mkdocs
pip install mkdocs-material
pip install pymdown-extensions
pip install mkdocstrings
pip install mkdocs-git-revision-date-plugin
pip install mkdocs-jupyter
```

```py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import torch
import cv2
import nltk

# 使用 sklearn 內建的資料集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 將資料集切割成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 Logistic Regression 分類器
clf = LogisticRegression()
clf.fit(X_train, y_train)

# 使用 Matplotlib 繪製資料和分類結果
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Data')
plt.show()

# 使用 Pandas 讀取資料集並顯示前幾筆資料
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# 使用 Keras 建立一個簡單的神經網路模型
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 使用 TensorFlow 設定運算配置並訓練模型
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
keras.backend.set_session(session)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# 使用 PyTorch 建立一個簡單的神經網路模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.fc2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# 設定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 定義損失函數和優化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

# 轉換訓練資料為 Tensor
inputs = torch.Tensor(X_train).to(device)
labels = torch.Tensor(y_train).long().to(device)

# 訓練模型
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("All packages are installed and working correctly!")

```