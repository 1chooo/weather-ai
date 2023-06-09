# %% [markdown]
# 
# # Section 1 統計指標實作範例
# ## 常見於迴歸問題的評估指標
# * Mean Absolute Error (MAE)
# * Mean Squared Error (MSE)
# 
# ## 常見於分類問題的指標
# * Binary Cross Entropy (CE)
# 
# ##### 後面的課程還會有更詳細的說明

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()

# %%
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()

# %%
def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))

# %% [markdown]
# ## 練習時間
# #### 請寫一個函式用來計算 Mean Square Error
# $ MSE = \frac{1}{n}\sum_{i=1}^{n}{(Y_i - \hat{Y}_i)^2} $
# 
# ### Hint: [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)

# %%


# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# # Section 2
# ### 讀取資料
# 首先，我們用 pandas 讀取最主要的資料 application_train.csv (記得到 https://www.kaggle.com/c/home-credit-default-risk/data 下載)
# 
# Note: `data/application_train.csv` 表示 `application_train.csv` 與該 `.ipynb` 的資料夾結構關係如下
# ```
# data
#     /application_train.csv
# 20201021.ipynb
# ```

# %%
import os
import numpy as np
import pandas as pd

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# 設定 data_path
dir_data = '/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/110-2/20220421 機器學習實作2 資料清理數據前處理/data'

# %% [markdown]
# #### 用 pd.read_csv 來讀取資料

# %%
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)

# %% [markdown]
# #### Note: 在 jupyter notebook 中，可以使用 `?` 來調查函數的定義

# %%
# for example
?pd.read_csv

# %% [markdown]
# #### 接下來我們可以用 .head() 這個函數來觀察前 5 row 資料

# %%
app_train.head()

# %% [markdown]
# ## 練習時間
# 資料的操作有很多，未來會介紹常被使用到的操作，大家不妨先自行想像一下，第一次看到資料，我們一般會想知道什麼訊息？
# 
# #### Ex: 如何知道資料的 row 數以及 column 數、有什麼欄位、多少欄位、如何截取部分的資料等等
# 
# 有了對資料的好奇之後，我們又怎麼通過程式碼來達成我們的目的呢？
# 
# #### 可參考該[基礎教材](https://bookdata.readthedocs.io/en/latest/base/01_pandas.html#DataFrame-%E5%85%A5%E9%97%A8)或自行 google

# %% [markdown]
# ### e.g.
# #### 資料的 row 數以及 column 數

# %%


# %% [markdown]
# #### 列出所有欄位

# %%


# %% [markdown]
# #### 截取部分資料 pd.iloc[]

# %%


# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# # Section 3
# ### 自己建立 datafreme

# %%
import pandas as pd

# %% [markdown]
# ### 方法一

# %%
data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'visitor': [139, 237, 326, 456]}

# %%
visitors_1 = pd.DataFrame(data)
visitors_1

# %% [markdown]
# ## 使用內建功能讀取 txt 檔

# %%
with open("/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/110-2/20220421 機器學習實作2 資料清理數據前處理/data/examples/example.txt", 'r') as f:
    data = f.readlines()
print(data)

# %% [markdown]
# ## 將 txt 轉成 pandas dataframe

# %%
import pandas as pd

data = []
with open("/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/110-2/20220421 機器學習實作2 資料清理數據前處理/data/examples/example.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',') # 將每句最後的 \n 取代成空值後，再以逗號斷句
        data.append(line)
data

# %%
df = pd.DataFrame(data[1:])
df.columns = data[0]
df

# %% [markdown]
# ## 將資料轉成 json 檔後輸出
# 將 json 讀回來後，是否與我們原本想要存入的方式一樣? (以 id 為 key)

# %%
import json
df.to_json('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example01.json')

# %%
# 上面的存入方式，會將 column name 做為主要的 key, row name 做為次要的 key
with open('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example01.json', 'r') as f:
    j1 = json.load(f)
j1

# %%
df.set_index('id', inplace=True)
df

# %%
df.to_json('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example02.json', orient='index')

# %%
with open('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example02.json', 'r') as f:
    j2 = json.load(f)
j2

# %% [markdown]
# ## 將檔案存為 npy 檔
# 一個專門儲存 numpy array 的檔案格式
# 使用 npy 通常可以讓你更快讀取資料喔!  
# [建議閱讀](https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161)

# %%
import numpy as np
# 將 data 的數值部分轉成 numpy array
array = np.array(data[1:])
array

# %%
np.save(arr=array, file='/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example.npy')

# %%
array_back = np.load('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example.npy')
array_back

# %% [markdown]
# ## Pickle
# 存成 pickle 檔  
# 什麼都包，什麼都不奇怪的 [Pickle](https://docs.python.org/3/library/pickle.html)  
# 比如說 [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) 的資料集就是用 pickle 包的喔!

# %%
import pickle
with open('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example.pkl', 'wb') as f:
    pickle.dump(file=f, obj=data)

# %%
with open('/content/drive/MyDrive/中大講師/中大講師/中大 - 天氣與人工智慧/111-1/20221109 機器學習實作2 資料清理數據前處理/data/examples/example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
pkl_data

# %% [markdown]
# ### 方法二

# %%
cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]

list_labels = ['city', 'weekday', 'visitor']
list_cols = [cities, weekdays, visitors]

zipped = list(zip(list_labels, list_cols))

# %%
zipped

# %%
visitors_2 = pd.DataFrame(dict(zipped))
visitors_2

# %% [markdown]
# ## 一個簡單例子
# 假設你想知道如果利用 pandas 計算上述資料中，每個 weekday 的平均 visitor 數量，
# 
# 通過 google 你找到了 https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas
# 
# 想要測試的時候就可以用 visitors_1 這個只有 4 筆資料的資料集來測試程式碼

# %%
visitors_1.groupby(by="weekday")['visitor'].mean()

# %% [markdown]
# ## 練習時間
# 在小量的資料上，我們用眼睛就可以看得出來程式碼是否有跑出我們理想中的結果
# 
# 請嘗試想像一個你需要的資料結構 (裡面的值可以是隨機的)，然後用上述的方法把它變成 pandas DataFrame
# 
# #### Ex: 生成一個 dataframe 有兩個欄位，一個是國家，一個是人口，求人口數最多的國家
# 
# ### Hints: [隨機產生數值](https://blog.csdn.net/christianashannon/article/details/78867204)
# 
# 

# %% [markdown]
# 
# <img src="./hw3-1.png">

# %%


# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ===========================================================================================

# %% [markdown]
# ## 讀取圖片
# 常見的套件:
# 1. skimage
# 2. PIL
# 3. OpenCV

# %%
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# %%
import skimage.io as skio
img1 = skio.imread('data/examples/example.jpg')
plt.imshow(img1)
plt.show()

# %%
from PIL import Image
img2 = Image.open('data/examples/example.jpg') # 這時候還是 PIL object
img2 = np.array(img2)
plt.imshow(img2)
plt.show()

# %%
import cv2
img3 = cv2.imread('data/examples/example.jpg')
plt.imshow(img3)
plt.show()

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()

# %% [markdown]
# ### 比較三種開圖方式的時間 - 比較讀取 1000 次

# %%
N_times = 1000

# %%
%%timeit
im = np.array([skio.imread('data/examples/example.jpg') for _ in range(N_times)])

# %%
%%timeit
im = np.array([np.array(Image.open('data/examples/example.jpg')) for _ in range(N_times)])

# %%
%%timeit
im = np.array([cv2.cvtColor(cv2.imread('data/examples/example.jpg'), cv2.COLOR_BGR2RGB) for _ in range(N_times)])

# %% [markdown]
# ## 將影像存成 mat

# %%
import scipy.io as sio
sio.savemat(file_name='data/examples/example.mat', mdict={'img': img1})

# %%
mat_arr = sio.loadmat('data/examples/example.mat')
print(mat_arr.keys())

# %%
mat_arr = mat_arr['img']
print(mat_arr.shape)

# %%
plt.imshow(mat_arr)
plt.show()

# %% [markdown]
# # 練習時間
# 
# ## 1-1 讀取 txt 檔
# * 請讀取 [text file](https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt)
# * 懶人複製連結: https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt
# 
# ## 1-2 將所提供的 txt 轉成 pandas dataframe
# 
# ## 2. 從所提供的 txt 中的連結讀取圖片，請讀取上面 data frame 中的前 5 張圖片

# %%
import pandas as pd
import requests
data = []
data = requests.get('https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt')
for i in data.content:
    print(i)

#    for line in f:
#       line = line.replace('\n', '').split(',') # 將每句最後的 /n 取代成空值後，再以逗號斷句
  #      data.append(line)


