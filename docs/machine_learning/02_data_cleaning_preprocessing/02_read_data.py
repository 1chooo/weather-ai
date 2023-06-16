# %% [markdown]
# # Section 2 - 讀取資料
# 首先，我們用 pandas 讀取最主要的資料 application_train.csv (記得到 [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) 下載)
# 
# Note: `data/application_train.csv` 表示 `application_train.csv` 與該 `.ipynb` 的資料夾結構關係如下
# ```
# PROJECT_ROOT
# ├── data/
# │   └── application_train.csv
# └── 02_read_data.ipynb
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
# #### 接下來我們可以用 `.head()` 這個函數來觀察前 5 row 資料

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
# 
# ### e.g.
# - 資料的 row 數以及 column 數
# - 列出所有欄位
# - 截取部分資料 `pd.iloc[]`