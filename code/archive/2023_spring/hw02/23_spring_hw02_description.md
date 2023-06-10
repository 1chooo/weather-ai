# Description
##### Spring, 2023
#### HW02

Please print all the interactive output without resorting to print, not only the last result. To do this, you should add the following code in Jupyter cell to the beginning of the file.

**1.**  
Cifar10 資料即包含 6 萬筆 32*32 低解析度之彩色圖片，其中 5 萬筆為訓練集；1 萬筆為測試集。所有圖片被分為 10 個類別：`0：airplane 1：automobile 2：bird 3：cat 4：deer 5：dog 6：frog 7：horse 8：ship 9：truck`
請在 Keras 中利用 CNN 盡你所能的訓練模型並完整報告出最好的一次結果及與第一次作業您訓練出來的 DNN 模型做比較。禁止使用 pre-trained model 課堂上未提及的方法請附加說明

**2.**  
承題 1，使用 pre-trained model `VGG16` 作為模型，並搭配 data augmentation(可fine-tuning)，將預測結果與題 1 做比較。  
使用：
```python
VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import Keras
from Keras.datasets import cifar10
```