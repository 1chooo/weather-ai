# Description
##### Spring, 2023
#### HW01
Please print all the interactive output without resorting to print, not only the last result. To do this, you should add the following code in Jupyter cell to the beginning of the file.
```py
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
Cifar10 資料即包含 6 萬筆 32*32 低解析度之彩色圖片，

其中 5 萬筆為訓練集；1 萬筆為測試集。所有圖片被分為 10 個類別：
0：airplane 1：automobile 2：bird 3：cat 4：deer
5：dog 6：frog 7：horse 8：ship 9：truck

請在 keras 中利用 DNN 盡你所能的訓練模型，並報告最好的一次結果，本次作業禁止使用 pre-training model，課堂上未提及的方法請附加說明。
```py
import Keras
from Keras.datasets import cifar10
```