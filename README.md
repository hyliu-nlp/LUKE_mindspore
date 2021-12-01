# LUKE_mindspore
 LukeForReadingcomprehension model for mindspore

### 1.代码目录结构说明
```text
  Luke_MindSpore
  ├─data                 # 存放squad_feature相关数据
  ├─dataset              # 存放squad数据集
  ├─model                # Bert model and Luke model
  ├─readingcomprehension # Luke model for reading comprehension
  ├─Eval_SquAD.ipynb     # 验证
  ├─Train_SquAD.ipynb    # 训练
  └─torch2ms.ipynb       # torch权重转换model
```
### 2.src/Squad.py
需要wiki文件：https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view?usp=sharing
get dev_data.npy
```
python src/Squad.py -eval True
```
get train_data.npy
```
python src/Squad.py -eval False
```
### 3.Eval_SquAD.ipynb验证所需要的文件
1.验证集文件：./data/dev_data.npy
2.权重文件：luke-large-qa.ckpt

### 4.Train_SquAD.ipynb训练所需要的文件
1.训练集文件：./data/train_data.npy
2.预训练权重文件：luke-large-qa.ckpt
