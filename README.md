# ZX_H2

## Usage

`main.py`为主函数，`SSM.py`为模型部分。

环境配置`pip install -r requirments.txt`。

使用时先新建datasets目录放入csv格式数据，运行参数全部写入configs目录当中的json文件中。
运行使用`python main.py XXX.json`，例如`python main.py configs/FI8133.json`。

## Config args

`ModelLoaded`与`ModelSaved`分别表示是否读取已有模型/是否保存训练后模型，模型路径统一使用`ModelPath`参数。