# ZX_H2

## Usage

`main.py`为主函数，`SSM.py`为模型部分。

环境配置`pip install -r requirments.txt`。

使用时先新建datasets目录放入csv格式数据，运行参数全部写入configs目录当中的json文件中。
运行使用`python main.py XXX.json`，例如`python main.py configs/FI8133.json`。

## Config

`ModelLoaded`与`ModelSaved`分别表示是否读取已有模型/是否保存训练后模型，模型路径可以指定。

`Test_size`可以设置为`-1`用以表示用除训练集之外的所有数据作为测试集。

## Output

在程序执行结束之后会输出在测试集上以步长`evaluation_pred_step`进行evaluation的RMSE结果。

在目录`results`中会保存测试集上以步长参数列表`prediction_steps`进行forcast的结果。