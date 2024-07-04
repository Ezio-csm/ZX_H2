# ZX_H2

## Usage

环境配置`pip install -r requirments.txt`。

`main.py`为主函数，`SSM.py`为模型部分。

**非必要请勿修改代码，仅通过修改`json`配置文件来改变运行参数**

使用时先新建datasets目录放入csv格式数据，运行参数全部写入configs目录当中的json文件中。
运行使用`python main.py XXX.json`，例如`python main.py configs/FI8133.json`。

## Config

| Config               | Description   |
| -----------          | -----------   |
| Data_path            | 数据路径       |
| CV_name              | 输出变量       |
| MV_name              | 输入变量       |
| Lags                 | 变量延迟       |
| Training_size        | 训练用数据条目 |
| Test_size            | 测试用数据条目 |
| evaluation_pred_step | 测试评估用步长 |
| prediction_steps     | 预测用步长列表 |
| EM_max_iter          | EM迭代次数     |
| blkMin               | 黑盒搜索Lag下界 |
| blkMax               | 黑盒搜索Lag上界 |
| ModelLoaded          | 是否读取模型    |
| ModelSaved           | 是否保存模型    |
| ModelLoadedPath      | 模型读取路径    |
| ModelSaveedPath      | 模型保存路径    |

`Test_size`可以设置为`-1`用以表示用除训练集之外的所有数据作为测试集。

## Output

在程序执行结束之后会输出在测试集上以步长`evaluation_pred_step`进行evaluation的RMSE结果。

在目录`results`中会保存测试集上以步长参数列表`prediction_steps`进行forcast的结果。

## Lag Find

调用`python blk_opt.py XXX.json`可以以`json`配置中的Lag为基础搜索最优Lag配置，结果会保存在`/logs`目录下。