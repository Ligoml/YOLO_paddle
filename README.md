本项目使用飞桨 PaddlePaddle 复现目标检测经典论文 YOLO，参考[《论文复现指南-cv版》](https://github.com/PaddlePaddle/models/blob/tipc/docs/lwfx/ArticleReproduction_CV.md)，按照步骤一步步完成[第四期飞桨论文复现赛](https://aistudio.baidu.com/aistudio/competition/detail/106/0/task-definition)真题：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，并制作一期飞桨论文复现入门课，你可以在 [百度AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3261974) 快速体验论文复现工作。

> 注：如果你想复现其他方向的论文，欢迎参考[《论文复现指南-nlp版》](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_NLP.md)、[《论文复现指南-Rec版》](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_REC.md)

## 论文介绍
复现代码之前，先来认识一下我们要复现的经典检测模型 YOLO，这是一个端到端的目标检测算法，与之前的多阶段目标检测算法不同，YOLO 只需要进行一次 CNN 网络计算即可得到预测结果，这使得 YOLO 模型成为了业界公认的高精度、高效率、高实用性的模型。

<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/a0b2ba74d09b46659ee2ee2107d06be9b190f87eeb96459ca8c64bd93c4fd468' width='600'/>
</div>

目前 YOLO 模型已经得到了学术界产业界的一致认可，并衍生出一系列的优质模型，其中包括 PaddleDetection 的明星模型 [PP-YOLO 和 PP-YOLOv2](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/ppyolo/README.md)，在速度和精度上甚至超过了官方团队新出的 YOLOv5。

<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/72b419d96b4a441283693354075e7a6d51d124dffd134e42bb7d4a74dc074088' width='300'/>
   <img src='https://ai-studio-static-online.cdn.bcebos.com/eb5d319ec175401783a92a8dd2b5ab27f525d8e962514a8e90c34fe1c6948a16' width='300'/>
</div>

YOLO 是如何实现"只用看一次"就得到目标检测框和物体分类结果的呢？整体来看，YOLO 算法采用一个单独的 CNN 模型实现端到端的目标检测，整个系统如下图所示：
- 首先将输入图片 resize 到 448 x 448，送入 CNN 网络；
- 之后网络将输入的图片分割成 S × S 网格，分别送到两条分支；
- 上面的检测分支每个单元格负责去检测那些中心点落在该格子内的目标，例如图中狗的中心点落在第5行第2格，则这一格负责预测狗的检测框；下面的分类分支每个单元格需要预测格子中图像的分类，并给出置信度。
- 最后结合检测分支与分类分支的输出结果，计算得出最终的检测图。

<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/450744c2723e443c9fb6f4e5cd26980965beda300a5442a5bdb130fa9dad1d63' width='600'/>
</div>


## 开始复现
面对一篇计算机视觉论文，复现该论文的整体流程如下图所示，总共包含11个步骤。为了高效复现论文，设置了5个打卡点，如图中黄色框所示。

<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/6198b7f186454bdd82a39bb900ee544950e32ca7c13c4beba275a7f39048adea' width='800'/>
</div>

接下来，我们将参考上图复现流程，逐步对齐并通过打卡点，复现 YOLO 模型。各步骤代码详见： 
```
YOLO_paddle
├─┬─ checkpoint_1
  ├───yolo_paddle
  ├───yolo_torch
  └───checkpoint_1.py
├─── checkpoint_2
├─── checkpoint_3
├─── checkpoint_4
└─── checkpoint_5
```

### 环境配置

- paddlepaddle>=2.2.0
- pytorch>=1.7.1
- [reprod_log](https://github.com/WenmuZhou/reprod_log)

> 注：你可以在 [百度AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/3261974) 中快速启动，并按照文档提示完成各打卡点代码练习。

### 打卡点1：前向对齐（含Step1：模型结构对齐）
对齐模型结构时，一般有3个主要步骤：
- 网络结构代码转换
- 权重转换
- 模型组网正确性验证

#### 1.1 网络结构代码转换

根据飞桨 API，使用 `paddle.nn` 与 `paddle.*` 下相关API 实现模型。

- `paddle.nn.*` 下包含108个模型组网类 API，覆盖 Conv、Pooling、Norm、RNN 等功能。
- `paddle.*` 下包含225个 Tensor 操作类 API，覆盖数学计算、逻辑运算、查找、初始化等功能。

> **注意事项**
>
> - 多数 API 在功能上与 torch 一致，名称上会有些许区别，详细信息可查询：[PyTorch-PaddlePaddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/08_api_mapping/pytorch_api_mapping_cn.html) ；
> - 有些 torch API 在 paddle 并未对应，但可以组合实现，详细信息可查询：[官网 API 文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/index_cn.html)、[官网 FAQ](https://www.paddlepaddle.org.cn/documentation/docs/zh/faq/index_cn.html)等，或者提 [github issue](https://github.com/PaddlePaddle/Paddle/issues) 寻求解答。

#### 1.2 权重转换
- 输入一致，权重一致，输出一致 ─> 模型一致
- 输入一致，权重一致，输出不一致 ─> 模型不一致

我们可以通过随机生成假数据获得一致的输入，通过权重转换获得一致的权重，控制变量判断模型结构是否一致。

随机生成假数据的代码放在 `./YOLO_paddle/fake_data/gen_fake_data.py`，
权重转换的代码放在 `./YOLO_paddle/weights_trans/torch2paddle.py`。

> **注意事项**
> 
> 由于设计不同， PaddlePaddle 和其他框架 API 的权重保存格式和名称会有不同，需要特殊处理：
> - 对于 paddle.nn.Linear 层的 weight 参数，PaddlePaddle 与 PyTorch 的保存方式不同，在转换时需要进行转置；
> - paddle.nn.BatchNorm2D 和 torch.nn.BatchNorm2d 的参数名对应关系如下：
> ```
> weight -> weight
> bias -> bias
> _variance -> running_var
> _mean -> running_mean
> ```
> 模型转换的注意事项详见[《论文复现指南-cv 版》](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_CV.md#312-%E6%9D%83%E9%87%8D%E8%BD%AC%E6%8D%A2)。

#### 1.3 模型组网正确性验证

```shell
# 打卡点1对齐（对齐网络结构）
cd ./checkpoint_1/
# 生成paddle的前向数据
cd yolo_paddle/ && python forward_yolo.py

# 生成官方实现模型的前向数据
cd ../yolo_torch && python forward_yolo.py

# 对比生成log
cd ..
python checkpoint_1.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `forward_diff.log` 文件中。
```
[2021/12/18 14:38:00] root INFO: logits: 
[2021/12/18 14:38:00] root INFO: 	mean diff: check passed: True, value: 1.155436994173588e-07
[2021/12/18 14:38:00] root INFO: diff check passed
```

### 打卡点2：评估对齐（含Step 2：验证/测试集数据读取对齐、Step 3：评估指标对齐）

对齐评估指标，你需要完成**Step 2：验证/测试集数据读取对齐** 和 **Step 3：评估指标对齐**，分别会打印两个对齐 log。

首先是 **Step 2：验证/测试集数据读取对齐**，在这一步，我们需要
* 下载数据集 VOCdevkit 并解压；
* 参考官方实现代码转写 paddle 数据加载代码；
* 固定预处理方式，在 `~/YOLO_paddle/checkpoint_2/test_vocdata.py` 中定义 `dataset` 与 `dataloader` ，对比输出误差，打印 log。

接下来是 **Step 3：评估指标对齐**，在这一步，我们需要：
* 将测试集数据载入网络，定义模型的 `eval` 部分代码；
* 打印评估结果并将评估指标 `mAP` 保存；
* 评估指标正确性验证。

> **注意事项**
> 
> - 下载数据集注意名称与论文保持一致，并建议从官方渠道下载，若从其他渠道下载数据集，需要先验证数据集是否完整、是否做过其他处理；
> - 数据集使用方式不同，有些论文中，可能只是抽取了该数据集的子集进行方法验证，此时需要注意抽取方法，需要保证抽取出的子集完全相同。

#### 2.1 验证/测试集数据读取正确性验证

随机选取5组数据，比对Dataset 和 DataLoader 的长度和数据内容 diff：

```shell
# dataset与dataloader输出对齐
cd ../checkpoint_2/
python test_vocdata.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `data_diff.log` 文件中。
```
[2021/12/18 21:34:30] root INFO: length: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataset_0: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataset_1: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataset_2: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataset_3: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataset_4: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataloader_0: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataloader_1: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataloader_2: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataloader_3: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: dataloader_4: 
[2021/12/18 21:34:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 21:34:30] root INFO: diff check passed
```

#### 2.2 评估指标正确性验证

分别定义官方实现与 PaddlePaddle 模型，分别加载训练好的权重，使用测试集获取评估结果，使用 `reprod_log` 保存结果并比对 diff：
```shell
# 打卡点2对齐（对齐评估指标）
# 评估paddle模型，生成评估指标map
cd yolo_paddle/
sh run_eval.sh

# 评估官方实现模型，生成评估指标map
cd ../yolo_torch/
sh run_eval.sh

# 对比生成log，阈值改为1e-5
cd ..
python checkpoint_2.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `metric_diff.log` 文件中。
```
[2021/12/18 15:04:42] root INFO: map: 
[2021/12/18 15:04:42] root INFO: 	mean diff: check passed: True, value: 4.860971909992351e-06
[2021/12/18 15:04:42] root INFO: diff check passed
```

### 打卡点3：损失函数对齐（含 Step 4:损失函数对齐）
对齐损失函数，你需要参考以下步骤：
* 定义模型训练代码，加载权重与假数据假标签，获取一次前向计算的 loss 结果并保存；
* 损失函数正确性验证。

> **注意事项**
> - `CrossEntropyLoss`有一定的区别需要注意：PaddlePaddle 提供了对软标签、指定 softmax 计算纬度的支持。即 `paddle.nn.CrossEntropyLoss` 默认是在最后一维(axis=-1)计算损失函数，而 `torch.nn.CrossEntropyLoss` 是在axis=1的地方计算损失函数，因此如果输入的维度大于 2，这里需要保证计算的维(axis)相同，否则可能会出错。
> - 更多注意事项详见[《论文复现指南-cv 版》](https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_CV.md#45-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E5%AF%B9%E9%BD%90)。

```shell
# 打卡点3对齐（对齐损失函数）
# 使用假数据与假标签获取paddle模型的损失函数
cd ../checkpoint_3/yolo_paddle/
python loss_train.py

# 使用假数据与假标签获取官方实现模型的损失函数
cd ../yolo_torch/
python loss_train.py

# 对比生成log，阈值改为1e-5
%cd ..
!python checkpoint_3.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `loss_diff.log` 文件中。
```
[2021/12/18 15:29:03] root INFO: conf_loss: 
[2021/12/18 15:29:03] root INFO: 	mean diff: check passed: True, value: 5.982089037459559e-07
[2021/12/18 15:29:03] root INFO: cls_loss: 
[2021/12/18 15:29:03] root INFO: 	mean diff: check passed: True, value: 7.152557373046875e-07
[2021/12/18 15:29:03] root INFO: txtytwth_loss: 
[2021/12/18 15:29:03] root INFO: 	mean diff: check passed: True, value: 4.291534423828125e-06
[2021/12/18 15:29:03] root INFO: total_loss: 
[2021/12/18 15:29:03] root INFO: 	mean diff: check passed: True, value: 3.2164883627672225e-06
[2021/12/18 15:29:03] root INFO: diff check passed
```

### 打卡点4：反向对齐（含 Step 5：优化器对齐、Step 6：学习率对齐、Step 7：正则化策略对齐、Step 8：反向对齐）

对齐反向计算，你需要完成 **Step 5+6：优化器学习率对齐** 和 **Step 8：反向对齐**，分别会打印两个对齐 log。

首先是 **Step 5+6：优化器学习率对齐**，在这一步，我们需要
* 定义优化器
* 定义学习率规则
* 设置 90 个 epoch，将每个 epoch 的学习率保存
* 比对学习率正确性

> **注意事项**
> 
> - 由于 PaddlePaddle 同时支持动静态图，所以有一些独特的用法，以SGD等优化器为例，PaddlePaddle 在优化器中增加了对梯度裁剪的支持，常用于 GAN 或一些 NLP、多模态场景中。
> - PaddlePaddle 的 SGD 不支持动量更新、动量衰减和 Nesterov 动量，需要用 `paddle.optimizer.Momentum` 实现。

接下来是 **Step 8：反向对齐**，此处推荐使用假数据与假标签，实现可复现的结果。在这一步，我们需要
* 检查两个代码的训练超参数全部一致，如优化器及其超参数、学习率、BatchNorm/LayerNorm中的eps等；
* 将 PaddlePaddle 与 PyTorch 网络中涉及的所有随机操作全部关闭，如 `dropout`、`drop_path` 等，推荐将模型设置为 `eval` 模式（ `model.eval()` ）；
* 加载相同的模型权重，将准备好的数据分别传入网络并迭代，观察二者loss是否一致（此处 `batch_size` 要一致，如果使用多个真实数据，要保证传入网络的顺序一致）；
* 如果经过2轮以上，loss 均可以对齐，则基本可以认为反向对齐。

#### 4.1 优化器学习率正确性验证
```shell
# 优化器与学习率对齐
cd ../checkpoint_4/yolo_paddle/
python lr_optim_eval.py
cd ../yolo_torch/
python optim_lr_eval.py
cd ..
python test_lr.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `lr_diff.log` 文件中。
```
[2021/12/18 15:29:26] root INFO: lr: 
[2021/12/18 15:29:26] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/12/18 15:29:26] root INFO: diff check passed
```
#### 4.2 反向对齐正确性验证
```shell
# 打卡点4对齐（对齐反向传播）
# 使用假数据与假标签获取paddle模型的5个epoch反向损失函数
cd yolo_paddle/
python bp_train.py

# 使用假数据与假标签获取官方实现模型的5个epoch反向损失函数
cd ../yolo_torch/
python bp_train.py

# 对比生成log，阈值改为1e-5
cd ..
python checkpoint_4.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `bp_align_diff.log` 文件中。
```
[2021/12/18 22:09:28] root INFO: total_loss_0: 
[2021/12/18 22:09:28] root INFO: 	mean diff: check passed: True, value: 6.395273999260098e-08
[2021/12/18 22:09:28] root INFO: total_loss_1: 
[2021/12/18 22:09:28] root INFO: 	mean diff: check passed: True, value: 1.903436405825687e-06
[2021/12/18 22:09:28] root INFO: total_loss_2: 
[2021/12/18 22:09:28] root INFO: 	mean diff: check passed: True, value: 7.995373726643606e-06
[2021/12/18 22:09:28] root INFO: total_loss_3: 
[2021/12/18 22:09:28] root INFO: 	mean diff: check passed: False, value: 4.382225737131762e-05
[2021/12/18 22:09:28] root INFO: total_loss_4: 
[2021/12/18 22:09:28] root INFO: 	mean diff: check passed: False, value: 7.397504902684204e-05
[2021/12/18 22:09:28] root INFO: diff check failed
```
> 注：反向对齐会受到许多因素的影响，不一定可以严格对齐，通常我们认为连续两个迭代轮次符合阈值即可。

### 打卡点5：精度对齐（含 Step9：训练集数据读取对齐、Step10：网络初始化对齐、Step11：训练对齐）

终于到了最激动人心的训练环节，~~让我们一把梭哈，不行回家~~~经过前4步的对齐，通常来讲训练精度不会有特别大的误差，我们定义好训练代码 `train.py` 后，就可以开启漫长的深度学习训练过程了。

训练后精度直接通过观察即可判断是否对齐，若你想量化结果，可继续使用 `reprod_log` 打卡。
```shell
# 开始训练，并将最佳模型参数保存
cd ../checkpoint_5/yolo_paddle/
sh train.sh

# 测试训练最佳模型在测试集上的指标并保存
sh eval.sh

# 对比精度，打印log，阈值调整为0.0015
cd ..
python checkpoint_5.py
```

产出日志如下，同时会将 `reprod_log` 的结果保存在 `train_align_diff.log` 文件中。
```
[2021/12/18 22:22:17] root INFO: map: 
[2021/12/18 22:22:17] root INFO: 	mean diff: check passed: True, value: 0.0002689957618713379
[2021/12/18 22:22:17] root INFO: diff check passed
```

### 模型预测效果展示
<div align='left'>
  <img src='https://github.com/Ligoml/YOLO_paddle/blob/main/output_img/000001.jpg' width='300'/>
  <img src='https://github.com/Ligoml/YOLO_paddle/blob/main/output_img/000001_predict.jpg' width='300'/>
</div>


至此，你完整的完成了使用 PaddlePaddle 进行目标检测经典论文 YOLO 复现的工作，你可以把 `./YOLO_paddle/checkpoint_5/yolo_paddle` 下的代码整理后上传 GitHub，开源你的论文复现工作给更多的开发者，代码结构和文档可以参考[《REPO 提交规范》](https://github.com/PaddlePaddle/models/blob/release/2.2/community/REPO_TEMPLATE_DESC.md)。

## 参考项目
- https://github.com/sunlizhuang/YOLOv1-PaddlePaddle
- https://github.com/yjh0410/new-YOLOv1_PyTorch
- https://github.com/AlexeyAB/darknet
