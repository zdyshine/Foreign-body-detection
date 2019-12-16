# Foreign-body-detection
### 1.代码结构

```
 {repo_root}
  ├── models	//模型文件夹
  ├── utils		//一些函数包
  |   ├── eval.py		// 求精度
  │   ├── misc.py		// 模型保存，参数初始化，优化函数选择
  │   ├── radam.py
  │   └── ...
  ├── args.py		//参数配置文件
  ├── build_net.py		//搭建模型
  ├── dataset.py		//数据批量加载文件
  ├── preprocess.py		//数据预处理文件，生成坐标标签
  ├── train.py		//训练运行文件
  ├── train_val.py	//使用训练好的模型进行验证，默认已知label
  ├── test.py	//使用训练好的模型进行验证，默认未知label
  ├── matrix.py	//绘制混淆矩阵
  ├── transform.py		//数据增强文件
```

### 2. 环境设置

可以直接通过`pip install -r requirements.txt`安装指定的函数包，python版本为3.6，具体的函数包如下：

* pytorch=1.0.1
* torchvision==0.2.2
* matplotlib==3.1.0
* numpy==1.16.4
* scikit-image
* pandas
* sklearn