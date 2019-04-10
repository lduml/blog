[TOC]

# 文本可视化

<https://github.com/lduml/TextCharactervVisualization>

<https://www.jianshu.com/p/08e00132eb48>

<http://web.jobbole.com/91775/>

<https://www.cnblogs.com/Sinte-Beuve/p/7617517.html>

<https://towardsdatascience.com/tagged/nlp>

<https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a>

---

# 使用**BERT**

> 从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史

https://zhuanlan.zhihu.com/p/49271699

- 使用BERT进行多标签分类; 来AI挑战者的细粒度情感分析

<https://www.ctolib.com/brightmart-sentiment_analysis_fine_grain.html>

- 小数据福音！BERT在极小数据下带来显著提升的开源实现

<https://mp.weixin.qq.com/s/XmeDjHSFI0UsQmKeOgwnyA>

<https://github.com/Socialbird-AILab/BERT-Classification-Tutorial>

- NER

<https://github.com/lduml/BERT-NER.git>

机器阅读理解数据集

<https://zhuanlan.zhihu.com/p/52894380>

QAnet

<https://github.com/NLPLearn/QANet.git>

<https://github.com/allenai/bi-att-flow>

<https://github.com/aswalin/SQuAD>

-----

https://github.com/allenai/bi-att-flow

---

# 爬虫

<https://github.com/CriseLYJ/awesome-python-login-model>

<https://github.com/CriseLYJ/awesome-python-login-model>

---

# 下载

1、比赛数据集

<http://www.digquant.com.cn/tipdm/intro#datazip>

2、bert模型

3、百度模型

4、可视化

<https://github.com/jessevig/bertviz.git>

5、[bert-as-service](https://github.com/hanxiao/bert-as-service)

<https://github.com/hanxiao/bert-as-service>

6\

<https://github.com/NLPScott/bert-Chinese-classification-task.git>

7\

<https://github.com/jadore801120/attention-is-all-you-need-pytorch>

8\

<https://github.com/NLPScott/pytorch-pretrained-BERT>

9\

<https://github.com/lduml/NLP-BERT--ChineseVersion>

10\

<https://github.com/lduml/BERT-BiLSTM-CRF-NER>

11\使用BERT进行多标签分类; AI挑战者的细粒度情感分析

<https://github.com/brightmart/sentiment_analysis_fine_grain>

---

# Docker

写在前面

docker常用命令：

docker ps 

docker ps -a

dokcer import  镜像文件名

docker images

docker rm 容器

docker rmi 镜像

docker stop 容器

docker build -t 镜像名 dockerfile的位置

docker run -d -p 5000:5000 镜像名

docker exec -it 容器名 bash

docker start 容器

docker restart 容器

从docker容器中向主机拷贝文件 docker cp containerID:container_path host_path

从主机复制到容器docker cp host_path containerID:container_path

在Docker中启动Cloudera

1、用docker import ***.tar.gz 加载镜像

2、用docker run 运行容器:

docker run -m 10G --memory-swap 10G --name cdh --hostname=quickstart.cloudera --privileged=true -t -i -p 8020:8020 -p 8888:8888 -p 8022:8022 -p 7180:7180 -p 21050:21050 -p 50070:50070 -p 50075:50075 -p 50010:50010 -p 50020:50020 -p 8890:8890 -p 60010:60010 -p 10002:10002 -p 25010:25010 -p 25020:25020 -p 18088:18088 -p 8088:8088 -p 19888:19888 -p 7187:7187 -p 11000:11000 -p 80:80 -p 8080:8080 cloudera/quickstart /bin/bash -c '/usr/bin/docker-quickstart && /home/cloudera/cloudera-manager --force --express && service ntpd start

docker导入镜像

docker import  my_ubuntu_v3.tar runoob/ubuntu:v4  

 my_ubuntu_v3.tar 镜像文件

runoob/ubuntu:v4  镜像新的名字

---

# **seq2seq**

https://github.com/tensorflow/nmt>

https://github.com/google/seq2seq>

---

# 安装软件 mongodb

**mongodb**

https://www.jianshu.com/p/7241f7c83f4a>

启动

mongo

关闭

use admin;

db.shutdownServer();

**mysql**

https://www.jianshu.com/p/e5c9e8ef8ccb>

把**mysql**安装目录，比如**MYSQLPATH/bin/mysql**，映射到**/usr/local/bin**目录下：

cd /usr/local/bin

ln -fs /usr/local/mysql-8.0.11-macos10.13-x86_64/bin/mysql mysql

修改密码

在**MySQL8.0.4**以前，执行

SET PASSWORD=PASSWORD('修改的密码'); 

---

# 情感分析

实践**Twitter**评论情感分析（数据集及代码）

<https://www.jianshu.com/p/8a2ad5e6a363>

<https://github.com/prateekjoshi565/twitter_sentiment_analysis.git>

【**Natural Language Processing**】跨语言情感分析(**NLP&CC 2013**)

[**https://blog.csdn.net/law_130625/article/details/71081497**](https://blog.csdn.net/law_130625/article/details/71081497)

【**Python**学习】**python**爬虫**Google**翻译的实现

[**https://blog.csdn.net/law_130625/article/details/70036916**](https://blog.csdn.net/law_130625/article/details/70036916)

---

# Ubuntu

安装tomcat

<https://www.linuxidc.com/Linux/2017-10/147773.htm>

**linux**服务器查看公网**IP**信息的方法

curl [ifconfig.me](http://ifconfig.me)

curl cip.cc

#github

<https://github.com/iliaschalkidis/ELMo-keras>

echo "# google" >> README.md

git init

git add README.md

git commit -m "first commit"

git remote add origin https://github.com/lduml/google.git

git push -u origin master

!git config --global user.email "zcrrcz001@live.com"

!git config --global user.name "lduml"

844247013@qq.com

---

# Pandas

1、修改表格名字

\# ①暴力

df.columns = ['a', 'b', 'c', 'd', 'e']

\#②修改

df.columns = df.columns.str.strip('$')

\#③修改

df.columns = df.columns.map(lambda x:x[1:])

\# ④暴力（好处：也可只修改特定的列）

df.rename(columns=('$a': 'a', '$b': 'b', '$c': 'c', '$d': 'd', '$e': 'e'}, inplace=True) 

\# ⑤修改

df.rename(columns=lambda x:x.replace('$',''), inplace=True)

2、添加列

\# 新列名字，值

df[**'label'**] = **0**

**3**、拼接

拼接两个表

all = pd.concat([neg,neutral])

4、去重

对指定列去重

all = all.drop_duplicates([**'content'**])

**5**、打乱顺序

df = all.sample(frac=**1.0**)

---

# ELMo

EMBEDDINGS FROM LANGUAGE MODELS

官方主页 以及如何使用 github

<https://allennlp.org/elmo>

<https://github.com/allenai/bilm-tf>

<https://zhuanlan.zhihu.com/p/38254332>

- pip install allennlp 

即可享用。

## BERT

> 1、从**Word Embedding**到**Bert**模型**—**自然语言处理中的预训练技术发展史

[**https://zhuanlan.zhihu.com/p/49271699**](https://zhuanlan.zhihu.com/p/49271699)



[**https://www.zhihu.com/people/zhang-jun-lin-76/activities**](https://www.zhihu.com/people/zhang-jun-lin-76/activities)

BERT相关论文、文章和代码资源汇总

[**http://www.52nlp.cn/tag/bert**解读](http://www.52nlp.cn/tag/bert%E8%A7%A3%E8%AF%BB)

【中文版 | 论文原文】BERT：语言理解的深度双向变换器预训练

<https://www.cnblogs.com/guoyaohua/p/bert.html>

## **pytorch**版

如何使用**BERT**实现中文的文本分类（附代码）

[**https://blog.csdn.net/Real_Brilliant/article/details/84880528#_1**](https://blog.csdn.net/Real_Brilliant/article/details/84880528#_1)

[**https://github.com/huggingface/pytorch-pretrained-BERT**](https://github.com/huggingface/pytorch-pretrained-BERT)

[**https://github.com/codertimo/BERT-pytorch**](https://github.com/codertimo/BERT-pytorch)

## **keras_bert**

[**https://github.com/Separius/BERT-keras**](https://github.com/Separius/BERT-keras)

[**https://github.com/CyberZHG/keras-bert**](https://github.com/CyberZHG/keras-bert)

## **Transformer —** 重点关注

可视化理解

<https://jalammar.github.io/illustrated-transformer/>

深度学习中的注意力模型（**2017**版）

<https://zhuanlan.zhihu.com/p/37601161>

代码原理双管齐下———待看

<http://nlp.seas.harvard.edu/2018/04/03/attention.html>

---

# **Nlp**

[**1.** 语言模型](https://www.cnblogs.com/huangyc/p/9861453.html)

[**2. Attention Is All You Need**（**Transformer**）算法原理解析](https://www.cnblogs.com/huangyc/p/9813907.html)

[**3. ELMo**算法原理解析](https://www.cnblogs.com/huangyc/p/9860430.html)

[**4. OpenAI GPT**算法原理解析](https://www.cnblogs.com/huangyc/p/9860181.html)

[**5. BERT**算法原理解析](https://www.cnblogs.com/huangyc/p/9898852.html)

[**6.** 从**Encoder-Decoder**(**Seq2Seq**)理解**Attention**的本质](https://www.cnblogs.com/huangyc/p/10409626.html)

<https://www.cnblogs.com/huangyc/p/9861453.html>

**Google  research bert**

<https://github.com/google-research/bert>

机器学习小知识

<https://zhuanlan.zhihu.com/yangyangfuture>

**ELMo**代码详解(一)：数据准备

[**https://blog.csdn.net/jeryjeryjery/article/details/80839291**](https://blog.csdn.net/jeryjeryjery/article/details/80839291)

**ELMo**代码详解(二)

[**https://blog.csdn.net/jeryjeryjery/article/details/81183433**](https://blog.csdn.net/jeryjeryjery/article/details/81183433)

中文自然语言处理相关资料集合指南

<https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247508547&idx=2&sn=b489dce82fdba2f2252adeacfbcc5e2a&chksm=fc864350cbf1ca461bf1f6d1d45f5440c1bd5827df79e4e39bb8632b2db0a4b6256294969fa5&mpshare=1&scene=23&srcid=03102gZJAFGyOvjHACsQDoIi%23rd>

**BERT**的理解

<https://blog.csdn.net/yangfengling1023/article/details/84025313>

NLP的四大类任务

（1）序列标注：分词、实体识别、语义标注……

（2）分类任务：文本分类、情感计算……

（3）句子关系判断：entailment、QA、自然语言推理

（4）生成式任务：机器翻译、文本摘

GLUE语料集的介绍

实验数据以及对应的NLP任务	

- MNLI：蕴含关系推断	
- QQP：问题对是否等价	
- QNLI：句子是都回答问句	
- SST-2：情感分析	
- CoLA：句子语言性判断	
- STS-B：语义相似	
- MRPC：句子对是都语义等价	
- RTE：蕴含关系推断	
- WNLI：蕴含关系推断	
- 

[Allen NLP系列文章之六：Textual Entailment（自然语言推理 - 文本蕴含...](https://www.baidu.com/link?url=sW5hVWAi572JUOPkbTRwwB7UP0GZF0WquR0jsSqI6Ka2lWiiAi5q1fzsa_S8nsM36GCgHTCt4NNd-JigTN1j_kxJQ-TKLzMn9taicLmKtYO&wd=&eqid=ea72ea0600087519000000035c8500e8)

小白都能看懂的**softmax**详解

<https://blog.csdn.net/bitcarmanlee/article/details/82320853>

逻辑斯特回归

<https://blog.csdn.net/sinat_29957455/article/details/78944939>

<https://blog.csdn.net/gwplovekimi/article/details/80288964>

# 1、NLP领域最优秀的8个预训练模型（附开源地址）

```
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652041848&idx=5&sn=42385ea2739f367ee1aee0557d905c87&chksm=f1218689c6560f9fbcf51a7004fde0aa384b765a75f58dd8d88776f35994601edd0211e07496&mpshare=1&scene=1&srcid=0407uTJIt1oP2IqqkdFLO2jl&key=dd903828d49d31534448c1b4095215f816786972552f436778e8e60035d133a6a58ba1a52fda4cba8c6ac864c186955858d66dddeb75cdba167b299a6b527d41f68f10f2c93d89d5cfc4392a14a5725a&ascene=1&uin=Mjc0OTcyNzA0MA%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=DjU6TLQfmA8BFnM3vWQUF9m1bWeutc7JHL5KJ%2FIyoOtA557uM0C1onB%2FfsxFYNIT
```

## 1、多用途自然语言处理模型

- - ULMFiT

  - Transformer

  - Google BERT

  - Transformer-XL

  - OpenAI GPT-2

## 2、词嵌入

- - ELMo
  - Flair
## 3、其他预训练模型

- - StanfordNLP



# 2、NLP不同任务Tensorflow深度学习模型大全

```
https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247508896&idx=3&sn=2a9886a03d47b55da4cd8ff1d72204d5&chksm=fc8642b3cbf1cba59434a0128d5948a8385c4659e9ff49a6130f38a388203bc3209f61f86cf5&mpshare=1&scene=1&srcid=0407mXrlqSO8qdWpGPaEmded&key=dd903828d49d31532d87291db1d4f070be35f449aa457236160a42102ebafcdb85716e0b41d11a9d3c68bc3448a247939b2feab49738a4872e880bf783dbadb8a00db780e86a9d79359d66f21c0c8045&ascene=1&uin=Mjc0OTcyNzA0MA%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=DjU6TLQfmA8BFnM3vWQUF9m1bWeutc7JHL5KJ%2FIyoOtA557uM0C1onB%2FfsxFYNIT
```

# 3、10大任务超越BERT，微软提出多任务深度神经网络MT-DNN

```
https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652041265&idx=3&sn=508f7b70139f662e8bcef6e4a3b10e97&chksm=f12184c0c6560dd630d52a48e4c9ebbb7492af8e0d918cfb01e006cd0a5a48bb11d9b62a064a&mpshare=1&scene=1&srcid=0407D1fCK81v3DSiHBoPqo18&key=bc5b2a816e73d108fa5f98fa0de23e5bdf5ea46c5841b7783da1dd9eddeba980d9369da37ed440fd9feb597687ab9b5cc99b04df2af7d9d81072a048d9e965f2fdf99a74a3493cb2d874281f564a22e9&ascene=1&uin=Mjc0OTcyNzA0MA%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=DjU6TLQfmA8BFnM3vWQUF9m1bWeutc7JHL5KJ%2FIyoOtA557uM0C1onB%2FfsxFYNIT
```

# 4、用免费TPU训练Keras模型，速度还能提高20倍！

```
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650759875&idx=3&sn=4da9426891c15ce0ac2a8ae294f439bb&chksm=871aa6bdb06d2fabc779dd9fb4ba2fdb49570e54d82e2264068181acb0fd797c7a80f11fbe82&mpshare=1&scene=1&srcid=04072ULmSrwkyV7kweGHo6dc&key=bc5b2a816e73d108017ea1d2d51acaad21c26967438283f37ab0bda0fab68b132e04a6cf287a60f1ee1a4f6c306be2d2291e85405366377e92d5b19a8cc04397d6540db49a97e6911c11479587d302c6&ascene=1&uin=Mjc0OTcyNzA0MA%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=DjU6TLQfmA8BFnM3vWQUF9m1bWeutc7JHL5KJ%2FIyoOtA557uM0C1onB%2FfsxFYNIT
```



**静态输入 Batch Size**



在 CPU 和 GPU 上运行的输入管道大多没有静态形状的要求，而在 XLA/TPU 环境中，则对静态形状和 batch size 有要求。



Could TPU 包含 8 个可作为独立处理单元运行的 TPU 核心。只有八个核心全部工作，TPU 才算被充分利用。为通过向量化充分提高训练速度，我们可以选择比在单个 GPU 上训练相同模型时更大的 batch size。最开始最好设定总 batch size 为 1024（每个核心 128 个）。



如果你要训练的 batch size 过大，可以慢慢减小 batch size，直到它适合 TPU 内存，只需确保总的 batch size 为 64 的倍数即可（每个核心的 batch size 大小应为 8 的倍数）。



使用较大的 batch size 进行训练也同样有价值：通常可以稳定地提高优化器的学习率，以实现更快的收敛。（参考论文：https://arxiv.org/pdf/1706.02677.pdf）



在 Keras 中，要定义静态 batch size，我们需使用其函数式 API，然后为 Input 层指定 batch_size 参数。请注意，模型在一个带有 batch_size 参数的函数中构建，这样方便我们再回来为 CPU 或 GPU 上的推理运行创建另一个模型，该模型采用可变的输入 batch size。



```
import tensorflow as tf
from tensorflow.python.keras.layers import Input, LSTM, Bidirectional, Dense, Embedding


def make_model(batch_size=None):
    source = Input(shape=(maxlen,), batch_size=batch_size,
                   dtype=tf.int32, name='Input')
    embedding = Embedding(input_dim=max_features,
                          output_dim=128, name='Embedding')(source)
    lstm = LSTM(32, name='LSTM')(embedding)
    predicted_var = Dense(1, activation='sigmoid', name='Output')(lstm)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['acc'])
    return model


training_model = make_model(batch_size=128)
```



此外，使用 tf.train.Optimizer，而不是标准的 Keras 优化器，因为 Keras 优化器对 TPU 而言还处于试验阶段。



**将 Keras 模型转换为 TPU 模型**



tf.contrib.tpu.keras_to_tpu_model 函数将 tf.keras 模型转换为同等的 TPU 模型。



```
import os
import tensorflow as tf
# This address identifies the TPU we'll use when configuring TensorFlow.
TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
tf.logging.set_verbosity(tf.logging.INFO)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    training_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
```



然后使用标准的 Keras 方法来训练、保存权重并评估模型。请注意，batch_size 设置为模型输入 batch_size 的八倍，这是为了使输入样本在 8 个 TPU 核心上均匀分布并运行。



```
history = tpu_model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128 * 8,
                        validation_split=0.2)
tpu_model.save_weights('./tpu_model.h5', overwrite=True)
tpu_model.evaluate(x_test, y_test, batch_size=128 * 8)
```



我设置了一个实验，比较在 Windows PC 上使用单个 GTX1070 和在 Colab 上运行 TPU 的训练速度，结果如下。



GPU 和 TPU 都将输入 [batch size](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650759875&idx=3&sn=4da9426891c15ce0ac2a8ae294f439bb&chksm=871aa6bdb06d2fabc779dd9fb4ba2fdb49570e54d82e2264068181acb0fd797c7a80f11fbe82&mpshare=1&scene=1&srcid=04072ULmSrwkyV7kweGHo6dc&key=bc5b2a816e73d108017ea1d2d51acaad21c26967438283f37ab0bda0fab68b132e04a6cf287a60f1ee1a4f6c306be2d2291e85405366377e92d5b19a8cc04397d6540db49a97e6911c11479587d302c6&ascene=1&uin=Mjc0OTcyNzA0MA%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=DjU6TLQfmA8BFnM3vWQUF9m1bWeutc7JHL5KJ%2FIyoOtA557uM0C1onB%2FfsxFYNIT) 设为 128，



- GPU：每个 epoch 需要 179 秒。20 个 epoch 后验证准确率达到 76.9％，总计 3600 秒。
- TPU：每个 epoch 需要 5 秒，第一个 epoch 除外（需 49 秒）。20 个 epoch 后验证准确率达到 95.2％，总计 150 秒。



20 个 epoch 后，TPU 上训练模型的验证准确率高于 GPU，这可能是由于在 GPU 上一次训练 8 个 batch，每个 batch 都有 128 个样本。



**在 CPU 上执行推理**



一旦我们获得模型权重，就可以像往常一样加载它，并在 CPU 或 GPU 等其他设备上执行预测。我们还希望推理模型接受灵活的输入 batch size，这可以使用之前的 make_model() 函数来实现。



```
inferencing_model = make_model(batch_size=None)
inferencing_model.load_weights('./tpu_model.h5')
inferencing_model.summary()
```



可以看到推理模型现在采用了可变的输入样本。



```
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
Input (InputLayer) (None, 500) 0
_________________________________________________________________
Embedding (Embedding) (None, 500, 128) 1280000
_________________________________________________________________
LSTM (LSTM) (None, 32) 20608
_________________________________________________________________
Output (Dense) (None, 1) 33
=================================================================
```



然后，你可以使用标准 fit()、evaluate() 函数与推理模型。



**结论**



本快速教程介绍了如何利用 Google Colab 上的免费 Cloud TPU 资源更快地训练 Keras 模型。



## 外文博客：

<https://www.dlology.com/blog/how-to-train-keras-model-x20-times-faster-with-tpu-for-free/>

<https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/>

# 5、github 制作博客

<https://pages.github.com/>

https://github.com/johno/pixyll

# 6、利用Colab上的TPU训练Keras模型（完整版）

<https://blog.csdn.net/big91987/article/details/87898100>

## colab上导入云盘，下载云盘数据

```
# 导入
from google.colab import drive
drive.mount('/content/drive/')

# 切换目录
import os
os.chdir('/content/drive/My Drive/app/DuReader/tensorflow')
os.getcwd()

# 下载
from google.colab import files
files.download('/content/drive/My Drive/app/DuReader/data/logs')
```





# 7、bert相关

## 1、keras实现bert

<https://github.com/Separius/BERT-keras>

## 2、Bert as service

https://github.com/lbda1/bert-as-service

![img](https://github.com/lbda1/bert-as-service/raw/master/.github/demo.gif)

# 8、TriviaQA：用于阅读理解和问答的大规模数据集

<http://nlp.cs.washington.edu/triviaqa/>

华盛顿大学

# 9、显卡算力对照

https://blog.csdn.net/Real_Myth/article/details/44308169

# 10、pytorch_pretrained-BERT

<https://github.com/huggingface/pytorch-pretrained-BERT>

## 1、运行

```
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',cache_dir='./model')

# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```



# 11、bert运行命令与结果记录

squadv2.0 --do_train=False \ 指定是否训练

```
!python run_squad.py \
   --vocab_file=BERT_LARGE_DIR/vocab.txt \
   --bert_config_file=BERT_LARGE_DIR/bert_config.json \
   --init_checkpoint=BERT_LARGE_DIR/bert_model.ckpt \
   --do_train=True \
   --train_file=SQUAD_DIR/train-v2.0.json \
   --do_predict=True \
   --predict_file=SQUAD_DIR/dev-v2.0.json \
   --train_batch_size=24 \
   --learning_rate=3e-5 \
   --num_train_epochs=1.0 \
   --max_seq_length=192 \
   --doc_stride=128 \
   --output_dir=./output \
   --use_tpu=False \
   --tpu_name='TPU_NAME' \
   --version_2_with_negative=True \

```

squadv1 - train 训练

```
!python run_squad.py \
  --vocab_file=BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=SQUAD_1/train-v1.1.json \
  --do_predict=True \
  --predict_file=SQUAD_1/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --max_seq_length=192 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

训练记录

```
INFO:tensorflow:Saving checkpoints for 7000 into ./tmp/squad_base/model.ckpt.
INFO:tensorflow:global_step/sec: 0.750541
INFO:tensorflow:examples/sec: 9.0065
INFO:tensorflow:global_step/sec: 0.823383
INFO:tensorflow:examples/sec: 9.88059
INFO:tensorflow:global_step/sec: 0.798603
INFO:tensorflow:examples/sec: 9.58323
INFO:tensorflow:Saving checkpoints for 7299 into ./tmp/squad_base/model.ckpt.
INFO:tensorflow:Loss for final step: 0.79459095.
INFO:tensorflow:training_loop marked as finished
```





TPU训练--无法训练

```
!python run_squad.py \
  --vocab_file=BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=SQUAD_1/train-v1.1.json \
  --do_predict=True \
  --predict_file=SQUAD_1/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./output_squad/ \
  --use_tpu=True \
  --tpu_name="TPU_NAME"
```





no train 验证

```
!python run_squad.py \
  --vocab_file=BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=SQUAD_1/train-v1.1.json \
  --do_predict=True \
  --predict_file=SQUAD_1/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=192 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

```
# 验证结果
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Processing example: 0
INFO:tensorflow:Processing example: 1000
INFO:tensorflow:Processing example: 2000
INFO:tensorflow:Processing example: 3000
INFO:tensorflow:Processing example: 4000
INFO:tensorflow:Processing example: 5000
INFO:tensorflow:Processing example: 6000
INFO:tensorflow:Processing example: 7000
INFO:tensorflow:Processing example: 8000
INFO:tensorflow:Processing example: 9000
INFO:tensorflow:Processing example: 10000
INFO:tensorflow:Processing example: 11000
INFO:tensorflow:Processing example: 12000
INFO:tensorflow:Processing example: 13000
INFO:tensorflow:Processing example: 14000
INFO:tensorflow:prediction_loop marked as finished
INFO:tensorflow:prediction_loop marked as finished
INFO:tensorflow:Writing predictions to: /tmp/squad_base/predictions.json
INFO:tensorflow:Writing nbest to: /tmp/squad_base/nbest_predictions.json
```





The dev set predictions will be saved into a file called `predictions.json` in the `output_dir`:

测试得分

```
!python SQUAD_1/evaluate-v1.1.py SQUAD_1/dev-v1.1.json /tmp/squad_base/predictions.json
```

结果

```\
# 训练4000步
In [2]: !python SQUAD_1/evaluate-v1.1.py SQUAD_1/dev-v1.1.json /tmp/squad_base/predictions.json
{"exact_match": 75.96026490066225, "f1": 84.63953261497892}
```

```
# 训练7000步
In [4]: !python SQUAD_1/evaluate-v1.1.py SQUAD_1/dev-v1.1.json /tmp/squad_base/predictions.json
{"exact_match": 78.18353831598864, "f1": 86.0753007751716}
```

# 12、TriviaQA：用于阅读理解的大规模远程监督挑战数据集

<https://github.com/mandarjoshi90/triviaqa>

数据集下载：<http://nlp.cs.washington.edu/triviaqa/>

<https://competitions.codalab.org/competitions/17208>



## 故事完形填空测验和ROCStories Corpora

<http://cs.rochester.edu/nlp/rocstories/>

# 13、外文论文地址

<https://arxiv.org/>

# 14、最先进的图像与自然语言处理技术

<https://paperswithcode.com/sota>

![1554735863627](C:\Users\zcr\AppData\Roaming\Typora\typora-user-images\1554735863627.png)

# 15、[DeepLearning新闻档案](http://deeplearning-news-archive.blogspot.com/)

<http://deeplearning-news-archive.blogspot.com/2019/04/>

# 16、[再谈embedding——bert详解（实战）上](https://www.jianshu.com/p/109505d2947a)

https://www.jianshu.com/u/abfe703a00fe

## 序列标注——实体识别BERT-BLSTM-CRF（下）

https://github.com/FuYanzhe2/Name-Entity-Recognition

# 17、微软开放问答数据集

<http://www.msmarco.org/>

# 18、机器阅读理解的随机答案网络

<https://github.com/kevinduh/san_mrc>

这个PyTorch软件包实现了用于机器阅读理解的随机应答网络（SAN），如下所述：

刘晓东，沉从龙，Kevin Duh，高剑峰
机器阅读理解随机答案网络
计算语言学协会第56届年会论文集（第1卷：长篇论文）
[arXiv版本](https://arxiv.org/abs/1712.03556)

Liu Xiaodong Liu，Wei Li，Yuwei Fang，Aerin Kim，Kevin Duh和
Jianfeng Gao STRAD 2.0随机应答网络
技术报告 [arXiv版本](https://arxiv.org/abs/1809.09194)

如果您使用此代码，请引用上述文章。

## 快速开始

### 设置环境

1. python3.6

2. 安装要求：

   > pip install -r requirements.txt

3. 下载数据/ word2vec

   > sh download.sh

4. 您可能需要为spacy下载en模块

   > python -m spacy download en 
   > \#default 英文模型（~50MB）python -m spacy下载en_core_web_md＃较大的英文模型（~1GB）

或者拉我们发布的码头：allenlao / pytorch-allennlp-rt

### 在SQuAD v1.1上训练SAN模型

1. 预处理数据

   > python prepro.py

2. 训练一个模型

   > python train.py

### 在SQuAD v2.0上训练SAN模型

1. 预处理数据

   > python prepro.py --v2_on

2. 训练一个模型

   > python train.py --v2_on --dev_gold data \ dev-v2.0.json

### 使用ELMo

1. 从AllenNLP下载ELMo资源

2. 使用ELMo训练模型

   > python train.py --elmo_on

请注意，我们仅在SQuAD v1.1上进行了测试。

# 19、Keras的R-NET实施

<https://github.com/YerevaNN/R-NET-in-Keras>

<http://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/>

# 20、Download_glue_data

```
''' Script for downloading all GLUE data.
Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized,
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).
mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi
'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile

# python download_glue_data.py --data_dir glue_data --tasks all

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {
    "CoLA": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
    "SST": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
    "MRPC": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
    "QQP": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
    "STS": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
    "MNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
    "SNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
    "QNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0',
    "RTE": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
    "WNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf',
    "diagnostic": 'https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'}

MRPC_TRAIN = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_test.txt'


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    # os.remove(data_file)
    print("\tCompleted!")


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
        urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
        urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
    urllib.request.urlretrieve(TASK2PATH["MRPC"], os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv")) as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
            open(os.path.join(mrpc_dir, "train.tsv"), 'w') as train_fh, \
            open(os.path.join(mrpc_dir, "dev.tsv"), 'w') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file) as data_fh, \
            open(os.path.join(mrpc_dir, "test.tsv"), 'w') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    parser.add_argument('--tasks', help='tasks to download data for as a comma separated string',
                        type=str, default='all')
    parser.add_argument('--path_to_mrpc',
                        help='path to directory containing extracted MRPC data, msr_paraphrase_train.txt and msr_paraphrase_text.txt',
                        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            format_mrpc(args.data_dir, args.path_to_mrpc)
        elif task == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
```

# 21 、使用tensorboard

```
tensorboard --logdir=D://TensorBoard//test
```

22