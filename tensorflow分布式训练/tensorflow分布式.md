# tensorflow分布式

本文记录基于spark集群进行tensorflow分布式训练的过程。

一、配置及提交

安装tensorflowonspark包

```shell
pip install tensorflowonspark==1.4.4  # 配合tensorflow版本为1.15.0
```

任务提交

```shell
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue root.report_queue \
    --num-executors 100 \ # 执行器总数=num_ps + num_worker
    --executor-memory 70G \ # 内存，目前不知道为啥会占用这么多内存
    --py-files ./lib/tfspark.zip,./src.zip \ # 如果安装到miniconda包里，应该可以不再上传
    --conf spark.dynamicAllocation.enabled=false \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.yarn.dist.archives=viewfs://jssz-bigdata-cluster/department/buss_product/chenjiawei/thirdparty/miniconda3.zip#py37 \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=py37/miniconda3/bin/python \
    ./src/demo3/spark_task.py \
    --train_data hdfs://jssz-bigdata-ns2/department/buss_product/chenjiawei/playpage/ftrl_ctr_service/feature_extract/dpa/20200503/sign \
    --mode train \ # 模式
    --num_ps 20 \
    --model hdfs://jssz-bigdata-ns2/department/buss_product/chenjiawei/playpage/ftrl_ctr_service/feature_extract/model/ # 模型保存路径
```

二、应用开发

2.1 整体流程

（1）启动集群

```python
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster
sc = SparkContext(conf=SparkConf().setAppName("app_name"))
cluster = TFCluster.run(sc=sc, # spark上下文
                         map_fun=tf_func, # 训练流程入口
                         tf_args=args, # 一些参数
                         num_executors=cluster_size, # 节点总数
                         num_ps=num_ps, # ps总数
                         input_mode=1) # 输入方式，这里选spark方式
```

（2）加载训练样本到rdd

```python
rdd = sc.textFile(data_path).filter(negative_sampling).repartition(160)
```

（3）启动训练

```python
cluster.train(rdd, 
              num_epochs, # 训练轮数，这里实现是复制rdd 
              feed_timeout=600000) # feed_timeout，等待时间
```

（4）关闭集群

```python
cluster.shutdown()
```

这里，关键的编程内容为tf_func的编写

2.2 tf_func的开发

（1）入口参数

```python
def tf_func(args, ctx)  # args是TFCluster.run接口传入的参数，ctx为TFNodeContext对象
```

（2）定义数据输入

```python
tf_feed = ctx.get_data_feed(args.mode == 'train') # 拿到读数据队列，底层通过多进程传输
def rdd_generator():
	while not tf_feed.should_stop():
		batch = tf_feed.next_batch(500)
		if len(batch) == 0:
			return
		feature_batch = []
		label_batch = []
		for row in batch:
			toks = row.split("\t")[1].split(" ")
			label = [int(toks[0])]
			feature = []
			for elem in toks[2:]:
				idx = mmh3.hash128(elem, signed=False) % (vocab_size - 1) + 1 
				feature.append(idx)
			label_batch.append(label)
			feature_batch.append(feature)
		feature_batch = tf.keras.preprocessing.sequence.pad_sequences(feature_batch)
		yield (feature_batch, label_batch)
```

（3）构造静态图

```python
    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        print("{} worker start".format(datetime.datetime.now().isoformat()))
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            feature = tf.placeholder(tf.int64, shape=[None, None])
            label = tf.placeholder(tf.float32, shape=[None, 1])

            partitioner = tf.min_max_variable_partitioner(
                max_partitions=args.num_ps,
                min_slice_size=64 << 20)

            with tf.variable_scope("sparse_embedding", partitioner = partitioner): 
                embedding_matrix = tf.get_variable("weight", [vocab_size, 1],
                        tf.float32,
                        initializer=tf. random_normal_initializer(stddev=0.1),
                        trainable=True) # 分布式的embedding变量

            emb = tf.nn.embedding_lookup(embedding_matrix, feature)
            logit = tf.reduce_sum(emb, axis=1)
            pred_op = tf.nn.sigmoid(logit)
            loss_op = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, logit))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_op)

            init_op = tf.global_variables_initializer()
```

