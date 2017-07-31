# MNIST with Caffe on Spark

This project intends to show that a pure CPU based neural network can scale on a spark cluster. The experimental platform is a three node cluster, each of which has 1 CPU of 4 cores.

## compile
- it requires a CPU-only caffe of version 1.0
- `sbt assembly`

## execution
+ `export MNIST_HOME=<your-path-to-this-repo>`
+ `<your-path-to-spark-submit> --master <your-spark-master-url> --driver-memory=4G --num-executors 6 --executor-cores 2 --class MnistApp target/scala-2.11/mnist-assembly-0.1-SNAPSHOT.jar 6`

## results
+ In single CPU mode, it tooks 190s to achieve 98% accuracy,
+ In cluster, with following settings, it tooks 91s to achieve the same accuracy.

```
  val num_batches_per_iter = 100
  val num_iters_per_round = 5
```
