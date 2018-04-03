## Logging Device placement

```
# python log_device.py
...
MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
2018-04-03 14:57:00.495959: I tensorflow/core/common_runtime/placer.cc:875] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-04-03 14:57:00.496005: I tensorflow/core/common_runtime/placer.cc:875] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-04-03 14:57:00.496037: I tensorflow/core/common_runtime/placer.cc:875] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
```

## Refs

* [Using GPUs](https://www.tensorflow.org/programmers_guide/using_gpu)
