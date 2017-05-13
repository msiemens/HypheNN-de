# HypheNN-de

A neural network that hyphenates German words. Following _B. Fritzke and
C. Nasahl, "A neural network that learns to do hyphenation"_, it uses a window
of 8 characters and determines whether the word can be hyphenated after position
4 of the current window. Currently the network achieves a success rate of 99.2 %.

The network input is encoded using one-hot encoding the character set and outputs
a single value indicating the probability of a hyphenation at the current position
being valid.


## Dependencies

- Python 3 with TensorFlow and Keras installed
- Rust


## Download & prepare data

1. Download Wiktionary dump from https://dumps.wikimedia.org/dewiktionary/latest/ (look for `dewiktionary-{timestamp}-pages-articles.xml.bz2`) and unpack it
2. Compile and run `prepare_data`:
    - `$ cd prepare_data`
    - `$ cargo run --release ../data/dewiktionary-*-pages-articles.xml ../wordlist.txt`

Notes: There's a fair amount of post-processing happening to cleanup the data. The whole process may work with other languages, but the data cleanup probably will need some adaption.
Note: The preprocessing randomizes the order of entries.


## Train

To train the network, run:

```
$ python train.py
Using TensorFlow backend.
Building model...
Done

Training model...
2017-05-13 21:58:40.012420: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:857] OS X does not support NUMA - returning NUMA node zero
2017-05-13 21:58:40.012582: I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:02:00.0
Total memory: 2.00GiB
Free memory: 1001.96MiB
2017-05-13 21:58:40.012595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:927] DMA: 0 
2017-05-13 21:58:40.012600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:937] 0:   Y 
2017-05-13 21:58:40.012610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:996] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:02:00.0)
Epoch 1/50
1253797/1253797 [==============================] - 7s - loss: 0.0450 - acc: 0.9468       
Epoch 2/50
1253797/1253797 [==============================] - 7s - loss: 0.0218 - acc: 0.9753       
...
Epoch 48/50
1253797/1253797 [==============================] - 7s - loss: 0.0076 - acc: 0.9920      
Epoch 49/50
1253797/1253797 [==============================] - 7s - loss: 0.0076 - acc: 0.9919       
Epoch 50/50
1253797/1253797 [==============================] - 7s - loss: 0.0076 - acc: 0.9920      
Done
Time: 0:06:05.55
```

The network weights are stored in `data/model.h5`

## Validate

```
$ python validate.py
Using TensorFlow backend.
Building model...
Done

2017-05-14 00:50:01.683992: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:857] OS X does not support NUMA - returning NUMA node zero
2017-05-14 00:50:01.684174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:02:00.0
Total memory: 2.00GiB
Free memory: 720.18MiB
2017-05-14 00:50:01.684190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:927] DMA: 0 
2017-05-14 00:50:01.684195: I tensorflow/core/common_runtime/gpu/gpu_device.cc:937] 0:   Y 
2017-05-14 00:50:01.684206: I tensorflow/core/common_runtime/gpu/gpu_device.cc:996] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:02:00.0)
Validating model...
1253536/1253797 [============================>.] - ETA: 0s   
Done
Result: [0.0073155245152076989, 0.99240706430147785]
Time: 0:00:40.17
```

Note: The first value is the mean square error, the second value is the
achieved accuracy.

## Run


```
$ python predict.py Silbentrennung
Using TensorFlow backend.
Building model...
Done

2017-05-14 00:49:08.660959: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:857] OS X does not support NUMA - returning NUMA node zero
2017-05-14 00:49:08.661122: I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] Found device 0 with properties: 
name: Quadro K2000
major: 3 minor: 0 memoryClockRate (GHz) 0.954
pciBusID 0000:02:00.0
Total memory: 2.00GiB
Free memory: 738.18MiB
2017-05-14 00:49:08.661136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:927] DMA: 0 
2017-05-14 00:49:08.661141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:937] 0:   Y 
2017-05-14 00:49:08.661152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:996] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K2000, pci bus id: 0000:02:00.0)
Input: Silbentrennung
Hyphenation: Sil·ben·tren·nung
```


## Tensorflow build notes

Setup: See https://gist.github.com/Mistobaan/dd32287eeb6859c6668d

Configure:

```
PYTHON_BIN_PATH=$(which python) CUDA_TOOLKIT_PATH="/usr/local/cuda" CUDNN_INSTALL_PATH="/usr/local/cuda" TF_UNOFFICIAL_SETTING=1 TF_NEED_CUDA=1 TF_CUDA_COMPUTE_CAPABILITIES="3.0" TF_CUDNN_VERSION="6" TF_CUDA_VERSION="8.0" TF_CUDA_VERSION_TOOLKIT=8.0 ./configure
```

Note: Use defaults everywhere.

Build:

```
bazel build -c opt --copt=-mavx --copt=-msse4.2 --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

If encountering problems with `Library not loaded: @rpath/libcudart.8.0.dylib`,
follow http://stackoverflow.com/a/40007947/997063.

If encountering problems regarding `-lgomp`, replace `-lgomp` with
`-L/usr/local/Cellar/llvm/4.0.0/lib/libiomp5.dylib` in
`third_party/gpus/cuda/BUILD.tpl` or comment out this line.


## References

- http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
- http://blog.aloni.org/posts/backprop-with-tensorflow/
- B. Fritzke and C. Nasahl, "A neural network that learns to do hyphenation," IJCNN-91-Seattle International Joint Conference on Neural Networks, Seattle, WA, USA, 1991, pp. 960 vol.2-. (doi: 10.1109/IJCNN.1991.155602)