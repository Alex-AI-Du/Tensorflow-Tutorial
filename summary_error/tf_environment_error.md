---------
"2018-02-19 14:01:12.094674: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2<br>
2018-02-19 14:01:12.362715: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:<br>
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.645<br>
pciBusID: 0000:01:00.0<br>
totalMemory: 11.00GiB freeMemory: 9.08GiB<br>
2018-02-19 14:01:12.362942: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0<br>
2018-02-19 14:01:12.847857: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 8794 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)":<br>
pip操作安装了的tensorflow会有提示：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2等上面提示，原因好像是编译问题，cpu没有得到充分的利用，如果从官网下载源码不会有这种问题，如果用gpu可以忽略，使用：<br>
		import os
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告

---------