# tfmodels

> GeForce GTX TITAN Z  6080MiB

| model      | alexnet      | VGGnet-16   | VGGnet-19 | Googlenet |Resnet-50  | Resnet-101 |Resnet-152 |  wrn       | 
| :----------| :----------  | :---------- |:----------|:----------|:----------| :----------|:----------| :---------- |
| describe   |           	|             |           |   v1      |      v2   |  v2        |    v2     |  28-10 cifar-100  |
| imagesize  | 224x224x3 	| 224x224x3   | 224x224x3 | 224x224x3 | 224x224x3 | 224x224x3  | 224x224x3 | 32x32x3     |
| num_class  | 1000      	|    1000     | 1000      | 1000      | 1000      | 1000       | 1000      | 100         |
| batch_size | 16      		|    16       |  16       | 16        | 16        | 16         |  16       | 16          |  
| peak_size  | 1833.94MiB 	| 24191MiB    | 30861.9MiB| 4605.31MiB| 6887.34MiB| 20824.4MiB | 27077.8MiB| 3918.91MiB |
| nvidia-smi | 1225MiB   	| 4393MiB     | 4393MiB   | 1191MiB   | 4263MiB   | 4263MiB    | 4263MiB   | 4263MiB |
| batch_size | 32        	|    32       | 32        | 32        | 32        |  32        | 32        |  32     |  
| peak_size  | 2866.66MiB 	| 31004.1MiB  |37318.6MiB | 6800.2MiB | 10874.9MiB| 22188.7MiB | 31335.3MiB| 5823.97MiB| 
| nvidia-smi | 1257MiB 		|  4393MiB    | 4393MiB   | 2279MiB   | 4263MiB   | 5876MiB    |5876MiB(OOM)| 4263MiB | 
| batch_size | 64       	|    64       | 64        | 64        | 64        |  64        | 64        |  64  |  
| peak_size  | 3005.58MiB 	|             |           | 12245.8MiB| 17360.7MiB|            |           |13609MiB|
| nvidia-smi | 2345MiB  	|			  |           | 2343MiB   | 5876MiB   |            |           |4263MiB|
| batch_size | 128      	|    128      | 128       | 128       | 128       |  128       | 128       |  128  |  
| peak_size  | 6510.83MiB 	|             |           | 22658.5MiB|			  | 		   |           | 32917MiB    |
| nvidia-smi | 2217MiB   	|             |           | 4455MiB   |           |            |           | 5877MiB(OOM)  |
| batch_size | 256       	|    256      | 256       | 256       | 256       |  256       | 256       |  256  |  
| peak_size  | 12072.5MiB 	|             |           | 
| nvidia-smi | 4265MiB   	|             |           |		      
| batch_size | 500      	|    500      | 500       | 500       | 500       |  500       | 500       |  500  |  
| peak_size  | 22769.8MiB 	|             |           |
| nvidia-smi | 5876MiB(OOM) |             |           |



