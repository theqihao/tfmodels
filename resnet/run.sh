#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 resnet.py > x.txt
cat x.txt | grep _TFProfRoot
cat x.txt | grep _TFProfRoot | awk -F '/'  '{print $3}' | awk -F 'M'  '{print (($1/1.024/1.024)"MiB") }'