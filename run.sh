#! /bin/bash

#run script 
CUDA_VISIBLE_DEVICES=0 python pipeline.py --input_file toilet_img_ids --out_file toilet_only_median_bg  --ablt_meth median_bg  --batch_size 32 --category toilet 
