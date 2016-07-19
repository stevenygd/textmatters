#! /bin/bash

#run script 
CUDA_VISIBLE_DEVICES=3 python pipeline.py --input_file no_texts_img_ids.pkl --out_file no_text_black_bg  --ablt_meth destroy  --batch_size 64 
