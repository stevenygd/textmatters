"""
This is the script version of the Pipeline notebook, which does the 
following things:
1. ablating an image with text.
2. passing the (original, ablated) image pair through a captioning network, and
3. scoring the difference.
"""

import matplotlib.pyplot as plt
import numpy as np 
from scipy import misc
import json
from pprint import pprint
import subprocess

import sys
sys.path.insert(0, "coco-text/")  #directory for ablation codes

import ablation
import coco_text
from semantic_dist import *

# n samples per batch since reading and writing to disk is crappishly slow
batch_size = 100

IMG_PATH = 'data/coco/'
IMG_TYPE = 'train2014'            #input directory to sample from
TMP_PATH = 'tmp/'                 #tmp folder to put tmp images, caption jsons, etc.
CAPTION_PATH = 'neuraltalk2/'       #captioning code folder
MODEL_PATH = 'model/neuraltalk2/model_id1-501-1448236541.t7'

#generate and save ablation
#TODO: no need to save to disk. REALLY stupid.
text_data = coco_text.COCO_Text("coco-text/COCO_Text.json")
results = ablation.gen_ablation(mode = 'gaussian', ct = text_data, ksize=(7,7),sigma=7.)
for imgId, old, new in results:
    misc.imsave(TMP_PATH+str(imgId)+"_orig.jpg",old)
    misc.imsave(TMP_PATH+str(imgId)+"_ablt.jpg",new)

#captioning using shell call to torch
run_cmd = "cd "+CAPTION_PATH + " && "+\
          "th eval.lua -model  ../" +MODEL_PATH+\
          " -image_folder ../"+TMP_PATH+" && "+\
          " mv vis/vis.json ../tmp/"
p = subprocess.Popen(run_cmd,shell=True, stdout=subprocess.PIPE)

#poll until finished
while True:
    out = p.stdout.readline()
    if out == '' and p.poll() != None:
        print "Exiting with {}".format(p.poll())
        break
    if out != '':
        print out

# load generated captions
with open(TMP_PATH+'vis.json') as f:
    result = json.load(f)

# split results
ablated = [d['caption'] for d in result[::2]]
original = [d['caption'] for d in result[1::2]]

#calculate scores
stoplist = set('for a of the and to in its his her'.split())
ablated, original = pre_process(ablated, ignore=stoplist),pre_process(original, ignore=stoplist)
scores = map(lambda x: calc_inter_union(*x), zip(ablated, original))
print scores