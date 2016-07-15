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
from six.moves import cPickle as pkl
import os
import sys
sys.path.insert(0, "coco-text/")  #directory for ablation codes
FD = os.path.dirname(os.path.realpath(__file__))
import ablation
import coco_text
from semantic_dist import *

# n samples per batch since reading and writing to disk is crappishly slow
batch_size = 100

INPUT_PATH   =  os.path.join(FD, 'input')
INPUT_FILE   = 'test.pkl' # contains imgIds to compute the score for
OUTPUT_PATH  = os.path.join(FD, 'output')
IMG_PATH     = os.path.join(FD, 'data', 'coco')
IMG_TYPE     = 'train2014'                            # input directory to sample from
TMP_PATH     = os.path.join(FD, 'tmp')                # tmp folder to put tmp images, caption jsons, etc.
CAPTION_PATH = os.path.join(FD, 'neuraltalk2')        # captioning code folder
MODEL_PATH   = os.path.join(FD, 'model', 'neuraltalk2', 'model_id1-501-1448236541.t7')

def run(amode='gaussian', input_file=INPUT_FILE, output_file=INPUT_FILE, tmp_path=TMP_PATH):

    # Clean up the previous results.
    print("Cleaning up")
    abs_tmp_dir = os.path.join(FD, tmp_path)
    if not os.path.exists(abs_tmp_dir):
        os.makedirs(abs_tmp_dir)
     
    run_cmd = "rm " + abs_tmp_dir + "/*"
    p = subprocess.Popen(run_cmd,shell=True)
    while True:
        if p.poll() != None:
            break

    #read input image ids
    with open(os.path.join(FD, INPUT_PATH, input_file)) as f:
        imgIds = pkl.load(f)
        assert len(imgIds) > 0, "Found empty input."

    # generate and save ablation
    # TODO: no need to save to disk. REALLY stupid.
    text_data = coco_text.COCO_Text("coco-text/COCO_Text.json")
    # Default mode is blackout
    imgIds = [int(x) for x in imgIds]
    results = ablation.gen_ablation(imgIds = imgIds, mode=amode, ct = text_data, ksize=(7,7),sigma=7., width=7)

    #sanity check
    assert len(results)==len(imgIds), "Image missing after ablation, original {}, after {}".format(len(imgIds), len(results))

    for idx, (imgId, old, new) in enumerate(results):
        misc.imsave(os.path.join(tmp_path, "%s_%s_orig.jpg"%(str(idx).zfill(16), str(imgId))),old)
        misc.imsave(os.path.join(tmp_path, "%s_%s_ablt.jpg"%(str(idx).zfill(16), str(imgId))),new)

    #captioning using shell call to torch
    run_cmd = "cd "+CAPTION_PATH + " && "+\
              "th eval.lua -model  " +MODEL_PATH+\
              " -num_images -1" + \
              " -batch_size 100" + \
              " -image_folder "+ abs_tmp_dir
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
    with open(os.path.join(FD, 'neuraltalk2', 'vis', 'vis.json')) as f:
        result = json.load(f)
        o = os.path.join(FD, OUTPUT_PATH, 'vis_%s.json'%output_file)
        json.dump(result, open(o, 'w+'))

    # split results
    ablated = [d['caption'] for d in result[::2]]
    original = [d['caption'] for d in result[1::2]]

    #calculate scores
    stoplist = set('for a of the and to in its his her'.split())
    ablated, original = pre_process(ablated, ignore=stoplist),pre_process(original, ignore=stoplist)
    scores = map(lambda x: calc_inter_union(*x), zip(ablated, original))

    #write results
    print("Saving results to %s"%OUTPUT_PATH)
    with open(os.path.join(FD, OUTPUT_PATH, "scores_%s.pkl"%output_file), 'w+') as f:
        pkl.dump(scores, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__=="__main__":
    # run()
    # run(amode="blackout", input_file="rel_texts_img_ids.pkl", output_file="rel_texts_gaussian", tmp_path="tmp_rel_texts_gaussian")
    run(amode="gaussian", input_file="rel_texts_img_ids.pkl", output_file="rel_texts_gaussian", tmp_path="tmp_rel_texts_gaussian")
    run(amode="blackout", input_file="no_rel_texts_img_ids.pkl", output_file="no_rel_texts_blackout", tmp_path="tmp_no_rel_texts_blackout")
    run(amode="gaussian", input_file="no_rel_texts_img_ids.pkl", output_file="no_rel_texts_gaussian", tmp_path="tmp_no_rel_texts_gaussian")
    # Previous experiments on the basic
    # run(amode="gaussian", input_file="large_text_img_ids.pkl", output_file="large_text_gaussian", tmp_path="tmp_large_text_gaussian")
    # run(amode="blackout", input_file="large_text_img_ids.pkl", output_file="large_text_blackout", tmp_path="tmp_large_text_blackout")
    # run(amode="gaussian", input_file="high_coexist_img_ids.pkl", output_file="highexist_gaussian", tmp_path="tmp_highexist_gaussian")
    # run(amode="blackout", input_file="high_coexist_img_ids.pkl", output_file="highexist_blackout", tmp_path="tmp_highexist_blackout")
