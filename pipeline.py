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
import shutil

# n samples per batch since reading and writing to disk is crappishly slow
batch_size = 100

INPUT_PATH   =  os.path.join(FD, 'input')
INPUT_FILE   = 'test.pkl' # contains imgIds to compute the score for
OUTPUT_PATH  = os.path.join(FD, 'output')
IMG_PATH     = os.path.join(FD, 'data', 'coco')
IMG_TYPE     = 'train2014'                            # input directory to sample from
CAPTION_PATH = os.path.join(FD, 'neuraltalk2')        # captioning code folder
MODEL_PATH   = os.path.join(FD, 'model', 'neuraltalk2', 'model_id1-501-1448236541.t7')

COCO_PATH = os.path.join(FD,'data','coco')
COCO_ANNO_PATH = os.path.join(COCO_PATH, 'annotations')
COCO_TEXT_PATH = os.path.join(FD, 'coco-text')
sys.path.insert(0, COCO_TEXT_PATH)
import coco_text as ct
ct = ct.COCO_Text(os.path.join(COCO_TEXT_PATH, 'COCO_Text.json'))

sys.path.insert(0, os.path.join(FD, 'coco', 'PythonAPI'))
from pycocotools.coco import COCO
coco = ablation.coco

def run(amode='gaussian', input_file=INPUT_FILE, output_file=INPUT_FILE, batch_size=1, category = ''):

    # Clean up the previous results.
    print("Cleaning up")
    abs_tmp_dir = os.path.join(FD, "tmp_%s"%output_file)
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
    # Default mode is blackout
    imgIds = [int(x) for x in imgIds]
    args = {'imgIds' = imgIds,
            'mode'=amode,
            'out_path'=abs_tmp_dir,
            'ct' = ct,
            'coco' = coco,
            'ksize'=(7,7),
            'sigma'=7.,
            'width'=7}
    if category != '': args['category'] = category
    results = ablation.ablate(**args)
    # results = ablation.ablate(
    #         imgIds = imgIds,
    #         mode=amode,
    #         out_path=abs_tmp_dir,
    #         ct = ct,
    #         coco = coco,
    #         ksize=(7,7),
    #         sigma=7.,
    #         width=7)

    #sanity check
    assert len(results)==len(imgIds), "Image missing after ablation, original %d, after %d"%(len(imgIds), len(results))

    # Move the original images into the folder, waiting for captioning.
    for idx, (imgId, src, _) in enumerate(results):
        dst = os.path.join(abs_tmp_dir, "%s_%s_orig.jpg"%(str(idx).zfill(16), str(imgId)))
        print "Copy from %s to %s"%(src, dst)
        shutil.copyfile(src, dst)

    #captioning using shell call to torch
    run_cmd = "cd "+CAPTION_PATH + " && "+\
              "th eval.lua -model  " +MODEL_PATH+\
              " -num_images -1" + \
              " -batch_size " + str(batch_size) + \
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

def makePickle(fname, coco_image_ids):
    with open(os.path.join(INPUT_PATH, fname),'w+') as f:
        pkl.dump(coco_image_ids, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Running experiments through neural talk caption generator')
    parser.add_argument('--input_file',   default=INPUT_FILE, type=str, help='The name of the input file')
    parser.add_argument('--ablt_meth' ,   default='gaussian', type=str, help='Add ablation methods.')
    parser.add_argument('--out_file',     default=INPUT_FILE, type=str, help='Name of the out file')
    parser.add_argument('--batch_size',   default=1,          type=int, help='The batch size')
    parser.add_argument('--category',     default='',         type=str, help='The category to be preserved in ablation,\
        If none will preserve all instances or according to ablt_meth')

    args = parser.parse_args()
    run(amode=args.ablt_meth, input_file=args.input_file, output_file=args.out_file, batch_size=args.batch_size,
        category = args.category)
