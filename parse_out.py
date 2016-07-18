"""
This script takes in the out log of the captioning network, figured the 
(COCO-imgId, newImgId) pairing by parsing, group together the 
(imgId, oldCap, newCap, score) tuple and export the result as a dictionary with
schema: imgId -> orig, ablt, score
"""
import re
from semantic_dist import *
from collections import OrderedDict

import argparse
import os
from six.moves import cPickle as pkl

def parse(outPath):
    """Takes in the log file of ONE run of captioning and groups
    together the result as described above

    NOTE: Reads everything into memory at once, since out file is ~10M"""
    def _parse_id(line):
        """parse out the COCO id from the 'cp ...' line """
        pat = re.compile('(?<=[0-9]{16}_)[0-9]+') # matches numbers preceded by 16 numbers followed by a '_'
        mat = pat.search(line)
        assert not mat is None, "this line does not contain a COCO image id: {}" % line 

        s, e = mat.start(), mat.end()
        return line[s:e], line[e+1:e+5] 

    with open(outPath, 'r') as f:
        content = f.read()
    
        l = content.split('\n')
        pattern = re.compile('^cp|^image')
        l = [x for x in l if pattern.search(x)]
        id_lines, cap_lines = l[::2],l[1::2]

        d = OrderedDict() #dictionary from COCO-id to (orig_cap, new_cap)

        for idx, id_line in enumerate(id_lines):
            cap = cap_lines[idx].split(':')[-1].strip()
            cocoid, cat = _parse_id(id_line)
            if not cocoid in d:
                d[cocoid] = {}
            d[cocoid][cat] = cap

        #compute scores, need to preprocess all ablated captions and original captions
        stoplist = set('for a of the and to in its his her'.split())
        #believe that ordered dict guarantees iteration order!!!
        ablated, original = [ d[k]['ablt'] for k in d.keys()], [ d[k]['orig'] for k in d.keys()]
        ablated, original = pre_process(ablated, ignore=stoplist),pre_process(original, ignore=stoplist)
        scores = map(lambda x: calc_inter_union(*x), zip(ablated, original))
        for idx, k in enumerate(d.keys()):
            d[k]['score'] = scores[idx]

        #get ablation method
        l = id_lines[0]
        if 'blackout' in l:
            d['ablation_method'] = 'blackout'
        elif 'gaussian' in l:
            d['ablation_method'] = 'gaussian'
        elif 'median' in l:
            d['ablation_method']  = 'median'
        return d

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='The .out file to be parsed')
    args = parser.parse_args()

    d = parse(args.file)
    with open( os.path.join('output', args.file.split('/')[-1].split('.')[0].strip())+'.pkl', 'w+') as f:
        pkl.dump(d,f,protocol=pkl.HIGHEST_PROTOCOL)

