#! /bin/bash


img=$1
server=$2
#copy image to remote server
scp $img $server:textmatters/anecdotal && \

#run the caption generation
ssh $server '
	th textmatters/neuraltalk2/eval.lua -model textmatters/model/neuraltalk2/model_id1-501-1448236541.t7 -image_folder textmatters/anecdotal  -num_images 1 
' && \

#get result back
scp $server:textmatters/neuraltalk2/vis/vis.json . && \
mv vis.json $img.json && \
cat $img.json

