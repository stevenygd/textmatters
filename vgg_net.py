
# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

import numpy as np
import matplotlib.pyplot as plt

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1 )
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1 )
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1 )
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1 )
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1 )
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1 )
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1 )
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1 )
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1 )
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1 )
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1 )
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1 )
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1 )
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    output_layer = net['prob']

    return net, net['prob'] # the whole net and the output layer

def load_param(output_layer, pickle_file):
    """
    load the model and parameters from specified pickle_file into the lasagne layers,
    with output layer set to [output_layer]
    """
    import pickle

    model = pickle.load(open(pickle_file))
    CLASSES = model['synset words']
    # MEAN_IMAGE = model['mean value']  # right now do not subtract mean image during testing

    lasagne.layers.set_all_param_values(output_layer, model['param values'])

def prepare_image(im):
    import io
    import skimage.transform

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    # WARNING: DID NOT subtract mean during testing
    # im = im - MEAN_IMAGE 

    return rawim, floatX(im[np.newaxis])

def extract_feature(img, layer):
    """
    run the pretrained network on the image and extract the output from [layer].
    [args]: img -- np array representing the test image
            layer  --  lasagne layer object specifying which layer to extract from 
    """
    rawim, im = prepare_image(img)
    feature = np.array(lasagne.layers.get_output(layer, im, deterministic=True).eval())
    assert feature.shape == (1, 512, 14, 14)
    return feature


if __name__ == "__main__":
    from scipy import misc
    import os 
    from six.moves import cPickle

    DATAPATH = os.getcwd()+os.sep+"data"+os.sep

    # set up the pretrained VGG Net 16
    net, output_layer = build_model()
    load_param(output_layer, DATAPATH+"vgg16.pkl")

    for f in os.listdir(DATAPATH+"Flicker8k_Dataset"):
        if f.endswith(".jpg"): 
            img = misc.imread(DATAPATH+"Flicker8k_Dataset"+os.sep+f)
            feat = extract_feature(img, net["conv5_3"])

            with open(DATAPATH+"Flicker8k_features"+os.sep+f.replace('.jpg','.pkl'), 'w+') as out:
                cPickle.dump(feat, out, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            continue
    # rawimd, im = prep_image(img)

    # prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
    # top5 = np.argsort(prob[0])[-1:-6:-1]

    # plt.figure()
    # plt.imshow(rawimd.astype('uint8'))
    # plt.axis('off')
    # for n, label in enumerate(top5):
    #     plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, CLASSES[label]), fontsize=14)

