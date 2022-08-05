import argparse
import random
import torch
import os
import cv2
import numpy as np
from torch.autograd import Variable
import models
import json
import compare_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='/remote-home/cs_ai_hch/hrj/dataset/iiw-dataset/data/', help='path to real images')
parser.add_argument('--modelRoot', default='experiment', help='the path to the model of first cascade')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')
parser.add_argument('--epochs', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--imWidth', type=int, default=160, help='the height / width of the input image to network')
parser.add_argument('--imHeight', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobes')

opt = parser.parse_args()
print(opt)
modelRoot = opt.modelRoot
opt.gpuId = opt.deviceIds[0]

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

encoders = models.encoder0(cascadeLevel = 0 )
albedoDecoders = models.decoder0(mode=0 )

# Load weight
encoders.load_state_dict(
        torch.load('{0}/encoder0_{1}.pth'.format(opt.modelRoot, opt.epochs-1)).state_dict())
albedoDecoders.load_state_dict(
        torch.load('{0}/albedo0_{1}.pth'.format(opt.modelRoot, opt.epochs-1)).state_dict())

for param in encoders.parameters():
    param.requires_grad = False
for param in albedoDecoders.parameters():
    param.requires_grad = False

##################################################

# Send things into GPU
encoders = encoders.cuda(opt.gpuId)
albedoDecoders = albedoDecoders.cuda(opt.gpuId)

with open('./IIWTest1.txt', 'r') as imIdIn:
    imIds = imIdIn.readlines()
imList = [os.path.join(opt.dataRoot,x.strip() ) for x in imIds ]
path = sorted(imList )

# path = glob(opt.dataRoot+'/*.*')

j = 0
# for i in range(len(path)):
count = 0.0
whdr_sum = 0.0
whdr_mean = 0.0
trainingLog = open('{0}/testLog_IIW_{1}.txt'.format(opt.modelRoot, opt.epochs), 'w')
for imName in imList:
    j += 1
    # print('%d/%d: %s' % (j, len(path), imName))
    imId = imName.split('/')[-1]
    # imId = imId.split('.')[0]

    # Load the image from cpu to gpu
    assert(os.path.isfile(imName))
    im_cpu = cv2.imread(imName)[:, :, ::-1]
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    if nh < nw:
        newW = opt.imWidth
        newH = int(float(opt.imWidth) / float(nw) * nh )
    else:
        newH = opt.imHeight
        newW = int(float(opt.imHeight) / float(nh) * nw )

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatches = (Variable(torch.from_numpy(im**(2.2))).cuda())


    ################# BRDF Prediction ######################
    inputBatch = imBatches
    x1, x2, x3, x4, x5, x6= encoders(inputBatch)

    albedoPred = 0.5 * (albedoDecoders(imBatches, x1, x2, x3, x4, x5, x6) + 1)

    # Normalize Albedo and depth
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)

    #################### Output Results #######################
    # Save the albedo
    # nh = 480
    # nw = 640
    albedoPred = albedoPred.data.cpu().numpy().squeeze()

    albedoPred = albedoPred.transpose([1, 2, 0])
    albedoPred = albedoPred ** (1.0/2.2)
    albedoPred = cv2.resize(albedoPred, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # albedoPredIm = (np.clip(255 * albedoPred, 0, 255)).astype(np.uint8)

    judgement_path = imName.replace('png', 'json')
    judgements = json.load(open(judgement_path))

    count += 1.0
    whdr, _, _ = utils.compute_whdr(albedoPred, judgements)
    whdr_sum += whdr

    print('img: {0}, whdr: current {1} average {2}'.
          format(imId, whdr, whdr_sum / count))
    utils.writewhdrToFile(imId, whdr, trainingLog, whdr_sum / count)

whdr_mean = whdr_sum / count
print('whdr : {0}'.format(whdr_mean))
print('albedo number: {0}'.format(count))





