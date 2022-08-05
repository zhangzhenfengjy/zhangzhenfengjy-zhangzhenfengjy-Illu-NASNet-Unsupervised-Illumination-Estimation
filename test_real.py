import torch
import numpy as np
from torch.autograd import Variable
import argparse
import random
import os
import models
import os.path as osp
import cv2
import FineTuneLayer as ft
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', help='path to  real images')
parser.add_argument('--imList',help='path to image list')

parser.add_argument('--experiment', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experimentLight', default=None, help='the path to the model of first cascade' )
parser.add_argument('--experimentFT', default=None, help='the path to the model of bilateral solver')
parser.add_argument('--testRoot',help='the path to save the testing errors' )

# The basic testing setting
parser.add_argument('--nepoch', type=int, default=14, help='the number of epoch for testing')
parser.add_argument('--nepochLight', type=int, default=14, help='the number of epoch for testing')
parser.add_argument('--nepochFT', type=int, default=15, help='the number of epoch for bilateral solver')
parser.add_argument('--niterFT', type=int, default=1000, help='the number of iterations for testing')

parser.add_argument('--imHeight', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=256, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height /width of the envmap predictions')
parser.add_argument('--envCol', type=int, default=160, help='the height /width of the envmap predictions')
parser.add_argument('--envHeight', type=int, default=8, help='the height /width of the envmap predictions')
parser.add_argument('--envWidth', type=int, default=16, help='the height /width of the envmap predictions')

parser.add_argument('--SGNum', type=int, default=12, help='the number of spherical Gaussian lobes')
parser.add_argument('--offset', type=float, default=1, help='the offset when train the lighting network')

parser.add_argument('--cuda', action = 'store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for testing network')

parser.add_argument('--level', type=int, default=1, help='the cascade level')
parser.add_argument('--isLight', action='store_true', help='whether to predict lightig')
parser.add_argument('--isFT', action='store_true', help='whether to use bilateral solver')

# Image Picking
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'w%d_h%d' % (opt.imWidth, opt.imHeight )

if opt.experimentLight is None:
    opt.experimentLight = 'Light_sg%d_offset%.1f' % \
            (opt.SGNum, opt.offset )

if opt.experimentFT is None:
    opt.experimentFT = 'fine_Tune_w%d_h%d' % (opt.imWidth, opt.imHeight )

experiments = opt.experiment
experimentsLight = opt.experimentLight
experimentsFT = opt.experimentFT
nepochs = opt.nepoch
nepochsLight = opt.nepochLight
nepochsFT = opt.nepochFT
nitersFT = opt.niterFT

imHeights = opt.imHeight
imWidths = opt.imWidth
#创建一个testRoot，文件夹名称为Real20
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp *.py %s' % opt.testRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

opt.batchSize = 1
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


imBatchSmall = Variable(torch.FloatTensor(opt.batchSize, 3, opt.envRow, opt.envCol ) )
# BRDF Predictioins
encoder = models.encoder0().eval()
albedoDecoder = models.decoder0(mode=0).eval()
normalDecoder = models.decoder0(mode=1).eval()
shadingDecoder = models.decoder0(mode=2).eval()


# Load weight
encoder.load_state_dict(
            torch.load('{0}/encoder{1}.pth'.format(experiments, nepochs-1), map_location='cpu' ).state_dict() )
albedoDecoder.load_state_dict(
            torch.load('{0}/albedo{1}.pth'.format(experiments, nepochs-1) , map_location='cpu').state_dict() )
normalDecoder.load_state_dict(
            torch.load('{0}/normal{1}.pth'.format(experiments, nepochs-1) , map_location='cpu').state_dict() )
shadingDecoder.load_state_dict(
            torch.load('{0}/shading{1}.pth'.format(experiments, nepochs-1) , map_location='cpu').state_dict() )


for param in encoder.parameters():
    param.requires_grad = False
for param in albedoDecoder.parameters():
    param.requires_grad = False
for param in normalDecoder.parameters():
    param.requires_grad = False
for param in shadingDecoder.parameters():
    param.requires_grad = False



    # Light network
    lightEncoder = models.encoderLight(SGNum = opt.SGNum).eval()
    axisDecoder = models.decoderLight(mode=0, SGNum = opt.SGNum).eval()
    lambDecoder = models.decoderLight(mode=1, SGNum = opt.SGNum).eval()
    weightDecoder = models.decoderLight(mode=2, SGNum = opt.SGNum).eval()

    lightEncoder.load_state_dict(
                torch.load('{0}/lightEncoder{1}.pth'.format(experimentsLight, nepochsLight-1) , map_location='cpu').state_dict() )
    axisDecoder.load_state_dict(
                torch.load('{0}/axisDecoder{1}.pth'.format(experimentsLight, nepochsLight-1) , map_location='cpu').state_dict() )
    lambDecoder.load_state_dict(
                torch.load('{0}/lambDecoder{1}.pth'.format(experimentsLight, nepochsLight-1) , map_location='cpu').state_dict() )
    weightDecoder.load_state_dict(
                torch.load('{0}/weightDecoder{1}.pth'.format(experimentsLight, nepochsLight-1) , map_location='cpu').state_dict() )

    for param in lightEncoder.parameters():
        param.requires_grad = False
    for param in axisDecoder.parameters():
        param.requires_grad = False
    for param in lambDecoder.parameters():
        param.requires_grad = False
    for param in weightDecoder.parameters():
        param.requires_grad = False

    if opt.isFT:
        # FT network
        albedoFT = ft.BilateralLayer(mode = 0 )
        shadingFT = ft.BilateralLayer(mode = 2 )
        normalFT = ft.BilateralLayer(mode = 4)

        albedoFT.load_state_dict(
                torch.load('{0}/albedoFt{1}_{2}.pth'.format(experimentsFT, nepochsFT-1, nitersFT ) , map_location='cpu').state_dict() )
        shadingFT.load_state_dict(
                torch.load('{0}/shadingFt{1}_{2}.pth'.format(experimentsFT, nepochsFT-1, nitersFT ) , map_location='cpu').state_dict() )
        normalFT.load_state_dict(
                torch.load('{0}/normalFt{1}_{2}.pth'.format(experimentsFT, nepochsFT-1, nitersFT ) , map_location='cpu').state_dict() )

        for param in albedoFT.parameters():
            param.requires_grad = False
        for param in shadingFT.parameters():
            param.requires_grad = False
        for param in normalFT.parameters():
            param.requires_grad = False

#########################################
####################################
outfilename = opt.testRoot + '/results'
outfilename = outfilename + '_brdf%d' % nepochs
if opt.isLight:
    outfilename += '_light%d' % nepochsLight

os.system('mkdir -p {0}'.format(outfilename ) )
print(os.system('mkdir {0}'.format(outfilename ) ))
#将imList中的路径和dataRoot拼接起来

with open(opt.imList, 'r') as imIdIn:
    imIds = imIdIn.readlines()
imList = [osp.join(opt.dataRoot,x.strip()) for x in imIds ]
imList = sorted(imList )

j = 0
for imName in imList:
    j += 1
    print('%d/%d: %s' % (j, len(imList), imName) )

    imBatches = []
    imOutputNames = []
    #imId = imName.split('/')[-1]
    imId = imName.split('\\')[-1]
    print(imId )
    imOutputNames.append(osp.join(outfilename, imId ) )

    print(imName)
    # Load the image from cpu to gpu
    assert(osp.isfile(imName ) )
    im_cpu = cv2.imread(imName )[:, :, ::-1]
    nh, nw = im_cpu.shape[0], im_cpu.shape[1]

    # Resize Input Images
    if nh < nw:
        newW = imWidths
        newH = int(float(imWidths ) / float(nw) * nh )
    else:
        newH = imHeights
        newW = int(float(imHeights ) / float(nh) * nw )

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

        newImWidth = newW
        newImHeight = newH

        im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
        im = im / im.max()
        imBatches.append( Variable(torch.from_numpy(im**(2.2) ) ).to(device) )

    nh, nw = newImHeight, newImWidth

    newEnvWidth, newEnvHeight, fov = 0, 0, 0
    if nh < nw:
        fov = 57
        newW = opt.envCol
        newH = int(float(opt.envCol ) / float(nw) * nh )
    else:
        fov = 42.75
        newH = opt.envRow
        newW = int(float(opt.envRow ) / float(nh) * nw )

    if nh < newH:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_AREA )
    else:
        im = cv2.resize(im_cpu, (newW, newH), interpolation = cv2.INTER_LINEAR )

    newEnvWidth = newW
    newEnvHeight = newH

    im = (np.transpose(im, [2, 0, 1] ).astype(np.float32 ) / 255.0 )[np.newaxis, :, :, :]
    im = im / im.max()
    imBatchSmall = Variable(torch.from_numpy(im**(2.2) ) ).to(device)
    renderLayer = models.renderingLayer(isCuda = opt.cuda,
                                        imWidth=newEnvWidth, imHeight=newEnvHeight, fov = fov,
                                        envWidth = opt.envWidth, envHeight = opt.envHeight)

    output2env = models.output2env(isCuda = opt.cuda,
                                   envWidth = opt.envWidth, envHeight = opt.envHeight, SGNum = opt.SGNum)

    ########################################################
    # Build the network architecture #

    ################# BRDF Prediction ######################
    inputBatch = imBatches[0]
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )

    albedoPred = 0.5 * (albedoDecoder(imBatches[0], x1, x2, x3, x4, x5, x6) + 1)
    normalPred = normalDecoder(imBatches[0], x1, x2, x3, x4, x5, x6)
    shadingPred = shadingDecoder(imBatches[0], x1, x2, x3, x4, x5, x6 )

    # Normalize Albedo
    bn, ch, nrow, ncol = albedoPred.size()
    albedoPred = albedoPred.view(bn, -1)
    albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
    albedoPred = albedoPred.view(bn, ch, nrow, ncol)

    ################# Lighting Prediction ###################
    if opt.isLight:

        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch )
        # Prediction
        axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, imBatchSmall )
        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum*3, envRow, envCol ), lambPred, weightPred], dim=1)

        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )

        diffusePred, specularPred = renderLayer.forwardEnv(albedoPred, normalPred,
                shadingPred, envmapsPredImage )

        diffusePredNew, specularPredNew = models.LSregressDiffSpec(
                diffusePred,
                specularPred,
                imBatchSmall,
                diffusePred, specularPred )
        renderedPred = diffusePredNew + specularPredNew

        cDiff, cSpec = (torch.sum(diffusePredNew) / torch.sum(diffusePred )).data.item(), ((torch.sum(specularPredNew) ) / (torch.sum(specularPred) ) ).data.item()
        if cSpec < 1e-3:
            cAlbedo = 1/ albedoPred.max().data.item()
            cLight = cDiff / cAlbedo
        else:
            cLight = cSpec
            cAlbedo = cDiff / cLight
            cAlbedo = np.clip(cAlbedo, 1e-3, 1 / albedoPred.max().data.item() )
            cLight = cDiff / cAlbedo
        envmapsPredImage = envmapsPredImage * cLight

        diffusePred = diffusePredNew
        specularPred = specularPredNew


    #################### Output Results #######################
    nh = 480
    nw = 720

    #save predictions
    # Save the albedo
    albedoPred = (albedoPred * cAlbedo).data.cpu().numpy().squeeze()


    albedoPred = albedoPred.transpose([1, 2, 0] )
    albedoPred = (albedoPred ) ** (1.0/2.2 )
    albedoPred = cv2.resize(albedoPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

    albedoPredIm = (np.clip(255 * albedoPred, 0, 255) ).astype(np.uint8)

    #cv2.imwrite(albedoImName, albedoPredIm[:, :, ::-1] )
    cv2.imwrite('Real20/results_brdf14_brdf7', albedoPredIm[:, :, ::-1])
    print(cv2.imwrite('your save path/albedo'+str(j)+'.png', albedoPredIm[:, :, ::-1] ))

    # Save the normal
    normalPred = normalPred.data.cpu().numpy().squeeze()
    normalPred = normalPred.transpose([1, 2, 0] )
    normalPred = cv2.resize(normalPred, (nw, nh), interpolation = cv2.INTER_LINEAR )
    #np.save(normalNames[n], normalPred )
    normalPredIm = (255 * 0.5*(normalPred+1) ).astype(np.uint8)
    #cv2.imwrite(normalImNames[n], normalPredIm[:, :, ::-1] )
    print(cv2.imwrite('your save path/normal'+str(j)+'.png', normalPredIm[:, :, ::-1]))

    if opt.isFT:
        # Save the albedo ft

        albedoFTPred = (albedoFTPred * cAlbedo).data.cpu().numpy().squeeze()
        albedoFTPred = albedoFTPred.transpose([1, 2, 0] )
        albedoFTPred = (albedoFTPred ) ** (1.0/2.2 )
        albedoFTPred = cv2.resize(albedoFTPred, (nw, nh), interpolation = cv2.INTER_LINEAR )

        albedoFTPredIm = ( np.clip(255 * albedoFTPred, 0, 255) ).astype(np.uint8)
        #cv2.imwrite(albedoImNames[n].replace('albedo', 'albedoFT'), albedoFTPredIm[:, :, ::-1] )


    if opt.isLight:
        # Save the envmap
        envmapsPredImage = envmapsPredImage.data.cpu().numpy().squeeze()
        envmapsPredImage = envmapsPredImage.transpose([1, 2, 3, 4, 0] )
        # Flip to be conincide with our dataset env.npz
            # np.savez_compressed(envmapPredImNames[n],
            #         env = np.ascontiguousarray(envmapsPredImage[:, :, :, :, ::-1] ) )
        np.savez_compressed('your save path/env'+str(j)+'.npz',
                                envmapsPredImage[:, :, :, :, ::-1])
        #utils.writeEnvToFile(envmapsPredImages[n], 0, envmapPredImNames[n], nrows=24, ncols=16 )

        # Save the shading

        envmapsPred = envmapsPred.data.cpu().numpy()
        #  np.save(envmapsPredSGNames[n], envmapsPred )
        shading = utils.predToShading(envmapsPred, SGNum = opt.SGNum )
        shading = shading.transpose([1, 2, 0] )
        shading = shading / np.mean(shading ) / 3.0
        shading = np.clip(shading, 0, 1)
        shading = cv2.resize(shading, (nw, nh), interpolation=cv2.INTER_LINEAR)
        shading = (255 * shading ** (1.0/2.2) ).astype(np.uint8 )
        #cv2.imwrite(shadingNames[n], shading[:, :, ::-1] )
        cv2.imwrite('your save path/shading'+str(j)+'.png', shading[:, :, ::-1])

        # for n in range(0, len(cLights) ):
        #     io.savemat(cLightNames[n], {'cLight': cLights[n] } )

        # Save the rendered image

        renderedPred = renderedPred.data.cpu().numpy().squeeze()
        renderedPred = renderedPred.transpose([1, 2, 0] )
        renderedPred = (renderedPred / renderedPred.max() ) ** (1.0/2.2)
        renderedPred = cv2.resize(renderedPred, (nw, nh), interpolation = cv2.INTER_LINEAR )
        #np.save(renderedNames[n], renderedPred )

        renderedPred = (np.clip(renderedPred, 0, 1) * 255).astype(np.uint8 )
        cv2.imwrite('your save path/renderedIm'+str(j)+'.png', renderedPred[:, :, ::-1])

    # Save the image

    #cv2.imwrite(imOutputNames[0], im_cpu[:,:, ::-1] )
    cv2.imwrite('your save path/im'+str(j)+'.png', im_cpu[:, :, ::-1])
