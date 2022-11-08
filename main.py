import torch
from tqdm import tqdm
import numpy as np

import models
import loss
import settings
import utils

from datetime import datetime

#dataset
import dataset
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image

def preprocess_input(data):
    if torch.cuda.is_available():
        data['image'] = data['image'].cuda()
        data['label'] = data['label'].cuda()

    #one hot label map (segmentation map / semantics map)
    label_map = data['label']
    #print(np.shape(label_map))
    bs, _, h, w = label_map.size() #there should be a batch size?
    nc = settings.NUM_CLASSES  # number of classes
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_() if torch.cuda.is_available() else torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    return input_semantics, data['image'], label_map

def divide_prediction(pred):
    fake = []
    real = []
    for p in pred:
        fake.append(p[:p.size(0) // 2]) #first half are the fake predictions
        real.append(p[p.size(0) // 2:]) #second half are the real predictions
        
    return fake, real

def generate_fake(real_image, seg_map, compute_KLD = False):
    mu, var = encoder(real_image)
    latent_vec = encoder.compute_latent_vec(mu, var)
    fake_image = generator(latent_vec=latent_vec, segmap=seg_map)

    kld_loss_batch = None
    if compute_KLD:
        kld_loss_batch = kld_loss(mu, var)  #ok

    return fake_image, kld_loss_batch

def discriminate(fake_image, real_image, seg_map):
    #TODO feed real and fake image at the same time to the discriminator, check pixp2pix_model.py line 209
    #the first half of the output is for the fake pred, the second half is the real output. 
            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
    fake_concat = torch.concat([fake_image, seg_map], dim=1)
    real_concat = torch.concat([real_image, seg_map], dim=1)
    fake_and_real_image = torch.concat([fake_concat, real_concat], dim=0)

    intermediate_predictions = discriminator(fake_and_real_image)

    #list of intermediate predictions from each layer of the discriminator
    pred_fake, pred_real =  divide_prediction(intermediate_predictions)

    return pred_fake, pred_real

def compute_generator_loss(real_image, seg_map):
    fake_image, kld_loss_batch = generate_fake(real_image, seg_map, compute_KLD=True)

    pred_fake, pred_real = discriminate(fake_image, real_image, seg_map)

    gen_loss_batch = gen_loss(pred_fake[-1]) #Take the last prediction of the discriminator and compute generator loss
    vgg_loss_batch = vgg_loss(real_image, fake_image)#OK
    #Exclude last item as it is the final prediction, only cound intermediate predictions
    feat_loss_batch = feat_loss(pred_real[:-1], pred_fake[:-1]) #TODO check if last item should be included in the feat losscompare pix2pix_model.py line 151
    #kld_loss_batch = kld_loss(mu, var)  #ok
    
    #kld loss lambda = 0.05
    #feat loss lampa = 10 - OK
    #vgg loss lambda = 10
    #TODO sum and mean? 
    generator_loss = gen_loss_batch + \
        kld_loss_batch * 0.05 + \
        vgg_loss_batch * 10 + \
        feat_loss_batch * 10

    return generator_loss

def compute_discriminator_loss(real_image, seg_map):
    with torch.no_grad(): 
        fake_image, _ = generate_fake(real_image, seg_map)
        fake_image.detach()
        fake_image.requires_grad_()
    #TODO compute prediction in one batch as described above
    
    pred_fake_d, pred_real_d = discriminate(fake_image, real_image.detach(), seg_map)
    #TODO compare against hinge loss computed as described above 
    #TODO check gen_loss => hinge loss
    #compute hinge loss as: 

    #minval = torch.min(input - 1, self.get_zero_tensor(input)) #function call can be found in loss.py line 51
    #loss = -torch.mean(minval)

    #if target is not real compute as torch.min(-input - 1, ...) ...
    # # real_disc_loss = disc_loss(pred_real_d[-1], True)
    # # fake_disc_loss = disc_loss(pred_fake_d[-1], False)
    # # d_loss = real_disc_loss + fake_disc_loss
    
    rdl = hinge_loss(pred_real_d[-1], True)
    fdl = hinge_loss(pred_fake_d[-1], False)
    d_loss = rdl + fdl

    return d_loss


def validateEpoch(path, epoch):
    print('-------- Validating image --------')
    encoder.eval()
    generator.eval()
    discriminator.eval()
    data = next(iter(validation_loader))
    seg, real, label = preprocess_input(data)
    with torch.no_grad():
        fake, _ = generate_fake(real, seg)
        #plotImages(fake[0].cpu(), real[0].cpu(), label[0].cpu())
        utils.saveValidationImage(fake[0].cpu(), real[0].cpu(), label[0].cpu(), path, epoch)
    
    encoder.train()
    generator.train()
    discriminator.train()
    print('-------- Validation complete --------')

def saveModels(filename, epoch, optional=''):
    from pathlib import Path

    Path('models/' + filename).mkdir(parents=True, exist_ok=True)

    print("saving models at epoch:", epoch)
    #print("g-loss:", generator_loss.item(), "d-loss:", d_loss.item())
    #print('g-loss: %.4f \t d-loss: %.4f \n'%(generator_loss.item(), d_loss.item()))
    torch.save(encoder.state_dict(), 'models/' + filename + '/encoder' + optional + '.pth')
    torch.save(generator.state_dict(), 'models/' + filename + '/generator' + optional + '.pth')
    torch.save(discriminator.state_dict(), 'models/' + filename + '/discriminator' + optional + '.pth')
    
def trainGauGAN(dataloader, encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator, nrEpochs, filename, startEpoch = 0):
    for epoch in tqdm(range(startEpoch, startEpoch + nrEpochs)):
        for batch_index, data in enumerate(dataloader):
            #inputs, targets = inputs.cuda(), targets.cuda()
            # print("img:",np.shape(data["image"]))
            # print("segmap",np.shape(data["seg_map"]))
            # print("label",np.shape(data["label"]))
            seg_map, real_image, _ = preprocess_input(data)

            #print(np.shape(seg_map))
            #print(np.shape(real_image))

            #train generator 1 step
            gen_optimizer.zero_grad()
            generator_loss = compute_generator_loss(real_image, seg_map)
            generator_loss.backward()
            gen_optimizer.step()

            #train discriminator 1 step
            disc_optimizer.zero_grad()
            #TODO check if we should generate new image and detach it from gpu => set requires_grad_()
            #A reason why we should generate a new image is that we have trained the generator a step so it would be performing better
            #so we should generate a new img with 
            d_loss = compute_discriminator_loss(real_image, seg_map)            
            #print(generator_loss, d_loss, rdl+fdl)
            d_loss.backward()
            disc_optimizer.step()

        #if (epoch) % 5 == 0:
        #Save generator and disrciminator loss in file
        with open('models/loss/'+filename+'.txt', 'a') as f:
            f.write('%.4f \t %.4f \n'%(generator_loss.item(), d_loss.item()))

        validateEpoch(filename, epoch)

        #Save model
        saveModels(filename, epoch)
        print('last save:', datetime.now())


        if (epoch % 5) == 0:
            saveModels(filename, epoch, '_' + str(epoch))
        # print("saving at epoch:", epoch)
        # #print("g-loss:", generator_loss.item(), "d-loss:", d_loss.item())
        # #print('g-loss: %.4f \t d-loss: %.4f \n'%(generator_loss.item(), d_loss.item()))
        # torch.save(encoder.state_dict(), 'models/encoder' + filename + '.pth')
        # torch.save(generator.state_dict(), 'models/generator'  + filename + '.pth')
        # torch.save(discriminator.state_dict(), 'models/discriminator' + filename + '.pth')


def testGauGAN(dataloader, encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator):
    encoder.eval()
    generator.eval()
    discriminator.eval()
    import os
    spadepaths = os.listdir('dataset/COCO/SPADE')
    paths = [os.path.join('dataset/COCO/SPADE', x) for x in spadepaths]

    with torch.no_grad():
        for batch_index, data in enumerate(dataloader):
                seg_map, real_image, label = preprocess_input(data)

                fake_image, _ = generate_fake(real_image, seg_map)
                #spadeimg = Image.open(paths[batch_index])
                #print("gen loss", generator_loss, "disc loss", d_loss)
                utils.plotImages(fake_image[0].cpu(),real_image[0].cpu(), label[0].cpu())
                #plotImage(fake_image[0].detach().cpu())
                #plotImage(real_image[0].detach().cpu())
                #plotLabel(seg_map[0].detach().cpu())


def loadModel(encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator = None, filename = '', optional='', _device = 'cpu'):
    encoder.load_state_dict(torch.load('models/' + filename + '/encoder' + optional + '.pth', map_location=_device))
    generator.load_state_dict(torch.load('models/' + filename + '/generator' + optional + '.pth', map_location=_device))
    if discriminator:
        discriminator.load_state_dict(torch.load('models/' + filename + '/discriminator' + optional + '.pth', map_location=_device))

if __name__=="__main__":

    transform_train = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_label_train = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


    # training_data = dataset.CamVidDataset('dataset/CamVid/train_img','dataset/CamVid/train_label', transform_train, transform_label_train )
    # test_data = dataset.CamVidDataset('dataset/CamVid/test_img','dataset/CamVid/test_label', transform_train, transform_label_train )
    
    training_data = dataset.CamVidDataset('dataset/COCO/test_img','dataset/COCO/test_label', transform_train, transform_label_train)
    test_data = dataset.CamVidDataset('dataset/COCO/test3','dataset/COCO/test3l', transform_train, transform_label_train )

    validation_data = Subset(training_data, [0, 2, 5])

    train_loader = DataLoader(training_data, batch_size=4, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=0)

    #print(torch.cuda.is_available())

    # img = cv2.imread('dataset/test.png')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen_loss = loss.Gen_loss().to(device)
    kld_loss = loss.KLD_loss().to(device)
    vgg_loss = loss.VGGLoss(0).to(device)
    feat_loss = loss.FeatureLossDisc().to(device)
    disc_loss = loss.HingeLoss().to(device)
    hinge_loss = loss.Hinge().to(device)

    encoder = models.Encoder()
    generator = models.Generator()
    discriminator = models.Discriminator()
    #print(generator)
    #print(discriminator)
    #print(encoder)

    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()
        discriminator.cuda()

    #lr = 0.0001, 0.0004 
    #beta1 = 0, beta2 = 0.999
    genParams = list(generator.parameters())
    genParams += encoder.parameters()
    gen_optimizer = torch.optim.Adam(genParams, lr=0.10e-4, betas=(0, 0.999)) #TODO lr=0.0002, beta1 = 0 beta = 0.9 but in paper its actually different so f this
    disc_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=4e-4, betas=(0, 0.999))

    #SETTINGS
    filename = '_coco_20_'
    version = ''
    nr_epochs = 100
    start_epoch = 165
    train = False
    load_model = True

    if load_model:
        loadModel(encoder, generator, discriminator, filename, optional=version, _device=device)

    #TODO initalize weights - uses default initialization rn 

    if train:
        trainGauGAN(train_loader, encoder, generator, discriminator, nr_epochs, filename, start_epoch)
    else:
        testGauGAN(test_loader, encoder, generator, discriminator)

    utils.plotLoss(filename)