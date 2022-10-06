import torch
from tqdm import tqdm
import numpy as np

import models
import loss
import settings
#dataset
import dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

def plotImages(fake, real, label):
    import matplotlib.pyplot as plt
    fakeimg = (np.asarray(fake).transpose(1,2,0) + 1) / 2.0
    realimg = (np.asarray(real).transpose(1,2,0) + 1) / 2.0
    labelimg = np.asarray(label).transpose(1,2,0)

    fig, plot = plt.subplots(1,3)
    plot[0].imshow(fakeimg)
    plot[0].set_title("fake")
    plot[1].imshow(realimg)
    plot[1].set_title("real")
    plot[2].imshow(labelimg)
    plot[2].set_title("seg map")

    plt.show()


def plotImage(img):
    import matplotlib.pyplot as plt
    print(np.shape(img))
    image = np.asarray(img).transpose(1,2,0)
    print(np.min(image))

    print(np.shape(image))
    plt.imshow(image)
    plt.show()

def plotLabel(label):
    import matplotlib.pyplot as plt
    #print(np.shape(image))
    image = np.asarray(label).transpose(1,2,0)
    plt.imshow(image)
    plt.show()

def preprocess_input(data):
    if torch.cuda.is_available():
        data['image'] = data['image'].cuda()
        data['label'] = data['label'].cuda()

    #one hot label map (segmentation map / semantics map)
    label_map = data['label']
    #print(np.shape(label_map))
    bs, _, h, w = label_map.size() #there should be a batch size?
    nc = 32  # number of classes
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_() if torch.cuda.is_available() else torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    return input_semantics, data['image'], label_map

def testGauGAN(dataloader, encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator):
    with torch.no_grad():
        for batch_index, data in enumerate(dataloader):
                seg_map, real_image, label = preprocess_input(data)

                #test generator
                #gen_optimizer.zero_grad()
                mu, var = encoder(real_image)
                latent_vec = encoder.compute_latent_vec(mu, var)
                fake_image = generator(latent_vec=latent_vec, segmap=seg_map)

                fake_disc_output = discriminator(fake_image, seg_map)
                real_disc_output = discriminator(real_image, seg_map)

                gen_loss_batch = gen_loss(fake_disc_output[-1])
                kld_loss_batch = kld_loss(mu, var)
                vgg_loss_batch = vgg_loss(real_image, fake_image)
                feat_loss_batch = feat_loss(real_disc_output, fake_disc_output)

                generator_loss = gen_loss_batch + \
                    kld_loss_batch * 0.1 + \
                    vgg_loss_batch * 0.1 + \
                    feat_loss_batch * 10

                #generator_loss.backward()
                #gen_optimizer.step()

                #discriminator
                #disc_optimizer.zero_grad()
                real_disc_output_d = discriminator(real_image.detach(), seg_map)
                fake_disc_output_d = discriminator(fake_image.detach(), seg_map)
                real_disc_loss = disc_loss(real_disc_output_d[-1], True) * 0.5
                fake_disc_loss = disc_loss(fake_disc_output_d[-1], False) * 0.5
                d_loss = real_disc_loss + fake_disc_loss
                #d_loss.backward()
                #disc_optimizer.step()
                print("gen loss", generator_loss, "disc loss", d_loss)
                plotImages(fake_image[0].detach().cpu(),real_image[0].detach().cpu(), label[0].detach().cpu())
                #plotImage(fake_image[0].detach().cpu())
                #plotImage(real_image[0].detach().cpu())
                #plotLabel(seg_map[0].detach().cpu())

def trainGauGAN(dataloader, encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator, nrEpochs, filename):
    for epoch in tqdm(range(nrEpochs + 1)):
        for batch_index, data in enumerate(dataloader):
            #inputs, targets = inputs.cuda(), targets.cuda()
            # print("img:",np.shape(data["image"]))
            # print("segmap",np.shape(data["seg_map"]))
            # print("label",np.shape(data["label"]))
            seg_map, real_image, _ = preprocess_input(data)

            #train generator
            gen_optimizer.zero_grad()
            mu, var = encoder(real_image)
            latent_vec = encoder.compute_latent_vec(mu, var)
            fake_image = generator(latent_vec=latent_vec, segmap=seg_map)


            fake_disc_output = discriminator(fake_image, seg_map)
            real_disc_output = discriminator(real_image, seg_map)

            gen_loss_batch = gen_loss(fake_disc_output[-1])
            kld_loss_batch = kld_loss(mu, var)
            vgg_loss_batch = vgg_loss(real_image, fake_image)
            feat_loss_batch = feat_loss(real_disc_output, fake_disc_output)

            generator_loss = gen_loss_batch + \
                kld_loss_batch * 0.1 + \
                vgg_loss_batch * 0.1 + \
                feat_loss_batch * 10

            generator_loss.backward()
            gen_optimizer.step()

            #train discriminator
            disc_optimizer.zero_grad()
            real_disc_output_d = discriminator(real_image.detach(), seg_map)
            fake_disc_output_d = discriminator(fake_image.detach(), seg_map)
            real_disc_loss = disc_loss(real_disc_output_d[-1], True) * 0.5
            fake_disc_loss = disc_loss(fake_disc_output_d[-1], False) * 0.5
            d_loss = real_disc_loss + fake_disc_loss
            d_loss.backward()
            disc_optimizer.step()

        # fkimg = fake_image[0].detach().cpu().permute(2,1,0)         
        # plt.imshow(fkimg)
        # plt.show()
        if (epoch) % 5 == 0:
            print("Saving at epoch:", epoch)
            print("g-loss:", generator_loss, "d-loss:", d_loss)
            torch.save(encoder.state_dict(), 'models/encoder' + filename + '.pth')
            torch.save(generator.state_dict(), 'models/generator'  + filename + '.pth')
            torch.save(discriminator.state_dict(), 'models/discriminator' + filename + '.pth')

def loadModel(encoder: models.Encoder, generator: models.Generator, discriminator: models.Discriminator = None, filename = '', _device = 'cpu'):
    encoder.load_state_dict(torch.load('models/encoder' + filename + '.pth', map_location=_device))
    generator.load_state_dict(torch.load('models/generator' + filename + '.pth', map_location=_device))
    if discriminator:
        discriminator.load_state_dict(torch.load('models/discriminator' + filename + '.pth', map_location=_device))
    # encoder.load_state_dict(torch.load('models/encoder.pth', map_location=device))
    # generator.load_state_dict(torch.load('models/generator.pth', map_location=device))
    # discriminator.load_state_dict(torch.load('models/discriminator.pth', map_location=device))


    # encoder.load_state_dict(torch.load('models/encoder_no_normalize.pth'))
    # generator.load_state_dict(torch.load('models/generator_no_normalize.pth'))
    # discriminator.load_state_dict(torch.load('models/discriminator_no_normalize.pth'))

    # encoder.load_state_dict(torch.load('models/encoder_no_normalize_200.pth'))
    # generator.load_state_dict(torch.load('models/generator_no_normalize_200.pth'))
    # discriminator.load_state_dict(torch.load('models/discriminator_no_normalize_200.pth'))

    # encoder.load_state_dict(torch.load('models/encoder_normalize_50pth'))
    # generator.load_state_dict(torch.load('models/generator_normalize_50.pth'))
    # discriminator.load_state_dict(torch.load('models/discriminator_normalize_50.pth'))

if __name__=="__main__":

    transform_train = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_label_train = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])


    training_data = dataset.CamVidDataset('dataset/CamVid/train_img','dataset/CamVid/train_label', transform_train, transform_label_train )
    test_data = dataset.CamVidDataset('dataset/CamVid/test_img','dataset/CamVid/test_label', transform_train, transform_label_train )

    train_loader = DataLoader(training_data, batch_size=5, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=False, num_workers=1)

    #print(torch.cuda.is_available())

    # img = cv2.imread('dataset/test.png')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen_loss = loss.Gen_loss().to(device)
    kld_loss = loss.KLD_loss().to(device)
    vgg_loss = loss.VGGLoss(0).to(device)
    feat_loss = loss.FeatureLossDisc().to(device)
    disc_loss = loss.HingeLoss().to(device)

    encoder = models.Encoder()
    generator = models.Generator()
    discriminator = models.Discriminator()

    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()
        discriminator.cuda()

    #lr = 0.0001, 0.0004 
    #beta1 = 0, beta2 = 0.999
    genParams = list(generator.parameters())
    genParams += encoder.parameters()
    gen_optimizer = torch.optim.Adam(genParams, lr=0.1e-4, betas=(0, 0.999))
    disc_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=4e-4, betas=(0, 0.999))


    loadModel(encoder, generator, discriminator, '_normalize_50')
    # loadModel(encoder, generator, discriminator, '_no_normalize_200')
    #loadModel(encoder, generator, discriminator, '_no_normalize')
    #loadModel(encoder, generator, discriminator, '_normalize')
    

    #TODO initalize weights - uses default initialization rn 

    testGauGAN(test_loader, encoder, generator, discriminator)
    #trainGauGAN(train_loader, encoder, generator, discriminator, 50, '_normalize_50')
    #import time


