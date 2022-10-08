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
    nc = settings.NUM_CLASSES  # number of classes
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

            #print(np.shape(seg_map))
            #print(np.shape(real_image))

            #train generator
            gen_optimizer.zero_grad()
            mu, var = encoder(real_image)
            latent_vec = encoder.compute_latent_vec(mu, var)
            fake_image = generator(latent_vec=latent_vec, segmap=seg_map)


            #TODO feed real and fake image at the same time to the discriminator, check pixp2pix_model.py line 209
                #the first half of the output is for the fake pred, the second half is the real output
            fake_disc_output = discriminator(fake_image, seg_map)
            real_disc_output = discriminator(real_image, seg_map)

            #print(np.shape(fake_disc_output))
            #print(fake_disc_output)

            #print(fake_disc_output[-1])

            #TODO check gen_loss => hinge loss
            #compute hinge loss as: 
            #input=fake_disc_output
            #minval = torch.min(input - 1, self.get_zero_tensor(input)) #function call can be found in loss.py line 51
            #loss = -torch.mean(minval)

            #if target is not real compute as torch.min(-input - 1, ...) ...
            gen_loss_batch = gen_loss(fake_disc_output[-1]) #TODO essentially this should be hinge

            kld_loss_batch = kld_loss(mu, var)  #ok
            vgg_loss_batch = vgg_loss(real_image, fake_image)#OK

            feat_loss_batch = feat_loss(real_disc_output, fake_disc_output) #TODO check if last item should be included in the feat losscompare pix2pix_model.py line 151

            #kld loss lambda = 0.05
            #feat loss lampa = 10 - OK
            #vgg loss lambda = 10
            #TODO sum and mean? 
            generator_loss = gen_loss_batch + \
                kld_loss_batch * 0.1 + \
                vgg_loss_batch * 0.1 + \
                feat_loss_batch * 10

            generator_loss.backward()
            gen_optimizer.step()

            #train discriminator
            disc_optimizer.zero_grad()
            #TODO check if we should generate new image and detach it from gpu => set requires_grad_()
            #A reason why we should generate a new image is that we have trained the generator a step so it would be performing better
            #so we should generate a new img with 
            #with torch.no_grad(): 
                #fake_img = generate_fake(input_seg, real_img)
                #fake_img.detach()
                #fake_img.requires_grad_()
            #TODO compute prediction in one batch as described above
            real_disc_output_d = discriminator(real_image.detach(), seg_map)
            fake_disc_output_d = discriminator(fake_image.detach(), seg_map)
            #TODO compare against hinge loss computed as described above 
            real_disc_loss = disc_loss(real_disc_output_d[-1], True) * 0.5
            fake_disc_loss = disc_loss(fake_disc_output_d[-1], False) * 0.5
            d_loss = real_disc_loss + fake_disc_loss
            d_loss.backward()
            disc_optimizer.step()

        # fkimg = fake_image[0].detach().cpu().permute(2,1,0)         
        # plt.imshow(fkimg)
        # plt.show()
        #if (epoch) % 5 == 0:
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
    
    training_data = dataset.CamVidDataset('dataset/COCO/test_img','dataset/COCO/test_label', transform_train, transform_label_train )
    test_data = dataset.CamVidDataset('dataset/COCO/test_img','dataset/COCO/test_label', transform_train, transform_label_train )

    train_loader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

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
    gen_optimizer = torch.optim.Adam(genParams, lr=0.1e-4, betas=(0, 0.999)) #TODO lr=0.0002, beta1 = 0 beta = 0.9 but in paper its actually different so f this
    disc_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=4e-4, betas=(0, 0.999))


    #loadModel(encoder, generator, discriminator, '_normalize_50')
    #loadModel(encoder, generator, discriminator, '_no_normalize_200')
    #loadModel(encoder, generator, discriminator, '_no_normalize')
    #loadModel(encoder, generator, discriminator, '_normalize')
    loadModel(encoder, generator, discriminator, '_coco_50')

    #TODO initalize weights - uses default initialization rn 

    testGauGAN(test_loader, encoder, generator, discriminator)
    #trainGauGAN(train_loader, encoder, generator, discriminator, 50, '_coco_test')
    #import time


