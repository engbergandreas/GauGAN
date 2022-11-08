import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from random import random

def plotLoss(filename):
    with open('models/loss/' + filename + '.txt') as f:
        data = f.read().split('\n')[:-1]
        data = [d.split('\t') for d in data]
        g_loss = [float(d[0]) for d in data]
        d_loss = [float(d[1]) for d in data]

        g_loss = np.array(g_loss)
        d_loss = np.array(d_loss)
        epochs = range(0, len(g_loss))
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(filename + ' loss')
        plt.plot(epochs, g_loss, 'b', epochs, d_loss, 'r')
        plt.legend(['g_loss', 'd_loss'])
        plt.grid(True)
        plt.show()

def plotImages(fake, real, label, spade=None, number=None):
    fakeimg = (np.asarray(fake).transpose(1,2,0) + 1) / 2.0
    realimg = (np.asarray(real).transpose(1,2,0) + 1) / 2.0
    labelimg = np.asarray(label).transpose(1,2,0)

    if(spade):
        fig, plot = plt.subplots(1,4)
        plot[3].axis('off')
        plot[3].set_title('Nvidia')
        plot[3].imshow(spade)
    else:
        fig, plot = plt.subplots(1,3)

    plot[0].axis('off')
    plot[1].axis('off')
    plot[2].axis('off')

    plot[0].set_title("Input label")
    plot[1].set_title("Real image")
    plot[2].set_title("Synthesized")
    
    plot[0].imshow(labelimg)
    plot[1].imshow(realimg)
    plot[2].imshow(fakeimg)

    if number:
        plt.savefig('result_' + str(number) + '_.png', bbox_inches='tight')
    plt.show()
    
def saveValidationImage(fake, real, label, path, epoch):
    Path('results/' + path).mkdir(parents=True, exist_ok=True)
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
    plt.savefig('results/' + path + '/validation_' + str(epoch) + '_.png', bbox_inches='tight')
    plt.close()


#create dictionary of 255 unique random colors
def createRandomColors():
    with open('colormapping.txt','a') as file:
        for i in range(256):
            r = int(random() * 255)
            g = int(random() * 255)
            b = int(random() * 255)
            file.write(str(r) + ' ' + str(g) + ' ' + str(b) + '\n')
