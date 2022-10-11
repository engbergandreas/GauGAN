import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def plotImages(fake, real, label):
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