import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from models import Encoder, Generator
import numpy as np
from torchvision.transforms import transforms
import torch
import settings
from main import loadModel

def rgb_to_hex(r,g,b):
    return '#%02x%02x%02x' % (r,g,b)

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

#retrieve all colors (given in hex) and label names used in the segmentation map
def getSegmentationColors(path):
    with open(path, "r") as file:
        label_colors = file.read().split("\n")[:-1]
        label_colors = [x.split("\t") for x in label_colors]
        label_names = [x[-1] for x in label_colors] #get label names
        colors = [x[0] for x in label_colors] #get colors
        colors = [x.split(" ") for x in colors] #split colors into rgb values
        for i,color in enumerate(colors): 
            colors[i] = [int(x) for x in color]
        
        colors = np.array(colors) 
        
        materials = {}
        for index in range(len(colors)): 
            materials[label_names[index]] = rgb_to_hex(colors[index][0], colors[index][1], colors[index][2])
        return materials

def generateFakeImageFromCanvas(segmap, stylepath): 
    style_image = Image.open(stylepath)
    style_image_tensor = transform_train(style_image).unsqueeze(0).to(device) #create a batch of 1 image

    with torch.no_grad():
        mu, var = encoder(style_image_tensor)
        z = encoder.compute_latent_vec(mu, var)
        fake_image = generator(latent_vec=z, segmap=segmap)

    #TODO rescale fake image to compensate mean and std? 
    return fake_image

def createBWLabel(img):
    img = img[:,:,:3] #ignore alpha channel
    #Convert colors in segmentation map to RGB
    colors = np.array([hex_to_rgb(materials[material]) for material in materials])
    h,w, _ = img.shape
    modified_annotation = np.zeros((h,w))

    for i,color in enumerate(colors):
        color = color.reshape(1,1,-1)

        mask = (color == img)
        
        r = mask[:,:,0]
        g = mask[:,:,1]
        b = mask[:,:,2]
        
        mask = np.logical_and(r,g)
        mask = np.logical_and(mask, b).astype(np.int64)
        mask *= i
        
        modified_annotation += mask
    
    return modified_annotation
    
materials = getSegmentationColors("dataset/CamVid/label_colors.txt")
encoder = Encoder()
generator = Generator()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loadModel(encoder, generator, filename='_no_normalize_200',_device=device)

transform_train = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
transform_label = transforms.Compose([
    transforms.ToTensor()
])
scale = 2
transform_upscale = transforms.Resize((settings.IMG_HEIHGT * scale,settings.IMG_WIDTH * scale))

styles = {
    'style1': 'dataset/FlickrLandScapes/test.jpg', 
    'style2': 'dataset/FlickrLandScapes/images/00000005_(4).jpg',
    'style3': 'dataset/CamVid/test_img/0001TP_006690.png'
}

#Sidebar settings
material = st.sidebar.radio('material', materials.keys(), horizontal=True)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 30)
#stroke_color = st.sidebar.color_picker("Stroke color hex: ")
#bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
upscale = st.sidebar.checkbox("Upscale", False)

c1,c2 = st.sidebar.columns(2)
with c1:
    style = st.radio('Style', styles.keys())
with c2:
    st.image(list(styles.values()))

stroke_color = materials[material]

# Create a canvas component
c1, c2 = st.columns(2)
with c1: 
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        #background_color=bg_color,
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas"
    )
# with c2:
if canvas_result.image_data is not None:
    stylePath = styles[style]
    #Convert canvas to bw segmentation map
    bwLabel = createBWLabel(canvas_result.image_data)
    #TODO maybe move to correct device
    label_tensor = transform_label(bwLabel).long().unsqueeze(0)
    #Create one hot label map
    bs, _, h, w = label_tensor.size()
    nc = settings.NUM_CLASSES
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_() if torch.cuda.is_available() else torch.FloatTensor(bs, nc, h, w).zero_()
    segmap = input_label.scatter_(1, label_tensor, 1.0)
    #Generate image from canvas segmentation map and style image
    if upscale:
        fakeimg = generateFakeImageFromCanvas(segmap, stylePath)
        fakeimg = transform_upscale(fakeimg)[0]
    else:
        fakeimg = generateFakeImageFromCanvas(segmap, stylePath)[0]
    #Compensate for mean and std 
    img = (np.asarray(fakeimg).transpose(1,2,0) + 1) / 2.0
    st.image(img)