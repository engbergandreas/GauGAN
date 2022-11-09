import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from models import Encoder, Generator
import numpy as np
from torchvision.transforms import transforms
import torch
import settings

#utils.createRandomColors() 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Utility helper functions to transform colors 
def rgb_to_hex(r,g,b):
    return '#%02x%02x%02x' % (r,g,b)

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def int_to_rgb(i):
    return colormapping[i]

def rgb_to_int(r,g,b):
    for i in range(len(colormapping)):
        if (int_to_rgb(i) == (r,g,b)).all():
            return i

    return -1

#Create a mapping between integer and a random generated color
@st.cache()
def getColorMapping():  
    colormaps = {}
    with open('app/colormapping.txt', "r") as file:
        colors = file.read().split("\n")[:-1]
        # colors = [x[-1] for x in pair]
        colors = [x.split(' ') for x in colors]

        for i, color in enumerate(colors):
            colors[i] = [int(x) for x in color]

        colors = np.array(colors)

        for i, color in enumerate(colors):
            colormaps[i] = color

    return colormaps

#retrieve all colors (given in hex) and label names used in the segmentation map
# def getSegmentationColors(path):
#     with open(path, "r") as file:
#         label_colors = file.read().split("\n")[:-1]
#         label_colors = [x.split(" ") for x in label_colors]
#         label_names = [x[-1] for x in label_colors] #get label names
#         st.write(label_names)
#         colors = [x[0] for x in label_colors] #get colors
#         colors = [x.split(" ") for x in colors] #split colors into rgb values
#         for i,color in enumerate(colors): 
#             colors[i] = [int(x) for x in color]
        
#         colors = np.array(colors) 
        
#         materials = {}
#         for index in range(len(colors)): 
#             materials[label_names[index]] = rgb_to_hex(colors[index][0], colors[index][1], colors[index][2])
#         return materials


#Create dictionary of name and corresponding color in hex from file at path
@st.cache()
def getSegmentationColors(path):
    with open(path, "r") as file:
        label_colors = file.read().split("\n")[:-1]
        label_colors = [x.split(" ", 1) for x in label_colors]
        label_names = [x[-1] for x in label_colors] #get label names
        colors = [x[0] for x in label_colors] #get colors
        #x - 1 beacuse gray scale values are shifted 1 step refer: COCO-Stuff documentation
        colors = [int_to_rgb(int(x) - 1) for x in colors] #split colors into rgb values

        colors = np.array(colors)
        
        #Store name and corresponding color as hex dict
        materials = {}
        for index in range(len(colors)): 
            materials[label_names[index]] = rgb_to_hex(colors[index][0], colors[index][1], colors[index][2])

        return materials

def generateFakeImageFromCanvas(segmap, stylepath): 
    style_image = Image.open(stylepath)
    style_image_tensor = transform_image(style_image).unsqueeze(0).to(device) #create a batch of 1 image
    encoder.eval()
    generator.eval()
    
    with torch.no_grad():
        mu, var = encoder(style_image_tensor)
        z = encoder.compute_latent_vec(mu, var)
        fake_image = generator(latent_vec=z, segmap=segmap)

    return fake_image

def createBWLabel(img):
    img = img[:,:,:3] #ignore alpha channel
    #Convert colors in segmentation map to RGB
    colors = np.array([hex_to_rgb(materials[material]) for material in materials])
    h,w, _ = img.shape
    modified_annotation = np.zeros((h,w))

    for i,color in enumerate(colors):
        value = rgb_to_int(color[0], color[1], color[2])
        color = color.reshape(1,1,-1)
        mask = (color == img)
        
        r = mask[:,:,0]
        g = mask[:,:,1]
        b = mask[:,:,2]
        
        mask = np.logical_and(r,g)
        mask = np.logical_and(mask, b).astype(np.int64)
        mask *= value
        
        modified_annotation += mask
    return modified_annotation

@st.cache(hash_funcs={"UnhashableClass": lambda _: None})
def load_model():
    encoder = Encoder()
    generator = Generator()

    if torch.cuda.is_available():
        encoder.cuda()
        generator.cuda()

    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    generator.load_state_dict(torch.load('generator.pth', map_location=device))

    return encoder, generator   


colormapping = getColorMapping()
materials = getSegmentationColors('app/label_colors_shorter.txt')
encoder, generator = load_model()

transform_image = transforms.Compose([
        transforms.Resize((settings.IMG_HEIHGT,settings.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
transform_label = transforms.Compose([
    transforms.ToTensor()
])

scale = 2
transform_upscale = transforms.Resize((settings.IMG_HEIHGT * scale,settings.IMG_WIDTH * scale))

#Style transfer images
styles = {
    'Baseball': 'app/000000153797.jpg',
    'Giraffe': 'app/000000000025.jpg',
    'Snow': 'app/000000080273.jpg',
    'Water': 'app/000000082715.jpg'
}

#Sidebar settings
material = st.sidebar.radio('material', materials.keys(), horizontal=True)
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "transform")
    )
stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 25)
realtime_update = st.sidebar.checkbox("Update image in realtime", True)
upscale = st.sidebar.checkbox("Upscale", False)

c1,c2 = st.sidebar.columns(2)
with c1:
    style = st.radio('Style', styles.keys())
with c2:
    st.image(list(styles.values()))

stroke_color = materials[material]

st.header("Draw semantic doodles and watch them turn into 'realistic' images!")
# Create a canvas component
c1, c2 = st.columns(2)
with c1: 
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        update_streamlit=realtime_update,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        key="canvas"
    )
with c2:
    if canvas_result.image_data is not None:
        stylePath = styles[style]
        #Convert canvas to bw segmentation map
        bwLabel = createBWLabel(canvas_result.image_data)
        label_tensor = transform_label(bwLabel).long().unsqueeze(0)
        if torch.cuda.is_available():
            label_tensor = label_tensor.cuda()

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
        img = (np.asarray(fakeimg.cpu()).transpose(1,2,0) + 1) / 2.0
        #Display synthesized image
        st.image(img)


st.header("Example images generated by the network.")
st.image('app/result_12_.png')
st.image('app/result_19_.png')
st.image('app/result_27_.png')
st.image('app/result_32_.png')
st.image('app/result_35_.png')
st.image('app/result_39_.png')

st.write("Project by Andreas Engberg - src code: https://github.com/engbergandreas/GauGAN")