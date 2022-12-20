# GauGAN - Synthesize Photorealistic Images from Semantic Doodles

Implementation of NVIDIA's generative SPADE network in PyTorch. Based on the original paper of Park et.al.

GauGAN is a generative network with a special normalization method SPADE, the generator takes as input a segmentation mask and a corresponding real image or noise vector in a high dimensional space and synthesize a photorealistic image as output. Depending on the dataset used during training it is capable of multi-modal image synthesis, which means that it can generate images in various different styles using the same input segmentation mask. In my implementation I trained the network on parts of COCO-Stuff dataset for about 160 epochs on my home computer which limit the rendering capabilities of the network. Beside the network I also created an interactive app hosted on streamlit, where you can draw your own doodles and turn them into semi-photorealistic images. 

## Interactive demo
https://user-images.githubusercontent.com/48772850/208649964-d1e65708-b5a4-4635-acf4-3ef40df68a5b.mov

You can test the demo yourself, note that the generation of images is quite slow without GPU acceleration. 
### How to:
1. Select one of the many materials in the panel to the left.
2. Draw your doodles. 
3. You can move objects by selecting 'transform' under drawing tool.
4. Images are automatically generated, this can be toggeled on/off by checking 'Update image in realtime'. 
5. Style image is supposed to be used as a multi-modal synthesis but does not work all to well with COCO-Stuff dataset.




Demo: https://engbergandreas-gaugan.streamlit.app/




## Sources
1. Semantic Image Synthesis with Spatially-Adaptive Normalization: https://arxiv.org/pdf/1903.07291.pdf 
2. Official implementation: https://github.com/NVlabs/SPADE
3. COCO-Stuff dataset: https://github.com/nightrome/cocostuff
4. Streamlit drawable canvas: https://github.com/andfanilo/streamlit-drawable-canvas

## Results
![Synthesized result 1](https://github.com/engbergandreas/GauGAN/blob/main/app/result_12_.png)
![Synthesized result 2](https://github.com/engbergandreas/GauGAN/blob/main/app/result_19_.png)
![Synthesized result 3](https://github.com/engbergandreas/GauGAN/blob/main/app/result_27_.png)
![Synthesized result 4](https://github.com/engbergandreas/GauGAN/blob/main/app/result_32_.png)
![Synthesized result 5](https://github.com/engbergandreas/GauGAN/blob/main/app/result_35_.png)
![Synthesized result 6](https://github.com/engbergandreas/GauGAN/blob/main/app/result_39_.png)
