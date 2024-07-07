---
title: 'RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths'
date: 2024-06-01
permalink: /posts/2024/05/raphael-post
categories: blog
tags:
  - RAPHAEL
  - Diffusion Model
  - Mixture of Experts
  - Computer Vision
  - Generative Model
---

In recent years Generative Artificial Intelligence has gained a lot of popularity. More and more people use it in their daily and professional life. Diffusion Models became popular with models like Stable Diffusion and DALL-E which showed the public what Diffusion Models are able to do. \
In this blog post, I aim to introduce and explain a new model, RAPHAEL, which outperforms models like Stable Diffusion and focuses on accurately displaying text in the generated images [[1]](#1).

Why Diffusion Models?
======
Have you ever taken a picture of something and later wanted to have more background or just a larger picture? Diffusion Models can be used to solve this problem, for example Adobe has introduced a Diffusion Model called "Adobe Firefly 3 Model" which can expand the image and even add new objects or remove objects from the picture. 

<!--- ![image](https://github.com/Florian-de/Florian-de.github.io/assets/64322175/21b10292-f31f-403d-8ef0-855d97dcfbe1) --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/de53ddee-3e90-4cae-97ee-e27ac328fca9">

Image from Adobe [[2]](#2)

Such technology can not only be used as a useful gadged for our daily lifes but also for professional use, for example for editing images for a commercial. \
It brings even more advantages in the use for science. For example in Biology, scientists use Diffusion Models to improve the image quality of their microscopes, which helps them to gain new insights and overall accelerates their work. 

A more radical application for Diffusion Models is in the field of chemistry where they are used to generate new molecules for a specific purpose, which vastly accelerates the process of finding the right molecule for a given problem, for example in drug discovery. One example of this is DiffSBDD, a Diffusion Model designed for Structure-based Drug Design, generating new molecules conditional on a protein pocket structure.

<!--- ![image](https://github.com/Florian-de/Florian-de.github.io/assets/64322175/d07a01e7-dc80-4a81-a979-bb9537b01c5b) --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/da79eb30-c39d-4977-bb6e-d3821ba0a49d">

Image from Charlie Harris [[3]](#3)

Recent models like Stable Diffusion or DALL-E have shown great success but often lack the ability to accurately display text in the generated images like shown below:

<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/df96156c-564a-4f49-afb0-7e6daae53899">

RAPHAEL aims at accurately displaying text in generated images and overall improving image quality.

What is a Diffusion Model?
======
Diffusion Models have the goal to generate images by removing noise from pure noise.  
<!--- ![image](https://github.com/Florian-de/Florian-de.github.io/assets/64322175/ca2a8f9b-b597-473c-af9d-9263b7f4cb41) --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/ca2a8f9b-b597-473c-af9d-9263b7f4cb41">

Image from Jaskaran Bhatia [[4]](#4)

As shown in the picture above, Diffusion Models constist of two parts, the forward diffusion process and the reverse diffusion process. 

Forward diffusion process
------
In the forward diffusion process the model takes an image as input and adds random noise to the image step by step starting with the input image $x_0$ and ending in pure noise $x_t$. \
In each step the process is defined as $q(x_t|x_{t−1}) := \mathcal{N}(x_t; \sqrt{1 − \beta_t}x_{t−1},\beta_tI)$ where $q$ is the process, $x_t$ the output of the current step, $x_{t-1}$ the output of the previous step and $\mathcal{N}$ the normal distribution with $\sqrt{1 − β_t}x_{t−1}$ as the mean $\mu$ and $\beta_tI$ as the variance $\sigma^2$. \
During this process $\beta_t$ is controlled by a schedule and has values in the range of 0 and 1. Such a schedule could be as simple as a linear schedule which would increase $\beta_t$ by a constant size each step, but in practice more advanced schedules are used. \
For efficient computation the entire process from $x_0$ to $x_t$ can be calculated using a closed form $q(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar\alpha_t}x_0, (1 − \bar\alpha_t)I)$ where $\alpha_t := 1 − β_t$ and $\bar\alpha_t := \Pi^{t}_{s=1} \alpha_s$ [[5]](#5).

Reverse diffusion process
------
In the reverse diffusion process the model tries to predict the total noise for each timestep starting with the pure noise $x_t$ and ending in a denoised image $x_0$. \
The process for each step can be described by the formula $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha}}\epsilon_\theta(x_t, t)) + \sqrt{\beta_t}\epsilon$ where $\epsilon_\theta(x_t, t)$ is output of the prediction model. \
For the prediction of the noise models typically use a modified UNet Neural Network Architecture. 
<!--- <img width="920" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/145c6a61-f5cf-447d-b5d1-e35c4f31b526"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/145c6a61-f5cf-447d-b5d1-e35c4f31b526">

Image from Kemal Erdem [[6]](#6)

An example for such an architecture for a text-conditional model is shown above. \
It takes the total noise an text as input. \
The text is embedded by an encoder network which takes the text as input and has vectors as output. Each vector represents a single text token. Like shown below the embedding vectors transport semantics, for example the difference between the embedding of man and woman is similiar to the difference of uncle and aunt. 

<!--- <img width="442" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/91b74c71-9a42-4c27-8be5-f26389a05fda"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/66038dff-15fc-417e-8516-1b5d42f77937">

Image from 3Blue1Brown [[7]](#7)

The network starts and ends with a pink rectangle which represents a ResNet block which takes the data from the previous layer as input. It is used to extract features from the image.\
The blue rectangles represent Downsample Blocks which takes data from the previous layer and data about the timestamp and the text embeddings as the two inputs. It is used to downsample the data from the previous layer to the size of the layer. \
The grey arrows represent skipping connections between the Downsampling and the Upsampling Blocks to prevent loss of information. \
The green rectangles represent Upsample Blocks which takes data from the previous layer, data about the timestamp and the text embeddings and data from the skipping connection as the three inputs. It is used to predict the noise. \
The orange rectangles represent Self-Attention Blocks which takes the data from the previous Downsample/Upsample Block as input. It is used to learn the connections between the different parts of the image. 

What are Mixture of Experts?
======
In general Mixture of Experts (MoE) is the method of using multiple expert models instead of using just a single big model and therefore dividing a problem. \
A MoE Layer consists of a gating function and many experts. The experts share the same architectur and are trained by the same algorithm. The gating function assigns input data to the best experts. To speed up the inference time a sparse gating function is used, which assigns the input only to the top-K experts. [[8]](#8) When we use a sparse gating function we speak of Space MoE Layers. \
A MoE Layer takes the data from the previous layer as input data and outputs sum kind of weighted average of the outputs of the experts.[[9]](#9)

<!---  ![image](https://github.com/Florian-de/Florian-de.github.io/assets/64322175/ce125e88-cd7f-4d73-9bed-7e92494039e0) --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/ce125e88-cd7f-4d73-9bed-7e92494039e0">

Image from Hugging Face [[9]](#9)

In Transformer blocks the MoE layer is normally implemented by replacing the MLP/FFN after the Attention Layer as shown in the image above. \
The use of MoE can provide benefits like overall better performance, efficient pretraining or faster inference compared to the use of a single MLP/FFN.

Mapping to a Deep Learning problem
======

Input 
------
The input for the RAPHAEL model consists of images, complete noise and text. \
The images are used in the forward diffusion process as explained in the section for Diffusion Models. \
In the "application phase" we only need the complete noise and text which are used by the forward process to generate the images. 

Output
------
The output of the model are image tokens which are decoded to images by an image decoder.

Loss function 
------

The model uses two loss function combined, by addding them: $L = L_{denoise} + L_{edge}$

The first loss function $L_{denoise}=E_{t,x_0,\epsilon ∼\mathcal{N}(0,I)}\Vert \epsilon −D_\theta(x_t,t)\Vert_2^2$ is used for the denoising neural network. \
The loss function $L_{edge}=Focal(P_θ(M),I_{edge})$ is used to train the transformer blocks with Edge-supervised Learning, Focal denotes the focal loss employed to measure the discrepancy between the predicted and the “ground-truth” edge maps. A focal loss is a modified cross-entropy loss. It has the benefit of performing better with class imbalances. [[10]](#10)

Further settings for training 
------
The model uses an AdamW optimizer with a learning rate of 1e-04, weight decay set to 0.0 and a batch size of 2,000 for training. 

AdamW optimization uses a stochastic gradient descent approach that is based on adaptive estimation of first-order and second-order moments with an added method to decay weights per the techniques discussed in the paper, 'Decoupled Weight Decay Regularization' [[11]](#11) [[12]](#12). \
Weight decay or L2 regularization is a regularization technique where the sum of the squares of the weights from the model is added to the loss function, what results in punishing the model for large weights. In contrast to L1 regularization it does not lead to sparse weights.

RAPHAEL Architecture
======

As explained earlier, in general a Diffusion Model consists of two parts: the forward diffusion process and the reverse diffusion process.

In the RAPHAEL model the forward diffusion process is implemented as described in the section about Diffusion Models. \
For the reverse diffusion process RAPHAEL uses an U-Net Architecture as the denoising network. It consists of 16 transformer blocks, which employ two MoE layers. The first is a Time-MoE layer and the second a Space-MoE layer. 

The general structure of the used UNet-Architecture is shown below.

<!--- ![image](https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/1809bc49-fbbd-4402-8717-484f0403f9c1) --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/5a8bce05-40a7-403a-991a-927c5d3e13a9">

How are Mixture of Experts used?
------
In the RAPHAEL model MoEs are used in two layers in each of the 16 transformer blocks. \
Every transformer block consists of four key components, the Self Attention layer, the Cross Attention layer, the Time-MoE layer and the Space-MoE layer as image below shows.

<!--- <img width="414" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/e5d19f96-5af1-4ba7-97d7-80fc75212dfc"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/dba935c6-beb3-4891-8b5e-dfa3efd42295">

For an intuitive understanding you can think of the different paths which the MoEs produce as different "painters" which are all responsible for a different point of the image, as the authors of the paper write.

What are Time-MoE?
------
Time-MoE are MoE Layers which assign the image in different denoising time steps to different expert models. 

The Time-MoE Layer takes the feature data from the Cross Attention layer as input. The output of the layer is the output of the selected expert. \
A Time-MoE layer constists of a Text Gate Network and the expert models as shown in the image below. It takes the time step and the features as input.

<!--- <img width="225" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/d3d0eb5b-1656-4932-83ca-f3c4638861f0"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/f6c0fbff-be5b-4a2e-afc9-11bca2800b1e">

The Time Gate Network is implemented as a feed forward network and chooses experts for the different features, this can be formulated with $h\prime(x) = te_{t_{router}(t_i)}(h_c(x_t))$ where $h_c(x_t)$ represents the features from the Cross Attention layer and $t_{router}(t_i)=argmax(softmax(G′(E′_θ(t_i))+ϵ))$ which returns the index of the chosen expert. In the formula $te_i$ represents the differen experts. \
An example of the result of the assignments can be seen in the image below:

<!--- <img width="597" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/37ea0970-9829-4005-9946-cb757f6d62ba"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/8eca0dbc-7edc-4985-bfaa-1440bc4df51e">

What are Space-MoE?
------
Space-MoE is a MoE Layer which assigns specific text tokens to their corresponding image regions. \
The layer takes the data from the Time-MoE layer as input. \
The output of the Space-MoE Layer is built by taking the mean of all expert models, calculated by the following formula: 
$\frac{1}{n_y} \Sigma_{i=1}^{n_y} e_{route(y_i)}(h′(x_t) \cdot M_i)$. $M_i$ is a binary two-dimensional matrix which can be understood as the image region the i-th text token should correspond to. $\cdot$ is the hadamard product and $h\prime(x_t)$ are the features from the Time-MoE layer.  

A Space-MoE layer constists of a Text Gate Network and the expert models as shown in the image below. It takes text as input.

<!--- <img width="335" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/3cefc66c-c482-49a0-a75b-f0bbafaba431"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/eba93db2-8496-464a-8262-0f009b1c3759">

The Text Gate Network does the assignment using the formula \$route(y_i)=argmax(softmax(G(E_θ(y_i))+ϵ))\$ which returns the index of the corresponding expert.
It is implemented as a feed forward network with text tokens as input. \
As a result of the Space-MoE Layer, as shown in the picture below, different categories activate different diffusion paths 

<!--- <img width="316" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/681216f0-19cd-4359-abef-51d55217c280"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/287fa16b-94d1-412d-87a9-e51d9fd3a60f">

What is Edge-supervised Learning?
------
Edge-supervised Learing uses an edge detection module to extract boundary information, which is then used to supervise the model in preserving detailed image features. \
The module takes the attention map M as input and is trained using the loss function $L_{edge}$. The output is the predicted edge map. \
Since with larger timesteps t the attention map loses detail a hyperparameter is used to stop edge-supervised learning when t becomes too large.

The image below shows the attention map from the transformer block and the edges extracted by the edge detector next to the image.

<!--- <img width="698" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/74df4905-9526-4a0f-aaa6-a7faa9e8b7fa"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/74df4905-9526-4a0f-aaa6-a7faa9e8b7fa">

(d) shows that nearly twice as much people prefer the results of the model using Edge-supervised Learning than people prefering the model without it.

Experiments and Benchmarks
======

Experiments
------
The LAION-5B and some more internal datasets are used for the experiments. Images from LAION-5B were previously filtered using the same aesthetic scorer as Stable Diffusion, only images with a score of at least 4.7 and without watermarks are used. The text description from the LAION-5B datasets were cleaned by removing useless information, for example HTML tags.  

<!--- <img width="634" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/86940ad0-1702-42b7-9c18-52113266eed3"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/23f35b64-ee85-4886-b310-c7e28d1a2585">

The hyperparameter \$\alpha\$ results in an optimal FID-5k score at about 0.2, smaller and larger alpha values decrease the performance. \
This is explained by the fact that $\alpha$ is part of the threshold which decides if an entry in the Cross Attention Map is 0 or 1, so a bigger $\alpha$ leads to a sparser Map, while a smaller to a less sparser Map. As the paper describes, the value of 0.2 implies a balance between preserving adequate features and avoiding the use of unimportant features. \
The hyperparameter \$T_c\$ is optimal at about 500, for smaller values it slowly decreases, for bigger it decreases fast. \
The result is logical, because a small $T_c$ stops edge-supervised learning earlier which can result in worse results. On the other side a large $T_c$ stops edge-supervised learning very late which can also worsen the results, because the edge detector module is no longer able to retrieve useful information from the image at that timestep.

Experiments between the CLIP score and FID-5k show that the model with all three Space-MoE, Time-MoE and Edge-supervised Learning is overall the best one. \
The best FID-5k value is achieved at a CLIP score at about 0.33 for all models except the model without Space-MoE which has its peak at a CLIP score of about 0.315. Since we want to achieve a high CLIP score and a low FID-5k value, the model with all the modules is as previously mentioned the best one. This implies that all modules contribute effectively. Interesting is also that the model without Space-MoE has its FID-5k peak with a significantly lower CLIP score than the other models while having a quite similiar FID-5k value compared to the model without Time-MoE and the model without Edge-supervised learning. This implies that the Space-MoE have a big contribution to the better text alignment of RAPHAEL.

The number of Experts can influence the FID-5k score and the computational complexity. \
The FID-5k score gets better very fast at the beginning but flattens out quite fast, for the Time-MoE earlier than for the Space-MoE. \
This shows that a larger number of both Space-MoE and Time-MoE has a positive effect on the quality of the output. 
But on the other hand the computational complexity grows with an increasing number of experts. \
This results makes sense since a larger number of experts increases the number of computations and therefore slows down the model. But overall a model using a sparse MoE approach such as RAPHAEL does will usually be more effient than a model using just a single more complex FFN/MLP. 

Benchmarks
------
For benchmarking 30,000 images from the MS-COCO 256 x 256 dataset and the zero-shot Frechet Inception Distance (FID) were used. The FID score is usually calculated using the Inception v3 model which compares the original dataset to the generated one in quality and diversity. 

<!--- <img width="640" alt="image" src="https://github.com/Florian-de/floriandreyer.github.io/assets/64322175/38885b83-95db-40fd-aa1b-bfce8e309df2"> --->
<img width="750" alt="image" src="https://github.com/Florian-de/Florian-de.github.io/assets/64322175/a226c0c8-8725-4466-bb76-d4c03b79db10">

The table shows that RAPHAEL outperforms all competitors on the Zero-shot FID-30k. \
Especially in comparison with the two popular models Stable Diffusion and DALL-E it beets them by 21% and 37%.

Discussion
======

The most obvious advantage of the model is the more accurate text in the generated images and the overall higher image quality. As discussed in the experiments section, the Space-MoE improve the text allignment a lot. But all three key differences from RAPHAEL, the Space-MoE, the Time-MoE and the Edge-supervised Learning distribute equally to the image quality. This can be derived from the fact, that as in the experiments section discussed the models without one of them all peak at about the same FID-5k value. \
On the other hand is the high GPU usage for the training. The model was trained on 1,000 NVIDIA A100s for two months, so in total about 1.46 million A100 GPU hours, in comparison Stable Diffusion was trained for about 150,000 A100 GPU hours [[13]](#13). So the GPU usage for the training of RAPHAEL was about 10x the GPU usage for Stable Diffusion. \
Another advantage is that MoE Architectures make the models more efficient during inference due to sparse assignements. \
The fact that the model is not open source is also a drawback, since it reduces the benefit for the research community.

References
======

<a id="1">[1]</a> 
Zeyue Xue et al. (May 2023),
"RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths". 
[arXiv:2305.18295](https://arxiv.org/abs/2305.18295)

<a id="2">[2]</a> 
Adobe,
[Generative Fill & Expand](https://helpx.adobe.com/photoshop/using/generative-fill.html)

<a id="3">[3]</a> 
Charlie Harris (Jun 2023), 
[Diffusion Models in Generative Chemistry for Drug Design](https://medium.com/@cch57/exploring-the-promise-of-generative-models-in-chemistry-an-introduction-to-diffusion-models-31530e9d1dcb)

<a id="4">[4]</a> 
Jaskaran Bhatia (Jul 2023), 
[Summarizing the Evolution of Diffusion Models: Insight from Three Research Papers](https://medium.com/@jaskaranbhatia/summarizing-the-evolution-of-diffusion-models-insights-from-three-research-papers-6889339eba4)

<a id="5">[5]</a> 
Jonathan Ho et al. (Dec 2020),
"Denoising Diffusion Probabilistic Models",
[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

<a id="6">[6]</a> 
Kemal Erdem (Nov 2023),
[Step by Step visual introduction to Diffusion Models](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models)

<a id="7">[7]</a> 
3Blue1Brown (Apr 2024),
[Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=71s)

<a id="8">[8]</a> 
Zixiang Chen et al. (Aug 2022),
"Towards Understanding Mixture of Experts in Deep Learning",
[arXiv:2208.02813](https://arxiv.org/abs/2208.02813)

<a id="9">[9]</a> 
Hugging Face (Dec 2023),
[Mixture of Experts Explained](https://huggingface.co/blog/moe)

<a id="10">[10]</a> 
Keras,
[Focal Loss](https://keras.io/api/keras_cv/losses/focal_loss/#:~:text=Focal%20loss%20is%20a%20modified,to%20deal%20with%20class%20imbalance)

<a id="11">[11]</a> 
Keras,
[AdamW](https://keras.io/api/optimizers/adamw/)

<a id="12">[12]</a> 
Ilya Loshchilov et al. (Jan 2019),
"Decoupled Weight Decay Regularization",
[arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

<a id="13">[13]</a> 
Emad Mostaque (Aug 2020),
[Post on X](https://x.com/emostaque/status/1563870674111832066)
