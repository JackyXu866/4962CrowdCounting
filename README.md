# Crowd Counting
CSCI 4962 Projects in Machine Learning & AI

Jacky Xu, Jerry Lu, Zach F., Youchen Shao

## Presentation: 
- [Google Slide](https://docs.google.com/presentation/d/1SNt_8uEQ2Ay64APaiU-T_Yes42mc7AzqIVF3-FzpkH4/edit?usp=sharing)

## Group Notes:
- [Project Proposal](https://docs.google.com/document/d/1Pmqh2bMFHnq9pJ7FMtc9t6LpOXq5ENqaQQWXjcr-hyY/edit?usp=sharing)
- [Meeting Notes](https://colab.research.google.com/drive/1hnV0JVIjQuB9-7PmgIsO_KnXKhxQprIF?usp=sharing)


# Analyze & Report

## highlights
 - network capacity is not the only factor of model design
 - MCNN creates diffenert patch sizes for various sizes of head
 - ic-CNN creates a two layer structure and use LR to guide the generation of HR
 - CG-DGCN extends the concept of LR HR, redesigns the loss function, and adds residual blocks to reinforce the model's ability
 - heatmap with gaussian blur
 - checkpoint during training
 - usage of tensorboard

Our plan was that each of us creates their own model based on the papers available, and we can build an ensemble model by aggregating everyone's model for general crowd-counting tasks after we finish training our own model respectively. However, we later realized that we falsely made the assumption that each crowd-counting model has its specified optimal scenario. In fact, every model is trying to build a generic model and the state-of-the-art model CG-DGCN is too complex to implement. Also, the datasets didn't limit to single scenes and are instead trying to include various cases such as different point of views, different crowd densities, and different light and weather conditions. It seems that it is unnecessary to build the ensemble model. In addition, the CG-DGCN is too complex and we can't really understand the explanation of the confidence-guided residual block in the network and the loss functions built on the CGRB. Nevertheless, we learned a lot when reading the paper and playing with the models.

We built the baseline model and designed the corresponding loss function based on our own knowledge. It has an encoder-decoder structure with great learning capacity (1,462,403 parameters). However, the model performs poorly on our task. The result indicates that capacity is not the only factor contributing to the performance of the model. Other factors such as the architecture of the model, the usage of the activation and batchnorm functions, and the design of the loss function are also vital for a good model. 

The summaries of the models and the corresponding training datasets are listed below
- City Street with MCNN - [link](CityStreet/README.md)
- Shanghai Tech with ic-CNN - [link](ShanghaiTechPartB/README.md)
- Also see comments in our codes

| Model                     | Year |     Params | Params size | MAE,MSE on UCF |
|---------------------------|------|-----------:|------------:|---------------:|
| baseline(encoder-decoder) | N/A  |  1,462,403 |     1.65 MB | N/A            |
| MCNN                      | 2016 |    134,245 |     0.54 MB | 377.6, 509.1   |
| ic-CNN                    | 2018 |  7,947,150 |    30.32 MB | 260.9, 365.5   |
| CG-DGCN                   | 2020 | 29,590,199 |   112.88 MB | 112.2, 176.3   |

We can see that newer models are built on the shoulder of older models, and are trying to solve some problems exposed by the older models. MCNN uses three different window sizes for detecting various head sizes, which gives desirable performance when the head size falls within the range of the three window sizes. However, MCNN performs poorly when the head size is not expected. (consider the extreme case that the crowd is very close to the camera) ic-CNN takes a different approach and tries to create a more general model. It first generates a LR heatmap which provides the spatial and density information of the crowd, and utilizes the LR heatmap to guide the generation of the HR heatmap, which fine grinds the information missed by the LR heatmap, and gives a more accurate estimation with higher resolution. However, it seems that design of the HR heatmap data stream (`hrCNN1`, `hrCNN2`) is problematic since the LR data streams and the HR data streams provide the same level of abstraction, and some concrete features (low-level features) are ignored by the model. CG-DGCN tackles this issue by using VGG16 as the backbone and extracting features at different levels to create the heatmap, which uses all the information in the image effectively. The model also redesigns the loss function and introduces residual learning to guide the model to the optimal state.

We can also see the trend of the increasing model complex. CG-DGCN has 300 times more parameters than MCNN. The boosted complex not only causes a longer training time but also brings problems such as small gradients and vanishing gradients. In our experiments, MCNN converges in 100 epochs and gives a good result, meanwhile ic-CNN gives a meaningful LR heatmap and an unstable HR heatmap after 100 epochs.


## Other thoughts

### loss function
We notice that almost all the models use MSE as the loss function using a pixel-by-pixel difference comparison. Since most pixels on the heatmaps are 0, It may tend to generate bias for the model to go toward 0, which means the model will be in favor of underestimating the counting of the crowd. In addition, we observed that the loss for pixel-wised compare is small since most pixels on the heatmap are 0 or 1, leading to a small gradient and longer training time. In the paper *Scale Aggregation Network for Accurate and Efficient Crowd Counting*, the authors proposed a novel loss function that uses the local pattern consistency loss to assist the Euclidean loss.
The local pattern consistency loss is calculated by SSIM index to measure the structural similarity between the estimated heatmap and corresponding ground truth.

### hardware limitation
Since there is no pretrained model for our models, we have to train the models from scratch. Colab by default gives T4 GPU, which has an FP32 performance of 8.141 TFLOPS and 16GB RAM. Since T4 is a low-end graphic card, it increases our training time significantly, leaving us less time to make adjustments on our model.

Also, since we are dealing with image inputs, the memory consumptions are larger than we expect. Our GPU somethings run out of memory, and we have to retrain the entire model. (before we created checkpoints for the model) As a result, we have to set our batch size to a very small size such as 2 images or 3 images to avoid out-of-memory.

In model ic-CNN, we mitigated this issue by creating checkpoint after each epoch. The checkpoints are automatically saved to google drive, which minimized the impact due to timeout and hardware failure.


## DataSets:
- [Shanghai Tech Crowd Data](https://www.kaggle.com/datasets/tthien/shanghaitech):
  - Part A: Jacky Xu
  - Part B: Youchen Shao 
- [JHU Crowd++](http://www.crowd-counting.com/): Zach F.
- [City Street](http://visal.cs.cityu.edu.hk/research/citystreet/): Jerry Lu

## Models:
- Custom Encoder & Decoder: Jacky Xu
- [MCNN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf): Jerry Lu
- [CG-DRCN](http://www.crowd-counting.com/assets/img/jhucrowdv1_iccv19.pdf): Zach F.
- [ic-CNN](https://arxiv.org/abs/1807.09959): Youchen Shao

## References:
- [Iterative Crowd Counting](https://arxiv.org/abs/1807.09959)
- [ShanghaiTech Papers with Code](https://paperswithcode.com/dataset/shanghaitech)
- [Crowd-Counting Attention Network](https://arxiv.org/pdf/2201.08983.pdf)
- [Iterative Crowd-Counting](https://paperswithcode.com/paper/iterative-crowd-counting)
- [ShanghaiTech Dataset on Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech)
- [Awesome Crowd Counting Repo](https://github.com/gjy3035/Awesome-Crowd-Counting)
- [Scale Aggregation Network for Accurate and Efficient Crowd Counting](https://link.springer.com/chapter/10.1007/978-3-030-01228-1_45)