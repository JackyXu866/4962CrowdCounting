# Crowd Counting
CSCI 4962 Projects in Machine Learning & AI

## Presentation: 
- [Google Slide](https://docs.google.com/presentation/d/1SNt_8uEQ2Ay64APaiU-T_Yes42mc7AzqIVF3-FzpkH4/edit?usp=sharing)

## Group Notes:
- [Project Proposal](https://docs.google.com/document/d/1Pmqh2bMFHnq9pJ7FMtc9t6LpOXq5ENqaQQWXjcr-hyY/edit?usp=sharing)
- [Meeting Notes](https://colab.research.google.com/drive/1hnV0JVIjQuB9-7PmgIsO_KnXKhxQprIF?usp=sharing)

## Analyze
- City Street with MCNN - [link](CityStreet/README.md)
- Shanghai Tech with ic-CNN - [link](ShanghaiTechPartB/README.md)

## Other thoughts

### loss function
We notice that almost all the models use MSE as the loss function using a pixel-by-pixel difference comparison. Since most pixels on the heatmaps are 0, It may tend to generate bias for the model to go toward 0, which means the model will be in favor of underestimating the counting of the crowd.

### hardware limitation
Since there is no pretrained model for our models, we have to train the models from scratch. Colab by default gives T4 GPU, which has an FP32 performance of 8.141 TFLOPS and 16GB RAM. Since T4 is a low-end graphic card, it increases our training time significantly, leaving us less time to make adjustments on our model.

Also, since we are dealing with image inputs, the memory consumptions are larger than we expect. Our GPU somethings run out of memory, and we have to retrain the entire model. (before we created checkpoints for the model) As a result, we have to set our batch size to a very small size such as 2 images or 3 images to avoid out-of-memory.

### dataset limitation
We were trying to provide a universal model for general crowd-counting tasks. But, almost all the datasets are under good lighting conditions, where the weather is usually sunny with a clear view. It would be good if we could find some dataset that is no under optimal conditions such as at night where the lighting condition is undesirable.

## DataSets:
- [Shanghai Tech Crowd Data](https://www.kaggle.com/datasets/tthien/shanghaitech):
  - Part A: Jacky Xu
  - Part B: YC Shao 
- [JHU Crowd++](http://www.crowd-counting.com/): Zach F.
- [City Street](http://visal.cs.cityu.edu.hk/research/citystreet/): Jerry Lu

## Models:
- Custom Encoder & Decoder: Jacky Xu
- [MCNN](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf): Jerry Lu
- [CG-DRCN](http://www.crowd-counting.com/assets/img/jhucrowdv1_iccv19.pdf): Zach F.
- [ic-CNN](https://arxiv.org/abs/1807.09959): YC Shao

## References:
- [Iterative Crowd Counting](https://arxiv.org/abs/1807.09959)
- [ShanghaiTech Papers with Code](https://paperswithcode.com/dataset/shanghaitech)
- [Crowd-Counting Attention Network](https://arxiv.org/pdf/2201.08983.pdf)
- [Iterative Crowd-Counting](https://paperswithcode.com/paper/iterative-crowd-counting)
- [ShanghaiTech Dataset on Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech)
- [Awesome Crowd Counting Repo](https://github.com/gjy3035/Awesome-Crowd-Counting)
