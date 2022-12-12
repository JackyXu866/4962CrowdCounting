# Shanghai Tech Part A Data Process
[Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)

Converted to `torch.utils.data.Dataset` class
```python
PartADataset(
  image_file_path:          required,
  ground-truth_file_path:   required,
  torchvision.transforms:   optional,
  transforms.GaussianBlur:  optional
) -> tuple(image, density_map)
```
return a tuple of tensors

`density_map`: normalized 2d array with the size of image


# Auto-Encoder

This is a baseline model for this project. It consists of 3 layers of convolutional layers in the encoder, 
and 3 layers of convolution transpose layers in the decoder. In addition, there are 2 1x1 convolution layers 
serve as fully connected nodes. After the auto-encoder, the model should output a heatmap as the input y value.

After many testing with different parameter, we found out the model is not suitable for high resolution images.
The optimal resolution for them would be 128 x 128. When testing with 512 x 512 or 1024 x 1024, the model's output
is completely off. To deal with that, I tried to add 1 more layer to both the encoder and decoder. However, the VRAM
usage for that increased to 20 GB, which I deicded to stop.

Learning rate adjustion is also the essential part of this training. I started out with a typical 0.01 LR. The output 
quickly becomes all zeros within 1 epoch. I would imagine it is because the target heatmap contains mostly 0s with only
relatively small amounts of 1s for head. After a series of testing, I finally decide an optimal LR for this model would
be 0.0000001, which could provide a descent output.

When looking at losses for each batch trained, I found out the losses between images with many crowds and a few crowds 
varies a lot when using the MSE. I think that is because it is getting the squared difference per pixel, which if the 
target's sum is small the loss will be smaller, and the target's sum is big, the loss will be bigger. So I decided to 
create my own custom loss function. Instead of dividing the number of pixel, I decided to divide the sum of target, which 
is the number of heads presented on the image. To keep the division in a reasonable range, I applied the log to it. It turned
out the losses for each batch does not bounce up and down too much.

As the output turns out, the model cannot really understand what is a face/human. It is just extracting every features of 
the image and convert it to dots. For example, in one of the sample, the model thinks there are multiple people in the sky,
which is just branches of the tree. For crowd counting, I think it requires a more featured model to handle this task.