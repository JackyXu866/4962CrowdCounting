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
