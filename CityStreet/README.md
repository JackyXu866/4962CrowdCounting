# CityStreet Dataset

## Structure

- `CityStreet.py` is a torch `Dataset`.

- `MCNN.ipynb` is the modified of MCNN.

- `test.ipynb` is for EDA and testing.

## Dataset

This dataset implements `CityStreet` class.

```python
CityStreet(
    path: os.PathLike,          # file path to CityStreet root folder
    train: bool,                # True for training set, False for testing
    view: OneOf[1,2,3],         # selecting viewpoints
    skip_empty: bool,           # True for skipping unlabeled images
    target_resize_factor: float,# resize target to the smaller resolution, heatmap height->1520/x
    transform: Callable,        # transforms on images
    target_transform: Callable, # transforms on target (heatmap)
)
```

This class requires `labels`, `image_frames` and `ROI_maps` in the `path`.
And the targets are heatmaps, each pixel represents the number of heads
in that area.

It is recommended to use `GaussianBlur` on the heatmap.  According to the
MCNN paper, it could be determined by the density of heads.

#### Example

```python
data_view1 = CityStreet(
    "./data", True, 1, target_resize_factor=1520/64,
    transform=transforms.Compose([transforms.Resize(512)]),
    target_transform=transforms.Compose([transforms.GaussianBlur(7, (2, 2))])
)
```

## Model

In `MCNN.ipynb`, it implements a slightly modified Multi-Column CNN for
crowd counting.  There are 3 columns with different kernel sizes for
different head sizes in images, because of the distance.

In each column, it performs four CNNs, with batch normalizations and
ReLUs; and two layers have max pooling layer which decrease the
resolution by 4 times in total.

#### Performance

CityStreet, all 3 views at 512px height (keep aspect-ratio), 100 epochs,
learning rate `0.0001`, using MSE on test set: `7.516` (mean),
`323.16` (sum).

## References

Qi Zhang and Antoni B. Chan.
[Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr19-wacc.pdf)

Yingying Zhang, et al.
[Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
