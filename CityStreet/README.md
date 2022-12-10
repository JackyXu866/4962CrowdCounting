# CityStreet Dataset, MCNN implementation

## Structure

- `CityStreet.py` is a torch `Dataset`.

- `MCNN.ipynb` is the modified version of MCNN.

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

#### Implementation

The code firstly checks the required directories.  After the check, it
then processes all the labels:

- If this is a training set, remove the testing set by the filename.
- Drop unused data columns.
- If `skip_empty` is defined, remove the empty labels.
- Parse the labels in corresponding JSON file.
- Remove the head positions that are out of scope, using the ROI map.

After processing all the labels, the initialization is complete.  At the
time of retrieving, it will:

- Read the image and convert it to (0—1) in torch Tensor.
- Create the heatmap based on the resize factor:
    - If the resize factor is specified, then one pixel in the heatmap
      may correspond with multiple head positions.
    - The energy in each pixel (0—255) represents the number of head in
      that area.
    - Drop possible out-of-scope pixels after the resize.
- Apply data transforms.
- Apply target transforms.
- Return `(image, heatmap)`.

#### Example

```python
# Read from `./data', only training set at viewport 1, resize the heatmap
#  height to 64px, resize the height of images to 512px, apply the Gaussian
#  blur with kernel 7 sigma 2.
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

#### Implementation

It firstly creates MCNN class based on the paper.  Then initializes all
the CityStreet dataset and concatenates three viewports.  The validation
set is 20% of the training set, and the test set is unchanged.

All the datasets are applied with a resize on images and a Gaussian blur
on heatmaps, with k=11 and sigma=3.

Onto the training part, it uses the following settings:

- Device: GPU
- Batch size: 6
- Optimizer: Adam, LR=0.0001
- Criterion: MSELoss, sum
- Epochs: 100

#### Performance

`7.516` (mean), `323.16` (sum), using the same settings above, on the
test set.

## References

Qi Zhang and Antoni B. Chan.
[Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs](http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr19-wacc.pdf)

Yingying Zhang, et al.
[Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
