from imgaug import augmenters as iaa

aug_seq = iaa.Sequential([
    iaa.Fliplr(p=0.5),
    iaa.Sometimes(
        0.3,
        iaa.Multiply((0.75, 1.25))
    ),
    iaa.Sometimes(
        0.3,
        iaa.AdditiveGaussianNoise()
    ),
    iaa.Affine(
        rotate=(-5, 5),
        scale=(0.8, 1.2)
    )
])
