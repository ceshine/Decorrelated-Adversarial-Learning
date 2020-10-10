mixture = {
    'train_root': '/data/fuzhuolin/cross_age/data/aligned_dlib_112x96/mixture/train',
    'val_root': '/data/fuzhuolin/cross_age/data/aligned_dlib_112x96/mixture/val',
    'pat': '_|\.',
    'pos': 1,
    'n_cls': 24652
}

FGNET = {
    'train_root': '/data/fuzhuolin/cross_age/data/aligned_dlib_112x96/FGNET/train',
    'val_root': '/data/fuzhuolin/cross_age/data/aligned_dlib_112x96/FGNET/val',
    'pat': '_|\.',
    'pos': 1,
    'n_cls': 82
}

vgg_toy = {
    'train_root': '/data/fuzhuolin/cross_age/data/aligned_dlib_112x96/VGG_toy',
    'val_root': None,
    'pat': '_|\.',
    'pos': 1,
    'n_cls': 8
}

cacd = {
    'train_root': '/data/CACD2000/train/',
    'val_root': '/data/CACD2000/valid/',
    'pat': '_',
    'pos': 0,
    'n_cls': 2000
}

age_cutoffs = [18, 30, 45, 55]
