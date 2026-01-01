
DATASETS = {
    "ASL": {"classes": 24, "modalities": ["rgb", "depth", "rgbd"]},
    "OUHANDS": {"classes": 10, "modalities": ["rgb", "depth", "rgbd"]},
    "JU_V2_DIGIT": {"classes": 10, "modalities": ["rgb", "depth", "rgbd"]},
    "JU_V2_ALPHA": {"classes": 24, "modalities": ["rgb", "depth", "rgbd"]},
    "NUSII": {"classes": 10, "modalities": ["rgb"]},
    "MUGD": {"classes": 36, "modalities": ["rgb"]},
}

IMAGE_SIZES = {
    "vgg19": 224,
    "mobilenetv2": 224,
    "inceptionv3": 299
}
