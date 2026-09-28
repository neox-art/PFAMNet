def classFactory(iface):
    from .cropland_sam2_labeler import CroplandSam2LabelerPlugin

    return CroplandSam2LabelerPlugin(iface)
