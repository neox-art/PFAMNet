from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsDefaultValue,
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)


DEFAULT_FIELDS = [
    QgsField("parcel_id", QVariant.String, len=36),
    QgsField("label", QVariant.String, len=32),
    QgsField("source", QVariant.String, len=32),
    QgsField("created_at", QVariant.String, len=24),
    QgsField("area_m2", QVariant.Double, len=20, prec=3),
    QgsField("confidence", QVariant.Double, len=10, prec=4),
    QgsField("notes", QVariant.String, len=128),
]


def default_fields():
    fields = QgsFields()
    for field in DEFAULT_FIELDS:
        fields.append(field)
    return fields


def ensure_shapefile(path, crs):
    import os
    for existing in QgsProject.instance().mapLayers().values():
        if isinstance(existing, QgsVectorLayer) and os.path.abspath(existing.source().split('|')[0]) == os.path.abspath(path):
            return existing
    layer = QgsVectorLayer(path, "cropland_labels", "ogr") if path else None
    if layer and layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        return layer
    if os.path.exists(path):
        raise RuntimeError('Existing output cannot be opened; refusing to overwrite it.')
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "UTF-8"
    writer = QgsVectorFileWriter.create(
        path,
        default_fields(),
        QgsWkbTypes.MultiPolygon,
        crs,
        QgsCoordinateTransformContext(),
        options,
    )
    if writer.hasError() != QgsVectorFileWriter.NoError:
        raise RuntimeError(writer.errorMessage())
    del writer
    layer = QgsVectorLayer(path, "cropland_labels", "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Unable to open created shapefile: {path}")
    QgsProject.instance().addMapLayer(layer)
    layer.setDefaultValueDefinition(layer.fields().indexFromName('parcel_id'), QgsDefaultValue('uuid()'))
    return layer


def append_feature(layer, geometry, attributes):
    if not layer or not layer.isValid():
        raise RuntimeError("Output shapefile layer is not valid.")
    feature = QgsFeature(layer.fields())
    feature.setGeometry(geometry)
    for key, value in attributes.items():
        if key in layer.fields().names():
            feature[key] = value
    if not layer.isEditable():
        layer.startEditing()
    layer.beginEditCommand('Add parcel')
    ok = layer.addFeature(feature)
    if not ok:
        layer.destroyEditCommand()
        raise RuntimeError("Failed to add feature to shapefile.")
    layer.endEditCommand()
    layer.triggerRepaint()
    return feature


def update_feature_geometry(layer, feature_id, geometry):
    if not layer or not layer.isValid():
        raise RuntimeError("Target layer is not valid.")
    if not layer.isEditable():
        layer.startEditing()
    layer.beginEditCommand('Trim parcel')
    ok = layer.changeGeometry(feature_id, geometry)
    if not ok:
        layer.destroyEditCommand()
        raise RuntimeError("Failed to update selected feature geometry.")
    layer.endEditCommand()
    layer.triggerRepaint()
