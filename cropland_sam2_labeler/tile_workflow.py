"""Pixel-aligned 512 x 512 GeoTIFF chips and paired parcel layers."""
from pathlib import Path
from osgeo import gdal
from qgis.core import (QgsCoordinateTransform, QgsProject, QgsPointXY, QgsRasterLayer,
                       QgsMapLayerType, QgsGeometry, Qgis)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QFileDialog
from qgis.gui import QgsMapTool, QgsRubberBand
from .shapefile_writer import ensure_shapefile


def tile_window(dataset, x, y):
    if dataset is None or not dataset.GetProjection():
        raise ValueError('无法打开带坐标参考的源影像。')
    if dataset.RasterXSize < 512 or dataset.RasterYSize < 512:
        raise ValueError('源影像不足 512×512 像素，不能生成完整标签块。')
    inverse = gdal.InvGeoTransform(dataset.GetGeoTransform())
    if inverse is None:
        raise ValueError('无法转换源影像像素坐标。')
    col, row = gdal.ApplyGeoTransform(inverse, x, y)
    if not (0 <= col < dataset.RasterXSize and 0 <= row < dataset.RasterYSize):
        raise ValueError('请在底图范围内选择裁剪中心。')
    left = min(max(int(col) - 256, 0), dataset.RasterXSize - 512)
    top = min(max(int(row) - 256, 0), dataset.RasterYSize - 512)
    return left, top


def crop_tile(source, destination, x, y):
    """Coordinates are in the source CRS. No resampling or padding is performed."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError('裁剪文件已存在，请换一个名称。')
    dataset = gdal.Open(source)
    left, top = tile_window(dataset, x, y)
    result = gdal.Translate(str(destination), dataset, format='GTiff',
                            srcWin=[left, top, 512, 512],
                            creationOptions=['COMPRESS=LZW', 'TILED=YES'])
    if result is None:
        raise RuntimeError('裁剪失败。')
    result.FlushCache()
    result = None
    return left, top


class TileFrameTool(QgsMapTool):
    pointAdded = pyqtSignal(float, float, int)
    failed = pyqtSignal(str)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.dataset = None
        self.raster = None
        self.frame_geometry = None
        self.rubber = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self.rubber.setColor(QColor(255, 220, 0, 30))
        self.rubber.setStrokeColor(QColor(255, 210, 0))
        self.rubber.setWidth(2)
        self.setCursor(Qt.CrossCursor)

    def configure(self, raster):
        self.dataset = gdal.Open(raster.source())
        self.raster = raster
        if self.dataset is None or self.dataset.RasterXSize < 512 or self.dataset.RasterYSize < 512:
            raise ValueError('底图必须至少为 512×512 像素。')

    def update_frame(self, point):
        self.frame_geometry = None
        self.rubber.reset(Qgis.GeometryType.Polygon)
        if self.dataset is None:
            return False
        try:
            map_crs = self.canvas.mapSettings().destinationCrs()
            to_source = QgsCoordinateTransform(map_crs, self.raster.crs(), QgsProject.instance())
            p = to_source.transform(point)
            left, top = tile_window(self.dataset, p.x(), p.y())
            gt = self.dataset.GetGeoTransform()
            corners = [(left, top), (left+512, top), (left+512, top+512), (left, top+512), (left, top)]
            # Densified edges stay accurate when map and raster projections differ.
            ring = []
            for a, b in zip(corners, corners[1:]):
                for step in range(16):
                    t = step / 16
                    ring.append(QgsPointXY(*gdal.ApplyGeoTransform(gt, a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))))
            ring.append(ring[0])
            geometry = QgsGeometry.fromPolygonXY([ring])
            geometry.transform(QgsCoordinateTransform(self.raster.crs(), map_crs, QgsProject.instance()))
            self.frame_geometry = geometry
            self.rubber.setToGeometry(geometry, None)
            return True
        except Exception:
            return False

    def activate(self):
        super().activate()
        self.update_frame(self.canvas.extent().center())

    def canvasMoveEvent(self, event):
        self.update_frame(self.toMapCoordinates(event.pos()))

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.RightButton:
            return
        point = self.toMapCoordinates(event.pos())
        if self.update_frame(point):
            self.pointAdded.emit(point.x(), point.y(), 1)
        else:
            self.failed.emit('请将截取框移入底图范围后再右键截取。')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.canvas.unsetMapTool(self)

    def deactivate(self):
        self.rubber.reset(Qgis.GeometryType.Polygon)
        self.frame_geometry = None
        super().deactivate()


class TileWorkflow:
    def start_tile_selection(self):
        if self.split_pieces or self.preview_geometries:
            self.set_error('请先确认或取消当前地块，再截取新的影像块。')
            return
        raster = self.raster_combo.currentLayer()
        return_to_overview = False
        if raster:
            source = QgsProject.instance().mapLayer(raster.customProperty('cropland/source_layer', ''))
            if source:
                raster = source
                return_to_overview = True
        if not raster or not raster.isValid():
            self.set_error('请先选择底图遥感影像。')
            return
        self.end_segmentation()
        self.tile_source = raster
        try:
            self.tile_tool.configure(raster)
            self.raster_combo.setLayer(raster)
            for node in QgsProject.instance().layerTreeRoot().findLayers():
                if node.layer() and node.layer().type() == QgsMapLayerType.RasterLayer:
                    node.setItemVisibilityChecked(node.layerId() == raster.id())
            if return_to_overview:
                extent = QgsCoordinateTransform(raster.crs(), self.canvas.mapSettings().destinationCrs(), QgsProject.instance()).transformBoundingBox(raster.extent())
                self.canvas.setExtent(extent)
            self.canvas.setMapTool(self.tile_tool)
            self.canvas.refresh()
            self.set_status('移动黄色 512×512 框，右键截取；Esc 取消。')
        except Exception as exc:
            self.set_error(str(exc))

    def select_tile_center(self, x, y, label):
        try:
            source = self.tile_source
            point = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(),
                                           source.crs(), QgsProject.instance()).transform(QgsPointXY(x, y))
            path, _ = QFileDialog.getSaveFileName(self, '保存 512×512 影像块', '', 'GeoTIFF (*.tif)')
            if not path:
                return
            if not path.lower().endswith('.tif'):
                path += '.tif'
            shp = str(Path(path).with_suffix('.shp'))
            if any(Path(shp).with_suffix(suffix).exists() for suffix in ('.shp', '.shx', '.dbf', '.prj')):
                raise ValueError('同名 SHP 已存在，请另选裁剪文件名。')
            crop_tile(source.source(), path, point.x(), point.y())
            tile = QgsRasterLayer(path, Path(path).stem + ' [512x512]')
            tile.setCustomProperty('cropland/source_layer', source.id())
            if not tile.isValid() or tile.width() != 512 or tile.height() != 512:
                raise ValueError('裁剪影像无效。')
            if source.renderer():
                tile.setRenderer(source.renderer().clone())
            QgsProject.instance().addMapLayer(tile)
            parcels = ensure_shapefile(shp, tile.crs())
            self.raster_combo.setLayer(tile)
            self.vector_combo.setLayer(parcels)
            self.output_path.setText(shp)
            self.window_size.setValue(512)
            # Hide the overview imagery, keeping the saved source layer in the project.
            for node in QgsProject.instance().layerTreeRoot().findLayers():
                layer = node.layer()
                if layer and layer.type() == QgsMapLayerType.RasterLayer:
                    node.setItemVisibilityChecked(layer.id() == tile.id())
            self.canvas.setExtent(tile.extent())
            self.canvas.refresh()
            self.canvas.unsetMapTool(self.tile_tool)
            if hasattr(self.iface, 'actionPan'):
                self.iface.actionPan().trigger()
            self.set_status('512×512 影像块已就绪。点击 SAM2 图标开始提取。')
        except Exception as exc:
            self.set_error(str(exc))
