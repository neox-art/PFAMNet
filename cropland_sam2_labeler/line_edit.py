"""Staged line edits for SAM previews and existing vector features."""
from uuid import uuid4
from datetime import datetime
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis, QgsGeometry, QgsProject, QgsCoordinateTransform, QgsFeature, QgsWkbTypes, QgsPointXY
from qgis.gui import QgsRubberBand
from .geometry_ops import line_edit_pieces, merge_geometries


class LineEditWorkflow:
    def finish_line_edit(self, line):
        self.preview_split(line)

    def start_line_edit(self, mode):
        try:
            if not self.preview_geometries and not self.split_pieces:
                self.target_vector_layer()
            self.split_tool.reset()
            self.line_mode = mode
            self.segmentation_active = True
            self.token += 1
            self.debounce.stop()
            self.canvas.setMapTool(self.split_tool)
            self.sync_tool_buttons()
            self.advanced_dialog.hide()
            self.set_status('右键结束本条线；继续画线可累积修改，空闲时右键确认地块。')
        except Exception as exc:
            self.sync_tool_buttons()
            self.set_error(str(exc))

    def preview_split(self, line_geometry):
        try:
            layer = self.output_layer
            if not layer or not layer.isValid():
                raise ValueError('请选择输出图层。')
            if self.split_pieces:
                sources = [QgsGeometry(g) for g in self.split_pieces]
                crs, target = self.split_crs, self.split_target
            elif self.preview_geometries:
                source = merge_geometries(self.preview_geometries)
                sources = [source]
                crs = self.preview_crs
                target = ('preview', layer.id(), source.asWkb(), crs.toWkt())
            else:
                layer = self.target_vector_layer()
                feature = next(layer.getSelectedFeatures())
                source, crs = feature.geometry(), layer.crs()
                sources = [source]
                target = ('feature', layer.id(), feature.id(), source.asWkb(), crs.toWkt())
            line = QgsGeometry(line_geometry)
            transform = QgsCoordinateTransform(self.canvas.mapSettings().destinationCrs(), crs, QgsProject.instance())
            center = line.boundingBox().center()
            a = transform.transform(center)
            b = transform.transform(QgsPointXY(center.x() + 6 * self.canvas.mapUnitsPerPixel(), center.y()))
            snap_tolerance = a.distance(b)
            line.transform(transform)
            pieces = []
            changed = False
            for source in sources:
                if source.intersects(line) or (self.line_mode == 'trim' and source.distance(line) <= snap_tolerance):
                    pieces.extend(line_edit_pieces(source, line, self.line_mode,
                                                   lambda g: self.area_m2(g, crs), snap_tolerance))
                    changed = True
                else:
                    pieces.append(source)
            if not changed:
                raise ValueError('线没有经过当前待确认地块，已有修改已保留。')
            if len(pieces) > 2:
                raise ValueError('本次切割会产生超过两个地块，已有修改已保留。')
            if len(pieces) == 2 and pieces[0].intersection(pieces[1]).area() > 0:
                raise ValueError('修边后两个待确认地块重叠，请调整修边线。已有修改已保留。')
            for band in self.split_bands:
                self.canvas.scene().removeItem(band)
            self.split_bands.clear()
            self.split_target, self.split_crs = target, crs
            self.split_pieces = pieces
            for band in self.preview_bands:
                band.hide()
            for index, piece in enumerate(pieces):
                display = QgsGeometry(piece)
                display.transform(QgsCoordinateTransform(crs, self.canvas.mapSettings().destinationCrs(), QgsProject.instance()))
                band = QgsRubberBand(self.canvas, Qgis.GeometryType.Polygon)
                band.setColor(QColor(30, 180, 80, 100) if index == 0 else QColor(30, 150, 220, 100))
                band.setWidth(2)
                band.setToGeometry(display, None)
                self.split_bands.append(band)
            self.set_status('待确认 %s 个地块；可继续画线修改，空闲时右键确认。' % len(pieces))
            self.sync_segmentation_ui()
        except Exception as exc:
            self.set_error(str(exc))

    def apply_split(self):
        if not self.split_pieces:
            return
        try:
            target = self.split_target
            layer = self.output_layer
            if not layer or layer.id() != target[1]:
                raise ValueError('输出图层已改变，请重新画线。')
            original = None
            if target[0] == 'feature':
                original = layer.getFeature(target[2])
                if (not original.isValid() or original.geometry().asWkb() != target[3]
                        or layer.selectedFeatureIds() != [target[2]] or layer.crs().toWkt() != target[4]):
                    raise ValueError('选中地块或几何已改变，请重新画线。')
            elif (not self.preview_geometries or merge_geometries(self.preview_geometries).asWkb() != target[2]
                  or self.preview_crs.toWkt() != target[3]):
                raise ValueError('SAM 预览已改变，请重新画线。')
            if not layer.isEditable() and not layer.startEditing():
                raise ValueError('无法编辑输出图层。')
            geometries = []
            for piece in self.split_pieces:
                geometry = QgsGeometry(piece)
                geometry.transform(QgsCoordinateTransform(self.split_crs, layer.crs(), QgsProject.instance()))
                if geometry.isEmpty() or not geometry.isGeosValid():
                    raise ValueError('结果几何无效。')
                if QgsWkbTypes.isMultiType(layer.wkbType()):
                    geometry.convertToMultiType()
                geometries.append(geometry)
            ids = []
            layer.beginEditCommand('Trim parcel' if self.line_mode == 'trim' else 'Split parcel')
            try:
                for index, geometry in enumerate(geometries):
                    feature = QgsFeature(original) if original else QgsFeature(layer.fields())
                    feature.setGeometry(geometry)
                    attributes = {'area_m2': self.area_m2(geometry, layer.crs())}
                    if not original or index:
                        attributes['parcel_id'] = str(uuid4())
                    if not original:
                        attributes.update(label=self.category.currentText(), source=self.backend_combo.currentText(),
                                          notes='accepted_line_edit',
                                          created_at=datetime.utcnow().isoformat(timespec='seconds') + 'Z')
                    for name, value in attributes.items():
                        if name in layer.fields().names():
                            feature[name] = value
                    if original and index == 0:
                        if not layer.changeGeometry(original.id(), geometry):
                            raise ValueError('更新地块失败。')
                        values = {layer.fields().indexFromName(k): v for k, v in attributes.items()
                                  if k in layer.fields().names()}
                        if values and not layer.changeAttributeValues(original.id(), values):
                            raise ValueError('更新属性失败。')
                    else:
                        feature.setId(-1)
                        if not layer.addFeature(feature):
                            raise ValueError('添加地块失败。')
                    ids.append(original.id() if original and index == 0 else feature.id())
                layer.endEditCommand()
            except Exception:
                layer.destroyEditCommand()
                raise
            self.cancel_split()
            self.clear_prompts()
            layer.selectByIds(ids)
            layer.triggerRepaint()
            self.refresh_parcels()
            self.canvas.unsetMapTool(self.canvas.mapTool())
            if hasattr(self.iface, 'actionPan'):
                self.iface.actionPan().trigger()
            self.set_status('已确认 %s 个地块，可撤销；保存后写入文件。' % len(ids))
        except Exception as exc:
            self.set_error(str(exc))

    def cancel_split(self):
        self.split_target = None
        for band in self.split_bands:
            self.canvas.scene().removeItem(band)
        self.split_bands.clear()
        self.split_pieces.clear()
        self.split_tool.reset()
        for band in self.preview_bands:
            band.show()
        if hasattr(self, 'main_ui') and hasattr(self, 'segmentation_active'):
            self.sync_segmentation_ui()

    def confirm_current(self):
        if self.split_pieces:
            self.apply_split()
        elif self.preview_geometries and self.canvas.mapTool() != self.split_tool:
            self.accept_preview()
