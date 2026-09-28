from datetime import datetime
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QTimer, QSettings
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (Qgis, QgsMapLayerProxyModel, QgsProject, QgsRasterLayer, QgsVectorLayer,
    QgsCoordinateTransform, QgsGeometry, QgsDistanceArea, QgsDefaultValue, QgsWkbTypes)
from qgis.gui import QgsMapLayerComboBox, QgsRubberBand

from .geometry_ops import clean_geometry, merge_geometries
from .prompt_state import PromptBox, PromptPoint, PromptState
from .sam2_backend import Sam2BackendRunner, backend_from_settings
from .settings import Sam2Settings
from .shapefile_writer import append_feature, ensure_shapefile
from .tools import BoxPromptTool, PromptOverlay, PromptPointTool, SplitLineTool, InteractivePromptTool
from .segmentation_workflow import SegmentationWorkflow
from .line_edit import LineEditWorkflow
from .tile_workflow import TileWorkflow, TileFrameTool
from .worker import PersistentWorker
from uuid import uuid4


class CroplandSam2DockWidget(QDockWidget, TileWorkflow, LineEditWorkflow, SegmentationWorkflow):
    def __init__(self, iface):
        super().__init__("Cropland SAM2 Labeler", iface.mainWindow())
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.prompt_state = PromptState()
        self.prompt_overlay = PromptOverlay(self.canvas)
        self.backend_runner = Sam2BackendRunner()
        self.output_layer = None
        self.preview_geometries = []
        self.preview_bands = []
        self.split_pieces = []
        self.split_bands = []
        self.token = 0
        self.preview_crs = None
        self.split_target = None
        self.line_mode = 'trim'
        self.worker = PersistentWorker(self)
        self.worker.ready.connect(self.prediction_ready)
        self.worker.failed.connect(self.prediction_failed)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.predict_mask)
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.auto_save)
        self._build_tools()
        self._build_ui()
        config_path = Path(__file__).with_name('backend_config.json')
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding='utf-8'))
            self.backend_combo.setCurrentText(config['backend'])
            self.local_command.setText(config['local_command'])
            self.weights_path.setText(config['weights_path'])
            self.device_combo.setCurrentText(config['device'])
            self.simplify_tolerance.setValue(0)
            self.min_area.setValue(0)
        self.restore_settings()
        self.vector_combo.layerChanged.connect(self.bind_layer)
        self.raster_combo.layerChanged.connect(self.invalidate_prompts)
        self.canvas.destinationCrsChanged.connect(self.invalidate_prompts)
        self.autosave.toggled.connect(self.update_save_timer)
        self.save_interval.valueChanged.connect(self.update_save_timer)
        self.update_save_timer()
        for control in (self.backend_combo, self.device_combo):
            control.currentTextChanged.connect(self.configuration_changed)
        for control in (self.local_command, self.weights_path, self.service_url):
            control.textChanged.connect(self.configuration_changed)
        for control in (self.window_size, self.simplify_tolerance, self.min_area):
            control.valueChanged.connect(self.configuration_changed)
        self._connect_project_signals()
        self.refresh_raster_layers()
        self.bind_layer(self.vector_combo.currentLayer())

    def _build_tools(self):
        self.tile_tool = TileFrameTool(self.canvas)
        self.tile_tool.pointAdded.connect(self.select_tile_center)
        self.tile_tool.failed.connect(self.set_error)
        self.interactive_tool = InteractivePromptTool(self.canvas)
        self.interactive_tool.pointAdded.connect(self.add_prompt_point)
        self.interactive_tool.confirmed.connect(self.confirm_current)
        self.positive_tool = PromptPointTool(self.canvas, 1)
        self.negative_tool = PromptPointTool(self.canvas, 0)
        self.box_tool = BoxPromptTool(self.canvas)
        self.split_tool = SplitLineTool(self.canvas)
        self.positive_tool.pointAdded.connect(self.add_prompt_point)
        self.negative_tool.pointAdded.connect(self.add_prompt_point)
        self.box_tool.boxAdded.connect(self.add_prompt_box)
        self.split_tool.lineFinished.connect(self.finish_line_edit)
        self.split_tool.confirmed.connect(self.confirm_current)
        self.split_tool.cancelled.connect(self.cancel_split)
        self.canvas.mapToolSet.connect(self.on_active_tool_changed)

    def on_active_tool_changed(self, *args):
        if self.canvas.mapTool() not in (self.interactive_tool, self.positive_tool,
                                        self.negative_tool, self.box_tool):
            self.token += 1
            self.debounce.stop()

    def _build_ui(self):
        from .workbench_ui import build_ui
        build_ui(self)

    def _connect_project_signals(self):
        project = QgsProject.instance()
        project.layersAdded.connect(self.refresh_raster_layers)
        project.layersRemoved.connect(self.refresh_raster_layers)

    def refresh_raster_layers(self, *args):
        if self.raster_combo.currentLayer():
            return
        self.raster_combo.setLayer(None)
        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                self.raster_combo.setLayer(layer)
                break

    def settings(self):
        settings = Sam2Settings(
            backend=self.backend_combo.currentText(),
            service_url=self.service_url.text().strip(),
            local_command=self.local_command.text().strip(),
            weights_path=self.weights_path.text().strip(),
            device=self.device_combo.currentText(),
            simplify_tolerance=self.simplify_tolerance.value(),
            min_area=self.min_area.value(),
        )
        errors = settings.validate()
        if errors:
            raise ValueError("\n".join(errors))
        return settings

    def choose_output_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Output Shapefile", "", "ESRI Shapefile (*.shp)")
        if path:
            if not path.lower().endswith(".shp"):
                path += ".shp"
            self.output_path.setText(path)

    def add_prompt_point(self, x, y, label):
        if self.split_pieces:
            self.set_error('当前有待确认的边界修改，请右键确认或 Esc 取消后再提取。')
            return
        point = PromptPoint(x=x, y=y, label=label)
        if label == 1:
            self.prompt_state.positive_points.append(point)
            self.prompt_overlay.add_point(point, positive=True)
        else:
            self.prompt_state.negative_points.append(point)
            self.prompt_overlay.add_point(point, positive=False)
        self.update_prompt_label()
        self.remember_prompt()
        self.prompt_changed()

    def add_prompt_box(self, xmin, ymin, xmax, ymax):
        if self.split_pieces:
            self.set_error('请先确认或取消当前边界修改。')
            return
        self.prompt_state.boxes.clear()
        self.prompt_overlay.clear()
        for p in self.prompt_state.positive_points + self.prompt_state.negative_points:
            self.prompt_overlay.add_point(p, p.label == 1)
        box = PromptBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        self.prompt_state.boxes.append(box)
        self.prompt_overlay.add_box(box)
        self.update_prompt_label()
        self.remember_prompt()
        self.prompt_changed()

    def update_prompt_label(self):
        self.prompt_label.setText(
            f"正点 {len(self.prompt_state.positive_points)}   "
            f"负点 {len(self.prompt_state.negative_points)}   "
            f"框选 {len(self.prompt_state.boxes)}"
        )

    def clear_prompts(self):
        self.cancel_split()
        self.token += 1
        self.debounce.stop()
        self.clear_prediction_preview()
        self.prompt_state.clear()
        if hasattr(self, 'prompt_history'):
            self.prompt_history.clear()
            self.prompt_future.clear()
        self.prompt_overlay.clear()
        self.update_prompt_label()
        self.set_status('提示已清空。')

    def predict_mask(self):
        if self.split_pieces:
            self.set_error('请先确认或取消当前边界修改。')
            return
        self.cancel_split()
        try:
            if not self.prompt_state.has_prompt():
                raise ValueError("Add positive/negative points or a box first.")
            raster = self.raster_combo.currentLayer()
            if not raster or not raster.isValid():
                raise ValueError("Choose a valid imagery layer.")
            if raster.width() != 512 or raster.height() != 512:
                raise ValueError('请先裁剪 512×512 标签块，再进行 SAM2 分割。')
            settings = self.settings()
            self.token += 1
            token = self.token
            self.request_crs = raster.crs()
            self.request_backend = settings.backend
            self.request_canvas_crs = self.canvas.mapSettings().destinationCrs()
            self.request_cleanup = (settings.simplify_tolerance, settings.min_area)
            payload = {
                "raster_path": raster.source(),
                "raster_layer_name": raster.name(),
                "crs_authid": raster.crs().authid(),
                "prompt_crs": self.canvas.mapSettings().destinationCrs().toWkt(),
                "window_size": self.window_size.value(),
                "extent": {
                    "xmin": raster.extent().xMinimum(),
                    "ymin": raster.extent().yMinimum(),
                    "xmax": raster.extent().xMaximum(),
                    "ymax": raster.extent().yMaximum(),
                },
                "map_units_per_pixel": self.canvas.mapUnitsPerPixel(),
                "prompts": self.prompt_state.to_payload(),
            }
            self.set_status('SAM2 推理中…')
            self.clear_prediction_preview()
            if settings.backend == 'local' and 'sam2_predict.py' in settings.local_command:
                self.worker.submit(token, settings, payload)
            else:
                self.backend_runner.submit(backend_from_settings(settings), payload,
                    lambda result: self.prediction_ready(token, result),
                    lambda error: self.prediction_failed(token, error))
        except Exception as exc:
            self.set_error(str(exc))

    def on_prediction_ready(self, geometries):
        self.predict_btn.setEnabled(True)
        cleaned = []
        for geometry in geometries:
            if self.request_backend == 'mock':
                geometry = QgsGeometry(geometry)
                geometry.transform(QgsCoordinateTransform(self.request_canvas_crs, self.request_crs, QgsProject.instance()))
            fixed = clean_geometry(geometry, *self.request_cleanup)
            if fixed and not fixed.isEmpty():
                cleaned.append(fixed)
        if not cleaned:
            self.set_error("Prediction produced no valid polygon after cleanup.")
            return
        self.preview_geometries = cleaned
        self.preview_crs = self.request_crs
        self.draw_prediction_preview()
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
        self.set_status('预览已更新。')

    def on_prediction_error(self, message):
        self.predict_btn.setEnabled(True)
        self.set_error(message)

    def draw_prediction_preview(self):
        for geometry in self.preview_geometries:
            band = QgsRubberBand(self.canvas, Qgis.GeometryType.Polygon)
            band.setColor(QColor(30, 180, 80, round(self.preview_opacity.value() * 2.55)))
            band.setStrokeColor(QColor(0, 130, 60))
            band.setWidth(2)
            display = QgsGeometry(geometry)
            display.transform(QgsCoordinateTransform(self.preview_crs,
                self.canvas.mapSettings().destinationCrs(), QgsProject.instance()))
            band.setToGeometry(display, None)
            self.preview_bands.append(band)

    def clear_prediction_preview(self):
        for band in self.preview_bands:
            self.canvas.scene().removeItem(band)
        self.preview_bands.clear()
        self.preview_geometries.clear()
        self.accept_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)

    def accept_preview(self):
        if self.split_tool.points:
            self.set_error('请先右键结束正在绘制的线。')
            return
        if self.split_pieces:
            self.apply_split()
            return
        try:
            raster = self.raster_combo.currentLayer()
            if not raster or not raster.isValid():
                raise ValueError("Choose imagery layer before saving.")
            if not self.preview_geometries or self.preview_crs is None:
                raise ValueError('No current preview to accept.')
            layer = self.edit_layer()
            geometry = merge_geometries(self.preview_geometries)
            geometry.transform(QgsCoordinateTransform(self.preview_crs, layer.crs(), QgsProject.instance()))
            if geometry.isEmpty():
                raise ValueError("Preview geometry is empty.")
            feature = append_feature(
                layer,
                geometry,
                {
                    "parcel_id": str(uuid4()),
                    "label": self.category.currentText(),
                    "source": self.backend_combo.currentText(),
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "area_m2": self.area_m2(geometry, layer.crs()),
                    "confidence": None,
                    "notes": "accepted_sam2_preview",
                },
            )
            self.clear_prediction_preview()
            self.clear_prompts()
            layer.selectByIds([feature.id()])
            self.refresh_parcels()
            self.set_status('地块已加入编辑缓冲区，可撤销；保存后写入 SHP。')
        except Exception as exc:
            self.set_error(str(exc))

    def reject_preview(self):
        self.cancel_split()
        self.token += 1
        self.debounce.stop()
        self.clear_prediction_preview()
        self.set_status('预览已取消。')

    def target_vector_layer(self):
        layer = self.output_layer or self.iface.activeLayer()
        if not isinstance(layer, QgsVectorLayer) or not layer.isValid():
            raise ValueError("Select a valid polygon layer or accept a preview into an output shapefile first.")
        if not layer.selectedFeatureIds():
            raise ValueError("Select one cropland polygon to trim.")
        if len(layer.selectedFeatureIds()) > 1:
            raise ValueError("Select exactly one polygon for split editing.")
        return layer

    def prompt_changed(self):
        if self.split_pieces:
            return
        self.cancel_split()
        self.token += 1
        self.clear_prediction_preview()
        if self.live_preview.isChecked() and (self.prompt_state.positive_points or self.prompt_state.boxes):
            self.debounce.start(350)
        else:
            self.debounce.stop()
        self.sync_segmentation_ui()

    def configuration_changed(self, *args):
        self.cancel_split()
        self.token += 1
        self.debounce.stop()
        self.clear_prediction_preview()

    def invalidate_prompts(self, *args):
        self.clear_prompts()
        self.cancel_split()

    def prediction_ready(self, token, geometries):
        if token == self.token:
            try:
                self.on_prediction_ready(geometries)
            except Exception as exc:
                self.set_error(str(exc))

    def prediction_failed(self, token, error):
        if token == self.token:
            self.on_prediction_error(error)

    def restore_settings(self):
        saved = QSettings().value('CroplandWorkbench/config', '')
        if saved:
            try:
                values = json.loads(saved)
                for name in ('local_command', 'weights_path', 'service_url'):
                    getattr(self, name).setText(values.get(name, ''))
                for name in ('backend_combo', 'device_combo'):
                    getattr(self, name).setCurrentText(values.get(name, 'local' if name == 'backend_combo' else 'cuda'))
                self.autosave.setChecked(values.get('autosave', False))
                self.save_interval.setValue(values.get('save_interval', 5))
                for name in ('window_size', 'simplify_tolerance', 'min_area'):
                    if name in values:
                        getattr(self, name).setValue(values[name])
            except (ValueError, TypeError):
                pass

    def persist_settings(self):
        try:
            self.settings()
            values = {name: getattr(self, name).text() for name in ('local_command', 'weights_path', 'service_url')}
            values.update({name: getattr(self, name).currentText() for name in ('backend_combo', 'device_combo')})
            values.update({name: getattr(self, name).value() for name in ('window_size', 'simplify_tolerance', 'min_area', 'save_interval')})
            values['autosave'] = self.autosave.isChecked()
            QSettings().setValue('CroplandWorkbench/config', json.dumps(values))
            self.worker.stop()
            self.invalidate_prompts()
            self.set_status('设置已保存。')
        except Exception as exc:
            self.set_error(str(exc))

    def update_save_timer(self, *args):
        self.save_timer.stop()
        if self.autosave.isChecked():
            self.save_timer.start(self.save_interval.value() * 60000)

    def auto_save(self):
        layer = self.output_layer
        if layer and layer.isEditable() and layer.isModified() and not self.split_pieces:
            self.save_layer()

    def bind_layer(self, layer):
        old = self.output_layer
        if old:
            for signal, slot in [(old.featureAdded, self.refresh_parcels), (old.featureDeleted, self.refresh_parcels),
                                 (old.geometryChanged, self.refresh_parcels), (old.selectionChanged, self.sync_selection)]:
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self.cancel_split()
        self.output_layer = layer
        if layer:
            for signal, slot in [(layer.featureAdded, self.refresh_parcels), (layer.featureDeleted, self.refresh_parcels),
                                 (layer.geometryChanged, self.refresh_parcels), (layer.selectionChanged, self.sync_selection)]:
                signal.connect(slot)
        self.refresh_parcels()

    def edit_layer(self):
        layer = self.vector_combo.currentLayer()
        if not layer:
            raster = self.raster_combo.currentLayer()
            path = self.output_path.text().strip()
            if not path or not raster:
                raise ValueError('请选择标注图层，或指定影像与输出 SHP。')
            layer = ensure_shapefile(path, raster.crs())
            self.vector_combo.setLayer(layer)
        if layer.geometryType() != QgsWkbTypes.PolygonGeometry:
            raise ValueError('标注图层必须是面图层。')
        self.output_layer = layer
        if not layer.isEditable() and not layer.startEditing():
            raise ValueError('无法编辑该图层。')
        for field, expression in [('parcel_id', 'uuid()'), ('label', "'" + self.category.currentText().replace("'", "''") + "'")]:
            index = layer.fields().indexFromName(field)
            if index >= 0:
                layer.setDefaultValueDefinition(index, QgsDefaultValue(expression))
        return layer

    def native_tool(self, action):
        try:
            layer = self.edit_layer()
            self.iface.setActiveLayer(layer)
            getattr(self.iface, action)().trigger()
        except Exception as exc:
            self.set_error(str(exc))

    def start_split(self):
        self.start_line_edit('split')

    def sync_tool_buttons(self, *args):
        for button, tool in self.mode_buttons:
            button.setChecked(self.canvas.mapTool() == tool)
        for mode in ('trim', 'split'):
            button = getattr(self, mode + '_main_btn', None)
            if button:
                button.setChecked(self.canvas.mapTool() == self.split_tool and self.line_mode == mode)
        if hasattr(self, 'sam_main_btn'):
            self.sam_main_btn.setChecked(self.canvas.mapTool() == self.interactive_tool)
        if hasattr(self, 'tile_btn'):
            self.tile_btn.setChecked(self.canvas.mapTool() == self.tile_tool)

    def update_preview_opacity(self, value):
        for band in self.preview_bands:
            band.setFillColor(QColor(30, 180, 80, round(value * 2.55)))
            band.update()

    def filter_parcels(self, *args):
        query = self.parcel_search.text().strip().casefold()
        for i in range(self.parcel_list.count()):
            item = self.parcel_list.item(i)
            item.setHidden(query not in item.text().casefold())

    def zoom_to_parcel(self, item):
        layer = self.output_layer
        if not layer or not layer.isValid():
            return
        geometry = QgsGeometry(layer.getFeature(item.data(Qt.UserRole)).geometry())
        if geometry.isEmpty():
            return
        try:
            geometry.transform(QgsCoordinateTransform(layer.crs(),
                self.canvas.mapSettings().destinationCrs(), QgsProject.instance()))
            bounds = geometry.boundingBox()
            bounds.scale(1.5)
            self.canvas.setExtent(bounds)
            self.canvas.refresh()
        except Exception as exc:
            self.set_error(str(exc))

    def refresh_parcels(self, *args):
        self.parcel_list.blockSignals(True)
        self.parcel_list.clear()
        layer = self.output_layer
        if layer and layer.isValid():
            from qgis.PyQt.QtWidgets import QListWidgetItem
            selected = set(layer.selectedFeatureIds())
            for i, feature in enumerate(layer.getFeatures()):
                if i >= 2000:
                    break
                label = feature['label'] if 'label' in feature.fields().names() else ''
                parcel_id = str(feature['parcel_id'])[:8] if 'parcel_id' in feature.fields().names() else str(feature.id())
                item = QListWidgetItem(f'{parcel_id}   {label}')
                item.setData(Qt.UserRole, feature.id())
                self.parcel_list.addItem(item)
                item.setSelected(feature.id() in selected)
        self.parcel_list.blockSignals(False)
        total = layer.featureCount() if layer and layer.isValid() else 0
        suffix = '（显示前 2000 个）' if total > 2000 else ''
        self.parcel_count.setText(f'{total} 个地块{suffix}')
        self.filter_parcels()

    def sync_selection(self, *args):
        if not self.output_layer:
            return
        ids = set(self.output_layer.selectedFeatureIds())
        self.parcel_list.blockSignals(True)
        for i in range(self.parcel_list.count()):
            item = self.parcel_list.item(i)
            item.setSelected(item.data(Qt.UserRole) in ids)
        self.parcel_list.blockSignals(False)

    def select_parcels(self):
        if self.output_layer:
            self.output_layer.selectByIds([item.data(Qt.UserRole) for item in self.parcel_list.selectedItems()])

    def merge_selected(self):
        try:
            layer = self.edit_layer()
            features = list(layer.getSelectedFeatures())
            if len(features) < 2:
                raise ValueError('请选择至少两个地块。')
            geometry = merge_geometries([f.geometry() for f in features])
            if geometry.isEmpty() or not geometry.isGeosValid():
                raise ValueError('合并结果无效。')
            layer.beginEditCommand('Merge parcels')
            try:
                if not layer.changeGeometry(features[0].id(), geometry):
                    raise ValueError('合并失败。')
                for feature in features[1:]:
                    if not layer.deleteFeature(feature.id()):
                        raise ValueError('删除原地块失败。')
                layer.endEditCommand()
            except Exception:
                layer.destroyEditCommand()
                raise
            layer.selectByIds([features[0].id()])
            layer.triggerRepaint()
            self.refresh_parcels()
            self.set_status('已合并，保留首个地块属性；可撤销。')
        except Exception as exc:
            self.set_error(str(exc))

    def delete_selected(self):
        try:
            layer = self.edit_layer()
            ids = layer.selectedFeatureIds()
            if not ids:
                raise ValueError('请选择要删除的地块。')
            layer.beginEditCommand('Delete parcels')
            if not layer.deleteFeatures(ids):
                layer.destroyEditCommand()
                raise ValueError('删除失败。')
            layer.endEditCommand()
            layer.triggerRepaint()
            self.refresh_parcels()
            self.set_status('已删除，可撤销。')
        except Exception as exc:
            self.set_error(str(exc))

    def undo(self):
        if self.output_layer and self.output_layer.isEditable():
            self.cancel_split()
            self.output_layer.undoStack().undo()
            self.output_layer.triggerRepaint()
            self.refresh_parcels()

    def redo(self):
        if self.output_layer and self.output_layer.isEditable():
            self.cancel_split()
            self.output_layer.undoStack().redo()
            self.output_layer.triggerRepaint()
            self.refresh_parcels()

    def area_m2(self, geometry, crs):
        measure = QgsDistanceArea()
        measure.setSourceCrs(crs, QgsProject.instance().transformContext())
        measure.setEllipsoid('WGS84')
        return measure.measureArea(geometry)

    def save_layer(self):
        if self.split_pieces or self.split_tool.points:
            self.set_error('请先右键确认地块，再保存。当前修改仍保留。')
            return
        try:
            layer = self.output_layer
            if not layer or not layer.isEditable():
                self.set_status('没有待保存的标注编辑。')
                return
            for feature in layer.getFeatures():
                geometry = feature.geometry()
                if geometry.isEmpty() or not geometry.isGeosValid():
                    raise ValueError(f'地块 {feature.id()} 几何无效，请修正后保存。')
            layer.beginEditCommand('Refresh parcel attributes')
            try:
                for feature in layer.getFeatures():
                    values = {}
                    area_index = layer.fields().indexFromName('area_m2')
                    id_index = layer.fields().indexFromName('parcel_id')
                    if area_index >= 0:
                        values[area_index] = self.area_m2(feature.geometry(), layer.crs())
                    if id_index >= 0 and (feature['parcel_id'] is None or str(feature['parcel_id']) in ('', 'NULL')):
                        values[id_index] = str(uuid4())
                    if values and not layer.changeAttributeValues(feature.id(), values):
                        raise ValueError('地块属性更新失败。')
                layer.endEditCommand()
            except Exception:
                layer.destroyEditCommand()
                raise
            if not layer.commitChanges(False):
                raise ValueError('; '.join(layer.commitErrors()))
            self.cancel_split()
            self.refresh_parcels()
            layer.triggerRepaint()
            self.set_status('已保存 SHP；撤销历史从此保存点重新开始。')
        except Exception as exc:
            self.set_error(str(exc))

    def set_status(self, message):
        self.status_label.setStyleSheet('')
        self.status_label.setText(message)
        if hasattr(self, 'main_ui'):
            self.status_label.setVisible(True)
            self.sync_segmentation_ui()

    def set_error(self, message):
        self.status_label.setStyleSheet('color: #b00020;')
        self.status_label.setText(message)
        self.status_label.show()

    def closeEvent(self, event):
        self.end_segmentation()
        self.advanced_dialog.close()
        self.token += 1
        self.debounce.stop()
        self.worker.stop()
        self.prompt_state.clear()
        self.prompt_overlay.clear()
        self.clear_prediction_preview()
        self.cancel_split()
        if self.canvas.mapTool() in (self.positive_tool, self.negative_tool, self.box_tool, self.split_tool, self.tile_tool):
            self.canvas.unsetMapTool(self.canvas.mapTool())
        super().closeEvent(event)

    def shutdown(self):
        self.save_timer.stop()
        self.debounce.stop()
        self.worker.stop()
        self.bind_layer(None)
        for signal, slot in [(QgsProject.instance().layersAdded, self.refresh_raster_layers),
                             (QgsProject.instance().layersRemoved, self.refresh_raster_layers),
                             (self.canvas.mapToolSet, self.sync_tool_buttons),
                             (self.canvas.destinationCrsChanged, self.invalidate_prompts)]:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
