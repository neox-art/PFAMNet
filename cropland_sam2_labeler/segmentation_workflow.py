from copy import deepcopy
from qgis.PyQt.QtCore import Qt
from .prompt_state import PromptPoint, PromptBox


class SegmentationWorkflow:
    def sync_segmentation_ui(self, *args):
        if not hasattr(self, 'main_ui'):
            return
        ui = self.main_ui
        active = self.segmentation_active
        ui.startBtn.hide()
        ui.startBtn.setEnabled(bool(self.raster_combo.currentLayer() and
                                   self.vector_combo.currentLayer() and self.category.currentText()))
        for button in [ui.confirmBtn, ui.undoBtn, ui.redoBtn, ui.endBtn]:
            button.setVisible(active)
        ui.endBtn.setEnabled(active)
        ui.confirmBtn.setEnabled(active and bool(self.preview_geometries or self.split_pieces))
        ui.undoBtn.setEnabled(active and bool(self.prompt_history) and not self.split_pieces)
        ui.redoBtn.setEnabled(active and bool(self.prompt_future) and not self.split_pieces)
        self.raster_combo.setEnabled(not active)
        self.vector_combo.setEnabled(not active)
        ui.modelSelect.setEnabled(not active)

    def start_segmentation(self):
        raster = self.raster_combo.currentLayer()
        if raster and (raster.width() != 512 or raster.height() != 512):
            self.set_error('请先点击裁剪图标，从底图选取 512×512 影像块。')
            return
        if not (self.raster_combo.currentLayer() and self.vector_combo.currentLayer()
                and self.category.currentText()):
            self.set_error('请选择影像、输出图层和类别。')
            return
        try:
            self.settings()
        except Exception as exc:
            self.set_error(str(exc))
            return
        self.segmentation_active = True
        if not self.split_pieces:
            self.live_preview.setChecked(True)
        self.canvas.setMapTool(self.interactive_tool)
        self.sync_segmentation_ui()
        self.sync_tool_buttons()

    def end_segmentation(self):
        self.segmentation_active = False
        self.cancel_split()
        self.clear_prompts()
        if self.canvas.mapTool() in (self.interactive_tool, self.positive_tool,
                                    self.negative_tool, self.box_tool, self.tile_tool, self.split_tool):
            self.canvas.unsetMapTool(self.canvas.mapTool())
        if hasattr(self.iface, 'actionPan'):
            self.iface.actionPan().trigger()
        self.sync_segmentation_ui()

    def remember_prompt(self):
        if not hasattr(self, 'prompt_history') or self.restoring_prompt:
            return
        self.prompt_history.append(deepcopy(self.prompt_state.to_payload()))
        self.prompt_future.clear()
        self.sync_segmentation_ui()

    def restore_prompt_history(self):
        state = self.prompt_history[-1] if self.prompt_history else {}
        self.restoring_prompt = True
        try:
            self.prompt_state.clear()
            self.prompt_overlay.clear()
            for name, label in [('positive_points', 1), ('negative_points', 0)]:
                for value in state.get(name, []):
                    point = PromptPoint(**value)
                    getattr(self.prompt_state, name).append(point)
                    self.prompt_overlay.add_point(point, label == 1)
            for value in state.get('boxes', []):
                box = PromptBox(**value)
                self.prompt_state.boxes.append(box)
                self.prompt_overlay.add_box(box)
            self.update_prompt_label()
            self.prompt_changed()
        finally:
            self.restoring_prompt = False
            self.sync_segmentation_ui()

    def undo_prompt(self):
        if self.prompt_history:
            self.prompt_future.append(self.prompt_history.pop())
            self.restore_prompt_history()

    def redo_prompt(self):
        if self.prompt_future:
            self.prompt_history.append(self.prompt_future.pop())
            self.restore_prompt_history()
