"""Native QGIS controls for the parcel annotation dock."""
from pathlib import Path

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QPlainTextEdit, QPushButton, QSizePolicy, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)
from qgis.core import QgsApplication, QgsProject, QgsVectorLayer

from .advanced_ui import build_advanced_ui


def icon_button(parent, name, icon, callback, checkable=False):
    button = QToolButton(parent)
    button.setIcon(QgsApplication.getThemeIcon(icon))
    button.setToolTip(name)
    button.setAccessibleName(name)
    button.setFixedSize(30, 28)
    button.setCheckable(checkable)
    button.clicked.connect(callback)
    return button


class ParcelPanel(QWidget):
    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName('parcelPanel')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setVerticalSpacing(8)
        layout.addLayout(form)
        self.modelSelect = QComboBox(self)
        self.modelDescription = QPlainTextEdit(self)
        self.modelDescription.setReadOnly(True)
        self.modelDescription.setFixedHeight(68)
        self.modelDescription.setFocusPolicy(Qt.NoFocus)
        form.addRow('Model', self.modelSelect)
        form.addRow(self.modelDescription)
        # Keep the existing controls and their configuration/layer signal connections.
        for label, control in [('Raster', dock.raster_combo), ('Output', dock.vector_combo),
                               ('Opacity', dock.preview_opacity)]:
            control.setParent(self)
            control.setMinimumWidth(0)
            control.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            form.addRow(label, control)
        dock.category.setEditable(True)
        dock.category.setParent(self)
        dock.category.setMinimumWidth(0)
        dock.category.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.class_edit = icon_button(self, 'Edit classes', '/mActionToggleEditing.svg', self.edit_classes)
        classes = QHBoxLayout()
        classes.addWidget(dock.category, 1)
        classes.addWidget(self.class_edit)
        form.addRow('Class', classes)
        dock.preview_opacity.setValue(50)
        dock.vector_combo.setAdditionalItems(['Create New'])
        dock.vector_combo.activated.connect(self.create_output)
        modes = QHBoxLayout()
        modes.setSpacing(4)
        for attribute, title, icon, callback, checkable in [
            ('tile_btn', 'Crop 512 x 512', '/mActionAddRasterLayer.svg', dock.start_tile_selection, True),
            ('sam_main_btn', 'SAM2 extraction', '/mActionSelect.svg', dock.start_segmentation, True),
            ('trim_main_btn', 'Reshape boundary', '/mActionReshape.svg', lambda: dock.start_line_edit('trim'), True),
            ('split_main_btn', 'Split parcel', '/mActionSplitFeatures.svg', lambda: dock.start_line_edit('split'), True),
            ('save_main_btn', 'Save parcels', '/mActionSaveEdits.svg', dock.save_layer, False),
        ]:
            button = icon_button(self, title, icon, callback, checkable)
            button.clicked.connect(dock.sync_tool_buttons)
            modes.addWidget(button)
            setattr(dock, attribute, button)
        modes.addStretch()
        layout.addLayout(modes)
        self.startBtn = QPushButton('Start', self)
        self.startBtn.clicked.connect(dock.start_segmentation)
        self.startBtn.hide()
        self.confirmBtn = dock.accept_btn
        self.confirmBtn.setParent(self)
        self.confirmBtn.setText('Confirm')
        self.confirmBtn.setShortcut(Qt.Key_Return)
        self.confirmBtn.setMinimumHeight(28)
        layout.addWidget(self.confirmBtn)
        history = QHBoxLayout()
        self.undoBtn = icon_button(self, 'Undo prompt', '/mActionUndo.svg', dock.undo_prompt)
        self.redoBtn = icon_button(self, 'Redo prompt', '/mActionRedo.svg', dock.redo_prompt)
        self.undoBtn.setShortcut(Qt.Key_Left)
        self.redoBtn.setShortcut(Qt.Key_Right)
        history.addWidget(self.undoBtn)
        history.addWidget(self.redoBtn)
        history.addStretch()
        self.endBtn = QPushButton('End', self)
        self.endBtn.setShortcut(Qt.Key_Escape)
        self.endBtn.clicked.connect(dock.end_segmentation)
        history.addWidget(self.endBtn)
        layout.addLayout(history)
        dock.status_label.setParent(self)
        layout.addWidget(dock.status_label)
        dock.status_label.hide()
        layout.addStretch(1)
        footer = QHBoxLayout()
        footer.addStretch()
        self.settings_button = icon_button(
            self, 'Settings', '/mActionMapSettings.svg', lambda: show_advanced(dock, True))
        footer.addWidget(self.settings_button)
        layout.addLayout(footer)
        self.modelSelect.setMinimumWidth(0)
        self.modelSelect.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for signal in (dock.weights_path.textChanged, dock.backend_combo.currentTextChanged,
                       dock.device_combo.currentTextChanged):
            signal.connect(self.refresh_model)
        self.refresh_model()

    def refresh_model(self):
        dock = self.dock
        backend = dock.backend_combo.currentText()
        model = Path(dock.weights_path.text()).stem or 'SAM2'
        self.modelSelect.blockSignals(True)
        self.modelSelect.clear()
        self.modelSelect.addItem(model if backend == 'local' else backend.upper())
        self.modelSelect.blockSignals(False)
        self.modelDescription.setPlainText(f'{model}\n{backend} / {dock.device_combo.currentText().upper()}')

    def edit_classes(self):
        combo = self.dock.category
        current = combo.currentText().strip()
        labels = list(dict.fromkeys([combo.itemText(i) for i in range(combo.count())] + [current]))
        text, accepted = QInputDialog.getMultiLineText(self, 'Classes', 'Labels', '\n'.join(filter(None, labels)))
        if not accepted:
            return
        labels = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
        if not labels:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(labels)
        combo.setCurrentText(current if current in labels else labels[0])
        combo.blockSignals(False)
        combo.currentTextChanged.emit(combo.currentText())

    def create_output(self):
        dock = self.dock
        if dock.vector_combo.currentText() != 'Create New':
            return
        raster = dock.raster_combo.currentLayer()
        crs = raster.crs() if raster else dock.canvas.mapSettings().destinationCrs()
        layer = QgsVectorLayer('MultiPolygon?field=parcel_id:string&field=label:string', 'Segmentation Results', 'memory')
        layer.setCrs(crs)
        QgsProject.instance().addMapLayer(layer)
        dock.vector_combo.setLayer(layer)


def build_ui(d):
    build_advanced_ui(d)
    d.advanced_dialog = QDialog(d)
    d.advanced_dialog.setWindowTitle('Parcel settings and vector editing')
    d.advanced_dialog.resize(420, 720)
    advanced = d.widget()
    advanced.setParent(d.advanced_dialog)
    advanced.setStyleSheet('')
    advanced.findChild(QLabel, 'brand').hide()
    d.model_summary.hide()
    QVBoxLayout(d.advanced_dialog).addWidget(advanced)
    d.advanced_dialog.hide()
    d.main_ui = ParcelPanel(d)
    d.setWidget(d.main_ui)
    d.setMinimumWidth(253)
    d.setWindowTitle('Cropland SAM2 Options')
    d.segmentation_active = False
    d.prompt_history = []
    d.prompt_future = []
    d.restoring_prompt = False
    d.sync_segmentation_ui()
    for signal in (d.raster_combo.layerChanged, d.vector_combo.layerChanged, d.category.currentTextChanged):
        signal.connect(d.sync_segmentation_ui)


def show_advanced(d, settings=False):
    d.advanced_dialog.findChild(QTabWidget).setCurrentIndex(2 if settings else 0)
    d.advanced_dialog.show()
    d.advanced_dialog.raise_()
