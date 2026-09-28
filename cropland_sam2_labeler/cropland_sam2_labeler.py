from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction

from .dock_widget import CroplandSam2DockWidget


class CroplandSam2LabelerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.action = None
        self.dock = None
        self.extra_actions = []

    def initGui(self):
        self.action = QAction("Cropland SAM2 Labeler", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toggle_dock)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu("&Cropland SAM2 Labeler", self.action)
        for title, settings in [('Cropland SAM2: Settings', True), ('Cropland SAM2: Vector Editing', False)]:
            action = QAction(title, self.iface.mainWindow())
            from qgis.core import QgsApplication
            action.setIcon(QgsApplication.getThemeIcon('/mActionMapSettings.svg' if settings else '/mActionVertexTool.svg'))
            action.triggered.connect(lambda checked=False, settings=settings: self.show_advanced(settings))
            self.iface.addToolBarIcon(action)
            self.iface.addPluginToMenu('&Cropland SAM2 Labeler', action)
            self.extra_actions.append(action)

    def show_advanced(self, settings):
        self.toggle_dock(True)
        from .workbench_ui import show_advanced
        show_advanced(self.dock, settings)

    def unload(self):
        for action in self.extra_actions:
            self.iface.removeToolBarIcon(action)
            self.iface.removePluginMenu('&Cropland SAM2 Labeler', action)
        self.extra_actions.clear()
        if self.dock is not None:
            self.dock.shutdown()
            self.iface.removeDockWidget(self.dock)
            self.dock.close()
            self.dock.deleteLater()
            self.dock = None
        if self.action is not None:
            self.iface.removePluginMenu("&Cropland SAM2 Labeler", self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None

    def toggle_dock(self, checked):
        if checked:
            if self.dock is None:
                self.dock = CroplandSam2DockWidget(self.iface)
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
                self.dock.visibilityChanged.connect(self.action.setChecked)
            self.dock.show()
            self.dock.raise_()
        elif self.dock is not None:
            self.dock.hide()
