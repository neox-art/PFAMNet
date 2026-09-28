from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis, QgsGeometry, QgsPointXY, QgsRectangle
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker

from .prompt_state import PromptBox, PromptPoint


class PromptPointTool(QgsMapTool):
    pointAdded = pyqtSignal(float, float, int)

    def __init__(self, canvas, label):
        super().__init__(canvas)
        self.canvas = canvas
        self.label = label
        self.setCursor(Qt.CrossCursor)

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        point = self.toMapCoordinates(event.pos())
        self.pointAdded.emit(point.x(), point.y(), self.label)


class InteractivePromptTool(PromptPointTool):
    """Left click adds a prompt; right click accepts the current preview."""
    confirmed = pyqtSignal()
    def __init__(self, canvas):
        super().__init__(canvas, 1)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.confirmed.emit()
        elif event.button() == Qt.LeftButton:
            point = self.toMapCoordinates(event.pos())
            self.pointAdded.emit(point.x(), point.y(), 1)


class BoxPromptTool(QgsMapTool):
    boxAdded = pyqtSignal(float, float, float, float)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.start_point = None
        self.rubber = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self.rubber.setColor(QColor(0, 160, 220, 90))
        self.rubber.setStrokeColor(QColor(0, 120, 200))
        self.rubber.setWidth(2)
        self.setCursor(Qt.CrossCursor)

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = self.toMapCoordinates(event.pos())

    def canvasMoveEvent(self, event):
        if not self.start_point:
            return
        current = self.toMapCoordinates(event.pos())
        self._show_rect(self.start_point, current)

    def canvasReleaseEvent(self, event):
        if not self.start_point:
            return
        end_point = self.toMapCoordinates(event.pos())
        rect = QgsRectangle(self.start_point, end_point)
        self.start_point = None
        self.rubber.reset(Qgis.GeometryType.Polygon)
        if rect.width() > 0 and rect.height() > 0:
            self.boxAdded.emit(rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum())

    def _show_rect(self, p1, p2):
        rect = QgsRectangle(p1, p2)
        self.rubber.setToGeometry(QgsGeometry.fromRect(rect), None)

    def deactivate(self):
        self.start_point = None
        self.rubber.reset(Qgis.GeometryType.Polygon)
        super().deactivate()


class SplitLineTool(QgsMapTool):
    lineFinished = pyqtSignal(object)
    confirmed = pyqtSignal()
    cancelled = pyqtSignal()
    drawingStarted = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        self.rubber = QgsRubberBand(canvas, Qgis.GeometryType.Line)
        self.rubber.setColor(QColor(255, 128, 0))
        self.rubber.setWidth(3)
        self.setCursor(Qt.CrossCursor)

    def canvasReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            if self.points:
                end = QgsPointXY(self.toMapCoordinates(event.pos()))
                if end != self.points[-1]:
                    self.points.append(end)
            if len(self.points) >= 2:
                line = QgsGeometry.fromPolylineXY(self.points)
                self.reset()
                self.lineFinished.emit(line)
            elif not self.points:
                self.confirmed.emit()
            return
        if event.button() != Qt.LeftButton:
            return
        if not self.points:
            self.drawingStarted.emit()
        point = self.toMapCoordinates(event.pos())
        self.points.append(QgsPointXY(point))
        self.rubber.addPoint(point)

    def canvasMoveEvent(self, event):
        if self.points:
            points = self.points + [QgsPointXY(self.toMapCoordinates(event.pos()))]
            self.rubber.setToGeometry(QgsGeometry.fromPolylineXY(points), None)

    def reset(self):
        self.points.clear()
        self.rubber.reset(Qgis.GeometryType.Line)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reset()
            self.cancelled.emit()
        elif event.key() == Qt.Key_Backspace and self.points:
            self.points.pop()
            self.rubber.reset(Qgis.GeometryType.Line)
            for point in self.points:
                self.rubber.addPoint(point)

    def deactivate(self):
        self.reset()
        super().deactivate()


class PromptOverlay:
    def __init__(self, canvas):
        self.canvas = canvas
        self.markers = []
        self.boxes = []

    def add_point(self, point, positive=True):
        marker = QgsVertexMarker(self.canvas)
        marker.setCenter(QgsPointXY(point.x, point.y))
        marker.setIconType(QgsVertexMarker.ICON_CIRCLE)
        marker.setIconSize(10)
        marker.setPenWidth(2)
        marker.setColor(QColor(25, 170, 75) if positive else QColor(220, 65, 65))
        self.markers.append(marker)

    def add_box(self, box):
        rubber = QgsRubberBand(self.canvas, Qgis.GeometryType.Polygon)
        rubber.setColor(QColor(0, 160, 220, 45))
        rubber.setStrokeColor(QColor(0, 120, 200))
        rubber.setWidth(2)
        rubber.setToGeometry(
            QgsGeometry.fromRect(QgsRectangle(box.xmin, box.ymin, box.xmax, box.ymax)),
            None,
        )
        self.boxes.append(rubber)

    def clear(self):
        for marker in self.markers:
            self.canvas.scene().removeItem(marker)
        for rubber in self.boxes:
            self.canvas.scene().removeItem(rubber)
        self.markers.clear()
        self.boxes.clear()
