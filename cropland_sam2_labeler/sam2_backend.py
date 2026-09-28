import json
import os
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.request

from qgis.PyQt.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from qgis.core import QgsGeometry, QgsPointXY, QgsRectangle


class BackendError(RuntimeError):
    pass


class BackendSignals(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)


class BackendJob(QRunnable):
    def __init__(self, backend, payload):
        super().__init__()
        self.backend = backend
        self.payload = payload
        self.signals = BackendSignals()

    @pyqtSlot()
    def run(self):
        try:
            geometries = self.backend.predict(self.payload)
            self.signals.finished.emit(geometries)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class Sam2BackendRunner:
    def __init__(self):
        self.pool = QThreadPool.globalInstance()

    def submit(self, backend, payload, on_success, on_error):
        job = BackendJob(backend, payload)
        job.signals.finished.connect(on_success)
        job.signals.failed.connect(on_error)
        self.pool.start(job)


def backend_from_settings(settings):
    if settings.backend == "mock":
        return MockSam2Backend(settings)
    if settings.backend == "http":
        return HttpSam2Backend(settings)
    if settings.backend == "local":
        return LocalProcessSam2Backend(settings)
    raise BackendError(f"Unsupported backend: {settings.backend}")


class MockSam2Backend:
    def __init__(self, settings):
        self.settings = settings

    def predict(self, payload):
        prompts = payload["prompts"]
        rectangles = []
        for box in prompts.get("boxes", []):
            rectangles.append(QgsRectangle(box["xmin"], box["ymin"], box["xmax"], box["ymax"]))
        for point in prompts.get("positive_points", []):
            size = max(float(payload.get("map_units_per_pixel", 1.0)) * 96.0, 5.0)
            rectangles.append(
                QgsRectangle(point["x"] - size, point["y"] - size, point["x"] + size, point["y"] + size)
            )
        if not rectangles:
            raise BackendError("Add at least one positive point or box before prediction.")
        geometries = [QgsGeometry.fromRect(rect) for rect in rectangles]
        merged = QgsGeometry.unaryUnion(geometries) if len(geometries) > 1 else geometries[0]
        return [merged]


class HttpSam2Backend:
    def __init__(self, settings):
        self.settings = settings

    def predict(self, payload):
        request_payload = dict(payload)
        request_payload["weights_path"] = self.settings.weights_path
        request_payload["device"] = self.settings.device
        data = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            self.settings.service_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise BackendError(f"SAM2 service request failed: {exc}") from exc
        return geometries_from_geojson(json.loads(body))


class LocalProcessSam2Backend:
    def __init__(self, settings):
        self.settings = settings

    def predict(self, payload):
        with tempfile.TemporaryDirectory(prefix="cropland_sam2_") as tmpdir:
            input_path = os.path.join(tmpdir, "request.json")
            output_path = os.path.join(tmpdir, "response.geojson")
            request_payload = dict(payload)
            request_payload["weights_path"] = self.settings.weights_path
            request_payload["device"] = self.settings.device
            request_payload["output_geojson"] = output_path
            with open(input_path, "w", encoding="utf-8") as handle:
                json.dump(request_payload, handle, ensure_ascii=False)
            command = shlex.split(self.settings.local_command)
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QGIS_PREFIX_PATH',
                        'GDAL_DATA', 'GDAL_DRIVER_PATH', 'PROJ_LIB', 'PROJ_DATA'):
                env.pop(key, None)
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            python_root = os.path.dirname(command[0])
            env['PATH'] = os.pathsep.join([python_root, os.path.join(python_root, 'Library', 'bin'),
                                         os.path.join(python_root, 'Scripts'), env.get('PATH', '')])
            result = subprocess.run(
                command + [input_path],
                cwd=tmpdir,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                capture_output=True,
                check=False,
                timeout=300,
            )
            if result.returncode != 0:
                raise BackendError(result.stderr.strip() or result.stdout.strip() or "Local SAM2 command failed.")
            if not os.path.exists(output_path):
                raise BackendError("Local SAM2 command did not write response.geojson.")
            with open(output_path, "r", encoding="utf-8") as handle:
                return geometries_from_geojson(json.load(handle))


def geometries_from_geojson(payload):
    features = payload.get("features", [])
    geometries = []
    for feature in features:
        geometry_payload = feature.get("geometry")
        if not geometry_payload:
            continue
        geometry = geometry_from_geojson_payload(geometry_payload)
        if geometry and not geometry.isEmpty():
            geometries.append(geometry)
    if not geometries:
        raise BackendError("SAM2 backend returned no polygons.")
    return geometries


def geometry_from_geojson_payload(geometry_payload):
    geometry_type = geometry_payload.get("type")
    coordinates = geometry_payload.get("coordinates")
    if geometry_type == "Polygon":
        return QgsGeometry.fromPolygonXY(_polygon_to_qgis(coordinates))
    if geometry_type == "MultiPolygon":
        return QgsGeometry.fromMultiPolygonXY([_polygon_to_qgis(poly) for poly in coordinates])
    raise BackendError(f"Unsupported GeoJSON geometry type: {geometry_type}")


def _polygon_to_qgis(coordinates):
    rings = []
    for ring in coordinates:
        rings.append([QgsPointXY(float(x), float(y)) for x, y in ring])
    return rings
