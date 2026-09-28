"""Persistent SAM2 subprocess with a single pending request."""
import json
import os
import queue
import shlex
import subprocess
import threading
from types import SimpleNamespace
from qgis.PyQt.QtCore import QObject, pyqtSignal
from .sam2_backend import geometries_from_geojson


class PersistentWorker(QObject):
    ready = pyqtSignal(int, list)
    failed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = None
        self.command = None
        self.image_cache_hit = False

    def submit(self, token, settings, payload):
        start_thread = False
        command = shlex.split(settings.local_command) + ['--worker']
        if command != self.command or self.state is None or not self.state.thread.is_alive():
            self.stop()
            self.command = command
            env = os.environ.copy()
            for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QGIS_PREFIX_PATH',
                        'GDAL_DATA', 'GDAL_DRIVER_PATH', 'PROJ_LIB', 'PROJ_DATA'):
                env.pop(key, None)
            root = os.path.dirname(command[0])
            windows = env.get('SystemRoot', 'C:/Windows')
            env['PATH'] = os.pathsep.join([root, os.path.join(root, 'Library', 'bin'),
                os.path.join(root, 'Scripts'), os.path.join(windows, 'System32'), windows])
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            state = SimpleNamespace(queue=queue.Queue(maxsize=1), stop=threading.Event(),
                                    process=None, stderr='')
            state.thread = threading.Thread(target=self.run, args=(state, command, env), daemon=True)
            self.state = state
            start_thread = True
        item = (token, dict(payload, weights_path=settings.weights_path, device=settings.device))
        try:
            self.state.queue.get_nowait()
        except queue.Empty:
            pass
        self.state.queue.put_nowait(item)
        if start_thread:
            self.state.thread.start()

    def run(self, state, command, env):
        process = None
        token = None
        try:
            process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                errors='replace', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            state.process = process
            def drain():
                for line in process.stderr:
                    state.stderr = (state.stderr + line)[-4000:]
                process.stderr.close()
            threading.Thread(target=drain, daemon=True).start()
            while not state.stop.is_set():
                try:
                    token, payload = state.queue.get(timeout=.2)
                except queue.Empty:
                    continue
                timer = threading.Timer(180, process.kill)
                timer.daemon = True
                timer.start()
                try:
                    process.stdin.write(json.dumps(payload) + '\n')
                    process.stdin.flush()
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError(state.stderr or 'SAM2 process stopped or timed out.')
                    result = json.loads(line)
                    if 'error' in result:
                        self.failed.emit(token, result['error'])
                    else:
                        self.image_cache_hit = result['result'].get('image_cache_hit', False)
                        self.ready.emit(token, geometries_from_geojson(result['result']))
                finally:
                    timer.cancel()
        except Exception as exc:
            if not state.stop.is_set():
                if token is not None:
                    self.failed.emit(token, str(exc))
                try:
                    pending = state.queue.get_nowait()
                    self.failed.emit(pending[0], str(exc))
                except queue.Empty:
                    pass
        finally:
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdin.close()
                process.stdout.close()

    def stop(self):
        if self.state:
            self.state.stop.set()
            if self.state.process and self.state.process.poll() is None:
                self.state.process.kill()
            self.state.thread.join(timeout=3)
            self.state = None
