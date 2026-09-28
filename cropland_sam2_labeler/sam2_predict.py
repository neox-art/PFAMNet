"""Local SAM2 adapter. Run in the existing field environment, not QGIS Python."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform
from rasterio.windows import Window
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_predictor = None
_model_key = None
_image_key = None
_window_cache = None


def predict(payload):
    global _predictor, _model_key, _image_key, _window_cache
    device = payload.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the selected Python environment.')
    checkpoint = Path(payload['weights_path'])
    configs = {
        'sam2.1_hiera_tiny.pt': 'configs/sam2.1/sam2.1_hiera_t.yaml',
        'sam2.1_hiera_small.pt': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'sam2.1_hiera_base_plus.pt': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    }
    config = configs.get(checkpoint.name)
    if not checkpoint.is_file() or config is None:
        raise ValueError('Choose an existing official SAM2.1 checkpoint with its original filename.')
    prompts = payload['prompts']
    points = prompts.get('positive_points', []) + prompts.get('negative_points', [])
    boxes = prompts.get('boxes', [])
    if not prompts.get('positive_points') and not boxes:
        raise ValueError('Add a positive point or a box.')
    with rasterio.open(payload['raster_path']) as src:
        if src.crs is None:
            raise ValueError('The imagery must have a defined CRS.')
        prompt_crs = payload.get('prompt_crs') or payload.get('crs_authid') or src.crs

        def pixel(x, y):
            xs, ys = transform(prompt_crs, src.crs, [x], [y])
            return (~src.transform) * (xs[0], ys[0])

        point_pixels = [pixel(p['x'], p['y']) for p in points]
        box_pixels = []
        for b in boxes:
            corners = np.asarray([pixel(x, y) for x in (b['xmin'], b['xmax'])
                                  for y in (b['ymin'], b['ymax'])])
            box_pixels.append([*corners.min(axis=0), *corners.max(axis=0)])
        coords = point_pixels + [tuple(b[:2]) for b in box_pixels] + [tuple(b[2:]) for b in box_pixels]
        coords = np.asarray(coords)
        if np.any(coords < 0) or np.any(coords[:, 0] >= src.width) or np.any(coords[:, 1] >= src.height):
            raise ValueError('All prompts must be inside the imagery.')
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        if np.any(hi - lo > 1792):
            raise ValueError('Prompts span too much imagery. Label one smaller parcel at a time (1792 pixels maximum).')
        center = (lo + hi) / 2
        size = np.maximum(int(payload.get('window_size', 1024)), np.ceil(hi - lo + 256)).astype(int)
        x0, y0 = np.maximum(0, np.floor(center - size / 2)).astype(int)
        width, height = min(int(size[0]), src.width - x0), min(int(size[1]), src.height - y0)
        if src.width == 512 and src.height == 512:
            x0, y0, width, height = 0, 0, 512, 512
        raster_key = (payload['raster_path'], Path(payload['raster_path']).stat().st_mtime_ns,
                      payload.get('window_size', 1024))
        if _window_cache and _window_cache[0] == raster_key:
            cx, cy, cw, ch = _window_cache[1]
            if lo[0] >= cx and lo[1] >= cy and hi[0] < cx + cw and hi[1] < cy + ch:
                x0, y0, width, height = cx, cy, cw, ch
        _window_cache = (raster_key, (x0, y0, width, height))
        window = Window(int(x0), int(y0), int(width), int(height))
        bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        data = src.read(bands, window=window)
        valid = src.dataset_mask(window=window) > 0
        image = np.zeros(data.shape, dtype=np.uint8)
        for i, band in enumerate(data):
            good = valid & np.isfinite(band)
            if not good.any():
                raise ValueError('Selected imagery window has no valid pixels.')
            if band.dtype == np.uint8:
                image[i] = band
            else:
                low, high = np.percentile(band[good], [2, 98])
                image[i] = np.nan_to_num(np.clip((band.astype(float) - low) * 255 / max(high - low, 1e-6), 0, 255)).astype(np.uint8)
        image[:, ~valid] = 0
        geo_transform = src.window_transform(window)
        crs = src.crs.to_wkt()
    model_key = (str(checkpoint), checkpoint.stat().st_mtime_ns, device)
    if _model_key != model_key:
        model = build_sam2(config, str(checkpoint), device=device, apply_postprocessing=False)
        _predictor = SAM2ImagePredictor(model)
        _model_key, _image_key = model_key, None
    predictor = _predictor
    image_key = (payload['raster_path'], Path(payload['raster_path']).stat().st_mtime_ns,
                 x0, y0, width, height)
    point_coords = np.asarray(point_pixels, dtype=np.float32) - [x0, y0] if points else None
    labels = np.asarray([1] * len(prompts.get('positive_points', [])) +
                        [0] * len(prompts.get('negative_points', []))) if points else None
    combined = np.zeros(valid.shape, dtype=bool)
    with torch.inference_mode():
        cache_hit = _image_key == image_key
        if not cache_hit:
            predictor.set_image(image.transpose(1, 2, 0))
            _image_key = image_key
        for box in box_pixels or [None]:
            local_box = np.asarray(box) - [x0, y0, x0, y0] if box is not None else None
            masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=labels,
                                                  box=local_box, multimask_output=True)
            combined |= masks[int(np.argmax(scores))].astype(bool)
    combined &= valid
    features = [{'type': 'Feature', 'properties': {'source': 'sam2.1'}, 'geometry': geom}
                for geom, value in shapes(combined.astype(np.uint8), mask=combined, transform=geo_transform)
                if value == 1]
    if not features:
        raise ValueError('SAM2 returned an empty mask. Adjust the prompts.')
    # Coordinates are in the source raster CRS, as required by the plugin contract.
    return {'type': 'FeatureCollection', 'features': features, 'crs_wkt': crs,
            'device': device, 'image_cache_hit': cache_hit}


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        import contextlib
        for line in sys.stdin:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = predict(json.loads(line))
                print(json.dumps({'result': response}), flush=True)
            except Exception as exc:
                print(json.dumps({'error': str(exc)}), flush=True)
        sys.exit(0)
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    result = predict(request)
    Path(request['output_geojson']).write_text(json.dumps(result), encoding='utf-8')
    print('SAM2_OK', len(result['features']), result['device'])
