from qgis.core import QgsGeometry, QgsWkbTypes


def clean_geometry(geometry, simplify_tolerance=0.0, min_area=0.0):
    if geometry is None or geometry.isEmpty():
        return QgsGeometry()
    cleaned = QgsGeometry(geometry)
    if simplify_tolerance > 0:
        cleaned = cleaned.simplify(simplify_tolerance)
    cleaned = cleaned.makeValid()
    cleaned = polygon_only(cleaned)
    if min_area > 0:
        cleaned = remove_small_parts(cleaned, min_area)
    return cleaned.makeValid() if cleaned and not cleaned.isEmpty() else QgsGeometry()


def polygon_only(geometry):
    if geometry is None or geometry.isEmpty():
        return QgsGeometry()
    flat_type = QgsWkbTypes.flatType(geometry.wkbType())
    if flat_type in {QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon}:
        return geometry
    parts = []
    for part in geometry.constParts():
        part_geometry = QgsGeometry(part.clone())
        if QgsWkbTypes.geometryType(part_geometry.wkbType()) == QgsWkbTypes.PolygonGeometry:
            parts.append(part_geometry)
    if not parts:
        return QgsGeometry()
    return QgsGeometry.unaryUnion(parts)


def remove_small_parts(geometry, min_area):
    if geometry is None or geometry.isEmpty():
        return QgsGeometry()
    if not geometry.isMultipart():
        return geometry if geometry.area() >= min_area else QgsGeometry()
    kept = []
    for part in geometry.asGeometryCollection():
        if part.area() >= min_area:
            kept.append(part)
    if not kept:
        return QgsGeometry()
    return QgsGeometry.unaryUnion(kept)


def split_geometry_by_line(geometry, line_geometry):
    if geometry is None or geometry.isEmpty():
        raise ValueError("No polygon geometry selected.")
    if line_geometry is None or line_geometry.isEmpty():
        raise ValueError("Draw a split line first.")
    original = polygon_only(QgsGeometry(geometry).makeValid())
    points = []
    for point in line_geometry.asPolyline():
        if not points or point != points[-1]:
            points.append(point)
    if len(points) < 2:
        raise ValueError('切割线至少需要两个不同位置。')
    line_geometry = QgsGeometry.fromPolylineXY(points)
    if not original.intersects(line_geometry):
        raise ValueError('切割线未经过目标地块，请在地块内画线。')
    geometry = QgsGeometry(original)
    result, new_geometries, _ = geometry.splitGeometry(points, False)
    # Interior endpoints cannot split a polygon. Extend only these endpoints,
    # along their existing segment directions, beyond the polygon's bounds.
    if result != 0:
        bounds = original.boundingBox()
        distance = 2 * (bounds.width() + bounds.height())
        start = original.contains(QgsGeometry.fromPointXY(points[0]))
        end = original.contains(QgsGeometry.fromPointXY(points[-1]))
        if start or end:
            extended = line_geometry.extendLine(distance if start else 0, distance if end else 0)
            geometry = QgsGeometry(original)
            result, new_geometries, _ = geometry.splitGeometry(extended.asPolyline(), False)
    if result != 0:
        raise ValueError('这条线没有将地块分开，请让线经过地块内部而不是沿边界画线。')
    pieces = [geometry] + new_geometries
    valid_pieces = [piece.makeValid() for piece in pieces if piece and not piece.isEmpty()]
    if len(valid_pieces) < 2:
        raise ValueError("Split did not create multiple polygon pieces.")
    return valid_pieces


def merge_geometries(geometries):
    non_empty = [g for g in geometries if g and not g.isEmpty()]
    if not non_empty:
        return QgsGeometry()
    return QgsGeometry.unaryUnion(non_empty).makeValid()


def reshape_boundary(geometry, line_geometry, area=None, snap_tolerance=0):
    """Use QGIS boundary replacement for both outward and inward corrections."""
    original = polygon_only(QgsGeometry(geometry).makeValid())
    line = QgsGeometry(line_geometry)
    points = line.asPolyline()
    if len(points) < 2:
        raise ValueError('修边线至少需要两个不同位置。')
    if snap_tolerance > 0:
        boundary = original.convertToType(QgsWkbTypes.LineGeometry)
        for index in (0, len(points)-1):
            point = QgsGeometry.fromPointXY(points[index])
            nearest = boundary.nearestPoint(point)
            if not nearest.isEmpty() and point.distance(nearest) <= snap_tolerance:
                points[index] = nearest.asPoint()
        line = QgsGeometry.fromPolylineXY(points)
    if line.length() == 0 or not line.isSimple():
        raise ValueError('修边线不能重合或自交，请重新画线。')
    result = QgsGeometry(original)
    status = result.reshapeGeometry(line.constGet())
    if status != 0:
        # Preserve the existing inward trim behavior for interior endpoints.
        try:
            candidates = split_geometry_by_line(original, line)
            result = max(candidates, key=area or (lambda g: g.area()))
        except ValueError:
            raise ValueError('修边线需连接原边界两处；填补凹口时连接凹口两侧，或两端略伸入地块。') from None
    if result.isEmpty() or not result.isGeosValid():
        raise ValueError('修边结果无效，当前轮廓未改变。')
    parts = result.asGeometryCollection() if result.isMultipart() else [result]
    result = max(parts, key=area or (lambda g: g.area()))
    if result.isGeosEqual(original):
        raise ValueError('修边线没有改变边界，请连接需要替换的边界两端。')
    return [result]


def line_edit_pieces(geometry, line_geometry, mode, area=None, snap_tolerance=0):
    """Flatten islands before choosing the largest or validating a two-way split."""
    if mode == 'trim':
        return reshape_boundary(geometry, line_geometry, area, snap_tolerance)
    pieces = []
    for result in split_geometry_by_line(geometry, line_geometry):
        result = polygon_only(result)
        parts = result.asGeometryCollection() if result.isMultipart() else [result]
        pieces.extend(part for part in parts if not part.isEmpty() and part.area() > 0)
    pieces.sort(key=area or (lambda part: part.area()), reverse=True)
    if mode != 'split' or len(pieces) != 2:
        raise ValueError('切割线必须产生两个完整地块；当前产生 %s 个片段，请重画。' % len(pieces))
    return pieces
