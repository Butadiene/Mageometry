"""Reconstruct serial model evaluators and file sources from recipes."""

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from .. import geopack, geopack_field
from ..io import GriddedField, load_hdf5, load_vtk, load_xdmf
from .specs import MODEL_KINDS, MODEL_PARAMETERS, validate_source


def source_label(source):
    """Describe the actual source used for model/file provenance."""
    if source['kind'] in MODEL_KINDS:
        return source['kind'].upper() + ' + dipole'
    if source['kind'] == 'dipole':
        return 'Dipole'
    return source['path']


def model_field(source):
    validate_source(source)
    parameters = source['parameters']
    epoch = parameters['epoch']
    ps = geopack.recalc(epoch)
    parmod = None
    kind = source['kind']
    if kind == 't89':
        parmod = parameters['iopt']
    elif kind in MODEL_KINDS:
        keys = MODEL_PARAMETERS[kind][1:]
        parmod = np.array([parameters[k] for k in keys] + [0.] * (10 - len(keys)))
        parmod.setflags(write=False)
    model = geopack_field(kind if kind in MODEL_KINDS else None, 'dip', parmod, ps)

    def field(x, y, z):
        # All such calls run serially in the session worker; epoch restoration
        # is required for both total fields and their assigned backgrounds.
        geopack.recalc(epoch)
        if kind not in ('t01', 't04'):
            return model(x, y, z)
        # The vectorized T01/T04 APIs clip x < -15. Such clamping would
        # create artificial gradients in a viewer stencil; leave it undefined.
        scalar_input = all(np.isscalar(c) for c in (x, y, z))
        coords = np.broadcast_arrays(x, y, z)
        valid = np.all(np.isfinite(coords), axis=0) & (coords[0] >= -15.)
        values = [np.full(coords[0].shape, np.nan) for _ in range(3)]
        if np.any(valid):
            for out, value in zip(values, model(*(c[valid] for c in coords))):
                out[valid] = value
        return tuple(value.item() if scalar_input else value for value in values)

    return field, ps


def source_paths(source):
    if source['kind'] in MODEL_KINDS or source['kind'] == 'dipole':
        return []
    path = Path(source['path']).resolve()
    paths = [path]
    if source['kind'] == 'xdmf':
        override = source.get('options', {}).get('h5_file')
        if override:
            paths.append(Path(override).resolve())
        else:
            root = ET.parse(path).getroot()
            for item in root.iter('DataItem'):
                if item.get('Format', '').upper() == 'HDF' and item.text:
                    filename = item.text.strip().partition(':')[0]
                    paths.append((path.parent / filename).resolve())
    return list(dict.fromkeys(paths))


def fingerprint(source, check=lambda: None):
    records = []
    for path in source_paths(source):
        check()
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                check()
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed while reading: {path}')
        records.append(dict(path=str(path), size=after.st_size,
                            mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest()))
    return records


def load_source(source, mask_radius=0., check=lambda: None):
    check()
    if source['kind'] in MODEL_KINDS or source['kind'] == 'dipole':
        field, ps = model_field(source)
        if source['kind'] == 'dipole':
            return None, field
        axes = [np.linspace(*interval, count) for interval, count in
                zip(source['bounds'], source['shape'])]
        coords = np.meshgrid(*axes, indexing='ij')
        valid = sum(c*c for c in coords) >= mask_radius**2
        values = [np.full(source['shape'], np.nan) for _ in range(3)]
        # Chunk expensive model calls for progress/cancellation boundaries.
        flat = np.flatnonzero(valid)
        for start in range(0, len(flat), 16384):
            check()
            index = flat[start:start + 16384]
            for output, sampled in zip(values, field(*(c.ravel()[index] for c in coords))):
                output.ravel()[index] = sampled
        p = source['parameters']
        labels = {'epoch': 'Epoch [Unix s]', 'iopt': 'T89 activity bin iopt',
                  'pdyn': 'Pdyn [nPa]', 'dst': 'Dst [nT]', 'by': 'IMF By [nT]',
                  'bz': 'IMF Bz [nT]', 'g1': 'G1', 'g2': 'G2'}
        display = {labels.get(key, key.upper()): p[key] for key in MODEL_PARAMETERS[source['kind']]}
        display['Dipole tilt [rad]'] = float(ps)
        metadata = dict(model=source_label(source), coordinate_system='GSM',
                        length_unit='Re', field_unit='nT',
                        epoch=p['epoch'], dipole_tilt=float(ps), parameters=display)
        metadata.update({{'by': 'imf_by', 'bz': 'imf_bz'}.get(key, key): value
                         for key, value in p.items() if key != 'epoch'})
        return GriddedField(*axes, *values, metadata=metadata), field
    loaders = {'xdmf': load_xdmf, 'hdf5': load_hdf5, 'vtk': load_vtk}
    grid = loaders[source['kind']](source['path'], **source.get('options', {}))
    grid.metadata.update(source.get('metadata', {}))
    return grid, None
