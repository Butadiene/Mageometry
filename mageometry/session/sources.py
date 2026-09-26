"""Reconstruct serial model evaluators and file sources from recipes."""

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np

from .. import geopack, geopack_field
from ..io import GriddedField, load_hdf5, load_vtk, load_xdmf


def model_field(source):
    parameters = source['parameters']
    epoch = parameters['epoch']
    ps = geopack.recalc(epoch)
    parmod = None
    if source['kind'] == 't96':
        parmod = np.array([parameters[k] for k in ('pdyn', 'dst', 'by', 'bz')] + [0.] * 6)
        parmod.setflags(write=False)
    model = geopack_field('t96' if parmod is not None else None, 'dip', parmod, ps)

    def field(x, y, z):
        # All such calls run serially in the session worker; epoch restoration
        # is required for both total fields and their assigned backgrounds.
        geopack.recalc(epoch)
        return model(x, y, z)

    return field, ps


def source_paths(source):
    if source['kind'] in ('t96', 'dipole'):
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
    if source['kind'] in ('t96', 'dipole'):
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
        metadata = dict(model='T96 + dipole', coordinate_system='GSM',
                        length_unit='Re', field_unit='nT',
                        parameters={'Epoch [Unix s]': p['epoch'], 'Dipole tilt [rad]': float(ps),
                                    'Pdyn [nPa]': p['pdyn'], 'Dst [nT]': p['dst'],
                                    'IMF By [nT]': p['by'], 'IMF Bz [nT]': p['bz']})
        return GriddedField(*axes, *values, metadata=metadata), field
    loaders = {'xdmf': load_xdmf, 'hdf5': load_hdf5, 'vtk': load_vtk}
    grid = loaders[source['kind']](source['path'], **source.get('options', {}))
    grid.metadata.update(source.get('metadata', {}))
    return grid, None
