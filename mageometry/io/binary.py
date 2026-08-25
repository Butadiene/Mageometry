# mageometry/io/binary.py
"""
Low-level helpers for reading raw simulation output.

These are building blocks for writing your own ``load_<format>()`` reader
(see ``docs/simulation_data_formats.md``); they know nothing about grids
or field components.
"""

import os

import numpy as np

__all__ = ["iter_fortran_records", "read_fortran_records"]


def iter_fortran_records(path, dtype='>f4', marker_dtype=None, start=0):
    """
    Iterate over the records of a Fortran unformatted *sequential* file.

    Each record is stored as ``<length><payload><length>`` where the length
    markers are 4-byte (or 8-byte) integers in the file's byte order. Many
    Fortran MHD codes dump their state this way, one record per variable,
    per plane, or per block.

    Parameters
    ----------
    path : str
        File path.
    dtype : dtype-like, optional
        Payload element type *including byte order*, e.g. ``'>f4'`` for
        big-endian float32 (typical of files written on big-endian systems
        or with ``convert='big_endian'``), ``'<f8'`` for little-endian
        float64. Default ``'>f4'``.
    marker_dtype : dtype-like, optional
        Record-length marker type. Default: 4-byte integer with the same
        byte order as ``dtype``. Use ``'>i8'`` / ``'<i8'`` for compilers
        that write 8-byte markers.
    start : int, optional
        Byte offset at which to start (to skip a header). Default 0.

    Yields
    ------
    ndarray
        1D array of the record payload (native byte order, read-only view
        converted with ``astype`` if you need to modify it).

    Raises
    ------
    ValueError
        If the leading and trailing markers of a record disagree (wrong
        ``dtype``/byte order, corrupt file, or not a sequential file).
    """
    dtype = np.dtype(dtype)
    if marker_dtype is None:
        marker_dtype = np.dtype('i4').newbyteorder(dtype.byteorder or '=')
    marker_dtype = np.dtype(marker_dtype)
    msize = marker_dtype.itemsize
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.seek(start)
        pos = start
        while pos < size:
            head = f.read(msize)
            if len(head) < msize:
                return
            n = int(np.frombuffer(head, dtype=marker_dtype)[0])
            if n < 0 or n % dtype.itemsize:
                raise ValueError(
                    f"Record length {n} at byte {pos} is not a multiple of "
                    f"{dtype.itemsize} bytes; check dtype/byte order."
                )
            payload = f.read(n)
            tail = f.read(msize)
            if len(payload) < n or len(tail) < msize:
                raise ValueError(f"Truncated record at byte {pos}.")
            n_tail = int(np.frombuffer(tail, dtype=marker_dtype)[0])
            if n_tail != n:
                raise ValueError(
                    f"Record markers disagree at byte {pos} ({n} vs {n_tail}); "
                    "check dtype/byte order or marker_dtype."
                )
            yield np.frombuffer(payload, dtype=dtype).astype(dtype.newbyteorder('='))
            pos += 2 * msize + n


def read_fortran_records(path, dtype='>f4', count=None, skip=0, **kwargs):
    """
    Read records of a Fortran unformatted sequential file into a list.

    Parameters
    ----------
    path : str
        File path.
    dtype : dtype-like, optional
        Payload element type including byte order; see `iter_fortran_records`.
    count : int, optional
        Number of records to read (after ``skip``). Default: all.
    skip : int, optional
        Number of leading records to skip. Default 0.
    **kwargs
        Passed to `iter_fortran_records` (``marker_dtype``, ``start``).

    Returns
    -------
    list of ndarray
        One 1D array per record. Reshape them yourself, e.g.
        ``rec.reshape(ny, nx)`` for an x-y plane written in Fortran order.
    """
    out = []
    for k, rec in enumerate(iter_fortran_records(path, dtype, **kwargs)):
        if k < skip:
            continue
        out.append(rec)
        if count is not None and len(out) >= count:
            break
    return out
