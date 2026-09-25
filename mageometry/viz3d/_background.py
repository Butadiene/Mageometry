"""Background sources and an in-viewer file chooser using only VTK controls."""

from collections.abc import Mapping
from pathlib import Path

from ..io import GriddedField, load_vtk, load_xdmf


SUFFIXES = {'.vti', '.vtr', '.xmf', '.xdmf'}


def _load_background(path, stride=1):
    suffix = Path(path).suffix.lower()
    if suffix in ('.xmf', '.xdmf'):
        return load_xdmf(path, stride=stride)
    if suffix in ('.vti', '.vtr'):
        return load_vtk(path, stride=stride)
    raise ValueError('Choose an XDMF (.xmf/.xdmf) or VTK (.vti/.vtr) background.')


class _BackgroundChoices:
    def __init__(self, choices=None, initial=None, initial_label='Background',
                 loader=None, directory=None):
        if choices is not None and not isinstance(choices, Mapping):
            raise TypeError('background_choices must map labels to callables or GriddedField objects.')
        self.sources = {}
        self.labels = {'none': 'None (total field)'}
        for index, (label, source) in enumerate((choices or {}).items()):
            self.add(f'preset-{index}', label, source)
        self.selected = 'none'
        if initial is not None:
            match = next((key for key, source in self.sources.items() if source is initial), None)
            if match is None:
                self.add('initial', initial_label, initial)
            self.selected = match or 'initial'
        if loader is not None and not callable(loader):
            raise TypeError('background_loader must be callable.')
        self.loader = loader or _load_background
        self.directory = Path(directory).expanduser().resolve() if directory is not None else Path.cwd()

    def add(self, key, label, source):
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Background labels must be nonempty strings.')
        if not callable(source) and not isinstance(source, GriddedField):
            raise TypeError('Each background must be callable or a GriddedField.')
        self.sources[key] = source
        self.labels[key] = label

    def options(self):
        return dict(self.labels, load='Load background file...')


class _BackgroundFileMenu:
    """Browse folders in the background dropdown; cancel restores the scene."""

    PAGE_SIZE = 10

    def __init__(self, menu, directory, accept, error):
        self.menu, self.directory = menu, Path(directory)
        self.accept, self.error = accept, error
        self.original = (dict(menu.options), menu.selected, menu.callback,
                         menu.caption.GetInput(), menu.on_dismiss)
        self.page = 0
        self.entries = []
        menu.on_dismiss = self.cancel
        menu.callback = self.choose
        try:
            self.show(self.directory)
        except Exception:
            self.cancel()
            raise

    def show(self, directory, page=0):
        # Read first: an inaccessible folder must not erase the current list.
        entries = sorted((p for p in directory.iterdir()
                          if p.is_dir() or p.suffix.lower() in SUFFIXES),
                         key=lambda p: (not p.is_dir(), p.name.casefold()))
        self.directory, self.entries = directory, entries
        pages = max(1, (len(entries) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page = min(max(page, 0), pages - 1)
        options = {'location': str(directory), 'parent': '../ (parent folder)',
                   'home': 'Home folder', 'root': 'Filesystem root'}
        start = self.page * self.PAGE_SIZE
        options.update({f'entry-{i}': ('[Folder] ' if entry.is_dir() else '') + entry.name
                        for i, entry in enumerate(entries[start:start + self.PAGE_SIZE], start)})
        if self.page:
            options['previous'] = 'Previous page'
        if self.page + 1 < pages:
            options['next'] = 'Next page'
        options['cancel'] = 'Cancel'
        self.menu.caption.SetInput(f'BACKGROUND FILE   /   {self.page + 1} of {pages}')
        self.menu.set_options(options, 'location')
        self.menu.opened = True
        self.menu._refresh()

    def cancel(self):
        options, selected, callback, caption, dismiss = self.original
        self.menu.callback, self.menu.on_dismiss = callback, dismiss
        self.menu.caption.SetInput(caption)
        self.menu.set_options(options, selected)

    def choose(self, key):
        try:
            if key == 'cancel':
                self.cancel()
            elif key in ('location', 'parent', 'home', 'root'):
                directory = {'location': self.directory, 'parent': self.directory.parent,
                             'home': Path.home(), 'root': Path(self.directory.anchor)}[key]
                self.show(directory)
            elif key in ('previous', 'next'):
                self.show(self.directory, self.page + (1 if key == 'next' else -1))
            else:
                path = self.entries[int(key.split('-')[1])]
                if path.is_dir():
                    self.show(path)
                else:
                    self.cancel()
                    self.accept(path)
        except Exception as exc:
            self.error(exc)
            # Reopen an intact directory menu after navigation failure.
            if self.menu.on_dismiss is not None:
                self.menu.opened = True
                self.menu._refresh()
