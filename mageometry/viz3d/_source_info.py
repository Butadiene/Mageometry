"""Model-independent source metadata displayed in a reserved header area."""

from collections.abc import Mapping
from numbers import Real
import textwrap
import weakref


def _source_lines(metadata):
    """Only display declared provenance; never infer parameters from a field."""
    def value_text(value):
        text = f'{value:.6g}' if isinstance(value, Real) else str(value)
        return ' '.join(text.split())

    identity = []
    for key, label in (('model', 'Model'), ('source', 'Source'), ('time', 'Time'),
                       ('coordinate_system', 'Coordinates')):
        if key in metadata:
            identity.append(f'{label}: {value_text(metadata[key])}')
    units = [f'{label}: {value_text(metadata[key])}' for key, label in
             (('length_unit', 'length'), ('field_unit', 'B')) if key in metadata]
    if units:
        identity.append('Units: ' + ', '.join(units))
    lines = ['  |  '.join(identity)] if identity else []
    parameters = metadata.get('parameters', {})
    if isinstance(parameters, Mapping) and parameters:
        lines.append('  |  '.join(f'{value_text(key)} = {value_text(value)}'
                                 for key, value in parameters.items()))
    return lines


class _SourceInfo:
    """Wrap and fit provenance without encroaching on other header text."""

    def __init__(self, plotter, focus, metadata):
        from vtkmodules.vtkRenderingCore import vtkTextActor

        self.plotter = weakref.proxy(plotter)
        self.renderer = plotter.renderer
        self.focus = focus
        self.actor = vtkTextActor()
        self.actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.actor.SetUseBounds(False)
        self.actor.PickableOff()
        prop = self.actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetColor(0.25, 0.32, 0.40)
        prop.SetBackgroundColor(0.94, 0.96, 0.98)
        prop.SetBackgroundOpacity(0.95)
        prop.SetVerticalJustificationToTop()
        self.renderer.add_actor(self.actor, name='geometry-source-info',
                                reset_camera=False, render=False)
        focus.props.add(self.actor)
        focus.layout_callbacks.append(self.layout)
        self.update(metadata)
        owner = weakref.ref(self)

        def on_render(obj, event):
            info = owner()
            if info is not None:
                info.layout()

        plotter.render_window.AddObserver('StartEvent', on_render)

    def update(self, metadata):
        self.lines = _source_lines(metadata)
        self.actor.SetVisibility(bool(self.lines))
        self._layout_state = None
        self.layout()

    def layout(self):
        viewport = self.renderer.GetViewport()
        size = tuple(self.plotter.window_size)
        state = (viewport, size, self.focus.active)
        if state == self._layout_state:
            return
        self._layout_state = state
        if not self.lines:
            return
        width = max(size[0] * (viewport[2] - viewport[0]), 1)
        height = max(size[1] * (viewport[3] - viewport[1]), 1)
        self.actor.SetPosition(0.035, 0.805 if self.focus.active else 0.755)
        # Fit the complete text to this reserved area, including long paths
        # and custom parameter labels. VTK sizes account for display DPI.
        prop = self.actor.GetTextProperty()
        measured = [0., 0.]
        def width_of(text):
            self.actor.SetInput(text)
            self.actor.GetSize(self.renderer, measured)
            return measured[0]

        for font in range(15, 0, -1):
            prop.SetFontSize(font)
            wrapped = []
            for line in self.lines:
                row = ''
                for entry in line.split('  |  '):
                    candidate = row + '  |  ' + entry if row else entry
                    if row and width_of(candidate) > 0.93 * width:
                        wrapped.append(row)
                        row = ''
                    if width_of(entry) > 0.93 * width:
                        columns = len(entry)
                        while columns > 1 and width_of(textwrap.fill(entry, columns)) > 0.93 * width:
                            columns -= 1
                        wrapped.extend(textwrap.wrap(entry, columns)[:-1])
                        entry = textwrap.wrap(entry, columns)[-1]
                    row = row + '  |  ' + entry if row else entry
                wrapped.append(row)
            self.actor.SetInput('\n'.join(wrapped))
            self.actor.GetSize(self.renderer, measured)
            if measured[0] <= 0.93 * width and measured[1] <= 0.11 * height:
                break
