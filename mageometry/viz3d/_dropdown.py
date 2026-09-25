"""A renderer-local dropdown, using only the existing VTK dependency."""

import weakref

from ._text_layout import _fit_text

class _Dropdown:
    """Clickable labelled choices that do not pass menu gestures to the camera.

    Actors use viewport coordinates; hit testing uses VTK display pixels.
    Interactor observers have priority over both trackball and image styles.
    """

    def __init__(self, plotter, options, selected, callback, *,
                 name='current-component',
                 caption='COMPONENT   /   F5: previous   F6: next',
                 x_range=(0.55, 0.96)):
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import (vtkActor2D, vtkCoordinate,
                                                 vtkPolyDataMapper2D, vtkTextActor)

        self.plotter = weakref.proxy(plotter)
        self.renderer = plotter.renderer
        self.name = name
        self.on_rebuild = None
        self.on_dismiss = None
        self.options = dict(options)
        self.keys = tuple(self.options)
        self.selected = selected
        self.callback = callback
        self.opened = False
        self.highlight = self.keys.index(selected)
        self.pressed = False
        self.props = set()
        self.peers = weakref.WeakSet()
        self.x_range = x_range
        self._layout_state = None

        def rectangle(name, color, layer):
            points = vtkPoints()
            points.SetNumberOfPoints(4)
            cells = vtkCellArray()
            cells.InsertNextCell(4)
            for index in range(4):
                cells.InsertCellPoint(index)
            data = vtkPolyData()
            data.SetPoints(points)
            data.SetPolys(cells)
            coords = vtkCoordinate()
            coords.SetCoordinateSystemToNormalizedViewport()
            mapper = vtkPolyDataMapper2D()
            mapper.SetInputData(data)
            mapper.SetTransformCoordinate(coords)
            actor = vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.SetLayerNumber(layer)
            self._add(actor, name)
            return actor

        def label(name, text, layer):
            actor = vtkTextActor()
            actor.SetInput(text)
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            prop = actor.GetTextProperty()
            prop.SetFontFamilyToArial()
            prop.SetColor(0.15, 0.21, 0.28)
            prop.SetVerticalJustificationToCentered()
            actor.SetLayerNumber(layer)
            self._add(actor, name)
            return actor

        self._make_rectangle = rectangle
        self._make_label = label
        self.header = rectangle(f'{name}-button', (0.88, 0.92, 0.96), 20)
        self.value = label(f'{name}-value', '', 22)
        self.arrow = label(f'{name}-arrow', 'v', 22)
        self.caption = label(f'{name}-caption', caption, 22)
        self.rows = []
        for key, text in self.options.items():
            background = rectangle(f'{name}-row-{key}', (1., 1., 1.), 30)
            text_actor = label(f'{name}-option-{key}', text, 32)
            self.rows.append((background, text_actor))
        self.layout()
        self._refresh()

        # Weak observer closures do not keep the plotter alive after close.
        owner = weakref.ref(self)

        def on_event(obj, event):
            menu = owner()
            if menu is not None:
                command = obj.GetCommand(menu._observers[event])
                command.SetAbortFlag(0)
                try:
                    consumed = menu._event(obj, event)
                except Exception:
                    # Do not start a camera drag if a selection's compute
                    # callback fails; its matching release is still ours.
                    command.SetAbortFlag(1)
                    raise
                if consumed:
                    command.SetAbortFlag(1)

        self._observers = {}
        for event in ('LeftButtonPressEvent', 'LeftButtonReleaseEvent', 'MouseMoveEvent',
                      'MouseWheelForwardEvent', 'MouseWheelBackwardEvent', 'KeyPressEvent'):
            self._observers[event] = plotter.iren.interactor.AddObserver(event, on_event, 1.0)

        def on_render(obj, event):
            menu = owner()
            if menu is not None:
                menu.layout()

        plotter.render_window.AddObserver('StartEvent', on_render)

    def _add(self, actor, name):
        actor.PickableOff()
        actor.SetUseBounds(False)
        self.renderer.add_actor(actor, name=name, reset_camera=False, render=False)
        self.props.add(actor)

    @staticmethod
    def _rectangle(actor, bounds):
        x0, y0, x1, y1 = bounds
        points = actor.GetMapper().GetInput().GetPoints()
        for index, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            points.SetPoint(index, x, y, 0)
        points.Modified()

    def layout(self):
        """Keep both drawing and hit boxes aligned after resize or F4."""
        viewport = self.renderer.GetViewport()
        size = tuple(self.plotter.window_size)
        state = (viewport, size)
        if state == self._layout_state:
            return
        self._layout_state = state
        width = max(size[0] * (viewport[2] - viewport[0]), 1)
        height = max(size[1] * (viewport[3] - viewport[1]), 1)
        font = max(9, min(17, int(width * 0.017)))
        row_height = min(font + 14, height * 0.065) / height
        header_height = min(42, height * 0.055) / height
        self.bounds = (self.x_range[0], 0.965 - header_height, self.x_range[1], 0.965)
        x0, y0, x1, y1 = self.bounds
        row_height = min(row_height, (y0 - 0.04) / len(self.rows))
        self._rectangle(self.header, self.bounds)
        inset = 12 / width
        self.value.SetPosition(x0 + inset, (y0 + y1) / 2)
        self.arrow.SetPosition(x1 - 23 / width, (y0 + y1) / 2)
        self.caption.SetPosition(x0, min(0.988, y1 + 14 / height))
        self.value.SetInput(self.options[self.selected])
        self.value.GetTextProperty().SetFontSize(font)
        self.arrow.GetTextProperty().SetFontSize(font)
        self.caption.GetTextProperty().SetFontSize(max(8, font - 5))
        _fit_text(self.value, self.renderer, (x1 - x0) * width - 42,
                  header_height * height, font)
        _fit_text(self.caption, self.renderer, (x1 - x0) * width,
                  20, max(8, font - 5))
        self.row_bounds = []
        for index, (background, text) in enumerate(self.rows):
            top = y0 - index * row_height
            bounds = (x0, top - row_height, x1, top)
            self.row_bounds.append(bounds)
            self._rectangle(background, bounds)
            text.SetPosition(x0 + inset, top - row_height / 2)
            text.GetTextProperty().SetFontSize(font)
            _fit_text(text, self.renderer, (x1 - x0) * width - 24,
                      row_height * height, font)

    def _refresh(self):
        self.value.SetInput(self.options[self.selected])
        self.arrow.SetInput('^' if self.opened else 'v')
        for index, (background, text) in enumerate(self.rows):
            selected = self.keys[index] == self.selected
            color = (0.83, 0.90, 1.) if index == self.highlight else (
                (0.92, 0.95, 0.99) if selected else (1., 1., 1.))
            background.GetProperty().SetColor(color)
            text.GetTextProperty().SetBold(selected)
            background.SetVisibility(self.opened)
            text.SetVisibility(self.opened)

    def set_selected(self, key):
        """Synchronize the closed header after clicks or F5/F6 shortcuts."""
        self.selected = key
        self.highlight = self.keys.index(key)
        self.opened = False
        self._refresh()
        self._layout_state = None
        self.layout()

    def set_options(self, options, selected):
        """Replace choices without replacing event observers or menu peers."""
        if not options or selected not in options:
            raise ValueError('Dropdown choices must include the selected key.')
        removed = {actor for row in self.rows for actor in row}
        for actor in removed:
            self.renderer.remove_actor(actor, reset_camera=False, render=False)
        self.props.difference_update(removed)
        self.options = dict(options)
        self.keys = tuple(options)
        self.rows = []
        for key, text in self.options.items():
            background = self._make_rectangle(f'{self.name}-row-{key}', (1., 1., 1.), 30)
            label = self._make_label(f'{self.name}-option-{key}', text, 32)
            self.rows.append((background, label))
        if self.on_rebuild is not None:
            self.on_rebuild(removed, {actor for row in self.rows for actor in row})
        self.set_selected(selected)

    def dismiss(self):
        self.opened = False
        self._refresh()
        if self.on_dismiss is not None:
            self.on_dismiss()

    def link(self, peer):
        """Allow one-click transfer between menus without a camera gesture."""
        self.peers.add(peer)
        peer.peers.add(self)

    def _choose(self, index):
        # The callback commits selection only after successful computation.
        self.opened = False
        self._refresh()
        self.callback(self.keys[index])

    def _hit(self, position, bounds):
        vx0, vy0, vx1, vy1 = self.renderer.GetViewport()
        width, height = self.plotter.window_size
        x = (position[0] / width - vx0) / (vx1 - vx0)
        y = (position[1] / height - vy0) / (vy1 - vy0)
        x0, y0, x1, y1 = bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    def _event(self, interactor, event):
        self.layout()
        if event == 'LeftButtonReleaseEvent':
            consumed, self.pressed = self.pressed, False
            return consumed
        position = interactor.GetEventPosition()
        if event == 'LeftButtonPressEvent':
            for peer in self.peers:
                peer.layout()
                if self.opened and peer._hit(position, peer.bounds):
                    self.dismiss()
                    return False
            if self._hit(position, self.bounds):
                for peer in self.peers:
                    if peer.opened:
                        peer.dismiss()
                if self.opened:
                    self.dismiss()
                else:
                    self.opened = True
                self.highlight = self.keys.index(self.selected)
            elif self.opened:
                for index, bounds in enumerate(self.row_bounds):
                    if self._hit(position, bounds):
                        self.pressed = True
                        self._choose(index)
                        self.plotter.render()
                        return True
                self.dismiss()  # outside click dismisses without rotating
            else:
                return False
            self.pressed = True
        elif self.opened and event == 'MouseMoveEvent':
            for index, bounds in enumerate(self.row_bounds):
                if self._hit(position, bounds):
                    if index == self.highlight:
                        return True
                    self.highlight = index
                    break
            else:
                return True
        elif self.opened and event in ('MouseWheelForwardEvent', 'MouseWheelBackwardEvent'):
            return True  # do not zoom the scene through an open menu
        elif self.opened and event == 'KeyPressEvent':
            key = interactor.GetKeySym()
            if key in ('Up', 'Down', 'Home', 'End'):
                self.highlight = {'Up': (self.highlight - 1) % len(self.keys),
                                  'Down': (self.highlight + 1) % len(self.keys),
                                  'Home': 0, 'End': len(self.keys) - 1}[key]
            elif key in ('Return', 'KP_Enter', 'space'):
                self._choose(self.highlight)
            else:
                self.dismiss()
                self.plotter.render()
                return key == 'Escape'  # other viewer shortcuts still work
        else:
            return self.pressed and event == 'MouseMoveEvent'
        self._refresh()
        self.plotter.render()
        return True
