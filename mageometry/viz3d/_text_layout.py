"""Keep viewer annotations within their allotted viewport rectangles."""

import weakref


def _fit_text(actor, renderer, width, height, font):
    prop = actor.GetTextProperty()
    size = [0., 0.]
    for candidate in range(font, 0, -1):
        prop.SetFontSize(candidate)
        actor.GetSize(renderer, size)
        if size[0] <= width and size[1] <= height:
            break


class _TextLayout:
    """Refit replaced annotations and resized windows, retaining base fonts."""

    def __init__(self, plotter, renderer, boxes):
        self.plotter = weakref.proxy(plotter)
        self.renderer = renderer
        self.boxes = boxes
        self.fonts = {}
        self.state = None
        owner = weakref.ref(self)

        def on_render(obj, event):
            layout = owner()
            if layout is not None:
                layout.refresh()

        plotter.render_window.AddObserver('StartEvent', on_render)
        self.refresh()

    def refresh(self):
        actors = self.renderer.actors
        selected = {key: actors[key] for key in self.boxes if key in actors}
        viewport = self.renderer.GetViewport()
        size = tuple(self.plotter.window_size)
        state = (viewport, size, tuple(selected.items()))
        if state == self.state:
            return
        self.state = state
        width = size[0] * (viewport[2] - viewport[0])
        height = size[1] * (viewport[3] - viewport[1])
        for key, actor in selected.items():
            old_actor, font = self.fonts.get(key, (None, None))
            if actor is not old_actor:
                font = actor.GetTextProperty().GetFontSize()
                self.fonts[key] = (actor, font)
            w, h = self.boxes[key]
            _fit_text(actor, self.renderer, w * width, h * height, font)
