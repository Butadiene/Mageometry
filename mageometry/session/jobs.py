"""Single numerical process, cancellable between bounded work units."""

from copy import deepcopy
import multiprocessing as mp
from queue import Empty
import traceback

from .engine import SessionEngine, calculation_key


class Cancelled(Exception):
    pass


def _worker(requests, responses, generation):
    engine = None
    previous_key = None
    while True:
        request = requests.get()
        if request is None:
            return
        token, group = request
        if token != generation.value:
            continue

        def progress(message):
            if token != generation.value:
                raise Cancelled()
            if message:
                responses.put((token, 'progress', message))

        try:
            key = calculation_key(group)
            if engine is None or key != previous_key:
                candidate = SessionEngine(group, progress)
                engine, previous_key = candidate, key
            engine.progress = progress
            for identity, records in group.get('resolved', {}).get('inputs', {}).items():
                if identity in engine.inputs:
                    engine._verify_fingerprint(records, engine.inputs[identity], identity)
            result = engine.prepare(group['view'])
            progress('')
            responses.put((token, 'result', result))
        except Cancelled:
            responses.put((token, 'cancelled', None))
        except Exception as exc:
            responses.put((token, 'error', (str(exc), traceback.format_exc())))


class JobRunner:
    """Own a spawned worker; caller polls events on its GUI event loop."""

    def __init__(self):
        context = mp.get_context('spawn')
        self.generation = context.Value('q', 0)
        self.requests = context.Queue()
        self.responses = context.Queue()
        self.process = context.Process(target=_worker,
                                       args=(self.requests, self.responses, self.generation),
                                       daemon=True)
        self.process.start()
        self.closed = False

    def submit(self, group):
        self.cancel()
        token = self.generation.value
        self.requests.put((token, deepcopy(group)))
        return token

    def cancel(self):
        with self.generation.get_lock():
            self.generation.value += 1

    def poll(self):
        events = []
        while True:
            try:
                event = self.responses.get_nowait()
            except Empty:
                break
            if event[0] == self.generation.value:
                events.append(event)
        return events

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel()
        self.requests.put(None)
        self.process.join(timeout=.5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for queue in (self.requests, self.responses):
            queue.cancel_join_thread()
            queue.close()
