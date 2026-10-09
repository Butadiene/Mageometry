"""Single numerical process, cancellable between bounded work units."""

from copy import deepcopy
import multiprocessing as mp
from queue import Empty
import traceback

from .engine import SessionEngine, calculation_key


class Cancelled(Exception):
    pass


def _worker(requests, responses, generation, profile_generation):
    engine = None
    previous_key = None
    while True:
        request = requests.get()
        if request is None:
            return
        token, group, profile = request
        if token != generation.value:
            continue
        if profile is not None and profile['id'] != profile_generation.value:
            continue

        def progress(message):
            if token != generation.value:
                raise Cancelled()
            if profile is not None and profile['id'] != profile_generation.value:
                raise Cancelled()
            if message:
                if profile is None:
                    responses.put((token, 'progress', message))

        try:
            key = calculation_key(group)
            if engine is None or key != previous_key:
                candidate = SessionEngine(group, progress, previous=engine)
                engine, previous_key = candidate, key
            engine.progress = progress
            for identity, records in group.get('resolved', {}).get('inputs', {}).items():
                if identity in engine.inputs:
                    engine._verify_fingerprint(records, engine.inputs[identity], identity)
            if profile is not None:
                data = engine.prepare_profile(group['view'], profile['seed_id'], profile['quantities'])
                progress('')
                responses.put((token, 'profile', dict(id=profile['id'], data=data)))
                continue
            result = engine.prepare(group['view'], include_traces=False)
            progress('')
            responses.put((token, 'result', result))
            if result['trace_status'] == 'pending':
                paths = engine.prepare_traces(group['view'])
                progress('')
                responses.put((token, 'traces', dict(paths=paths, records=engine.trace_records(group['view']))))
        except Cancelled:
            if profile is None:
                responses.put((token, 'cancelled', None))
        except Exception as exc:
            if profile is None:
                responses.put((token, 'error', (str(exc), traceback.format_exc())))
            else:
                responses.put((token, 'profile_error', dict(id=profile['id'], message=str(exc))))


class JobRunner:
    """Own a spawned worker; caller polls events on its GUI event loop."""

    def __init__(self):
        context = mp.get_context('spawn')
        self.generation = context.Value('q', 0)
        self.profile_generation = context.Value('q', 0)
        self.requests = context.Queue()
        self.responses = context.Queue()
        self.process = context.Process(target=_worker,
                                       args=(self.requests, self.responses, self.generation, self.profile_generation),
                                       daemon=True)
        self.process.start()
        self.closed = False

    def submit(self, group):
        self.cancel()
        token = self.generation.value
        self.requests.put((token, deepcopy(group), None))
        return token

    def submit_profile(self, group, seed_id, quantities):
        """Queue a profile without cancelling the field or its pending traces."""
        self.cancel_profile()
        token, identity = self.generation.value, self.profile_generation.value
        self.requests.put((token, deepcopy(group), dict(id=identity, seed_id=seed_id, quantities=list(quantities))))
        return token, identity

    def cancel_profile(self):
        with self.profile_generation.get_lock():
            self.profile_generation.value += 1

    def cancel(self):
        self.cancel_profile()
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
