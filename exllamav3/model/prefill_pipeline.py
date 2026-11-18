from __future__ import annotations
import copy
import os
import queue
import threading
import weakref
from functools import partial

import torch

from ..cache.recurrent_util import advance_recurrent_states
from ..util.device_copy import to_device

"""
Prefill across a layer split, with the devices working on different chunks at once.

The stream of each device is in order and transfers between devices are ordered by stream events, so a chunk can be
queued on a device as soon as the host gets to it. What the host can't do by itself is get to it: expert routing reads
counts back to plan its dispatch, which holds the launching thread until the layer's device has caught up, so one
thread can't issue a chunk to the first device while the last device works through the previous one. Each device
gets its own thread here instead.

Everything that touches a device during a prefill is an item that walks one fixed route through the devices, running
one step at each: a chunk runs its section of the model there, a checkpoint copies the states of the layers there, and
the draft model's update for the chunk is a last step on its device. Each device runs its steps first come, first
served, on one thread. That gives the two properties everything depends on:

- Nothing else touches the device while a step runs, so the statics scratch buffers come from can be shared.
- An item can't overtake one issued before it: at the first device it runs later, so it reaches the next device later,
  and so on. A checkpoint therefore reads the states of every layer exactly as the chunks before it left them.

Only prefill goes through here. Model.forward and everything else that uses the devices, or reads what the items
write, waits for the route to clear first (flush) and then has them to itself.
"""


# EXL3_DS_PIPELINE=0 restores the stock rule (no pipeline with CPU-offloaded experts or a pinned attention device)
_DS = os.environ.get("EXL3_DS_PIPELINE", "1") != "0"
# Cap on the chunks in flight at once in that mode. 0 (default): the stock limit, the number of devices + 1, so
# every device can work on a chunk at the same time, except with a pinned attention device, where every chunk in
# flight also allocates on that one device for each of its layers and _DS_PINNED_DEPTH applies instead
_DS_DEPTH = int(os.environ.get("EXL3_DS_PIPELINE_DEPTH", 0))
_DS_PINNED_DEPTH = 2
# Chunks shorter than this run serially: decode-class row counts take static workspaces (GTensorCache.get_scratch,
# the bsz-1 graphs), which chunks in flight on one device would share
_DS_MIN_ROWS = max(int(os.environ.get("EXL3_DS_PIPELINE_MIN_ROWS", 256)), 64)
# 0: don't wait for a device to catch up before it reaches into the attention device (see _SharedAttnDevice)
_DS_PRESYNC = os.environ.get("EXL3_DS_PIPELINE_PRESYNC", "1") != "0"


class _SharedAttnDevice:
    """
    The extra rules of a pipeline over a split whose attention layers all sit on the first device
    (caps["attn_device"], DeepSeek-V4.1) and/or whose experts are partly on the CPU worker (moe_cpu_hosts).

    A block on a later device runs its attention on the first device, from its own device's thread, while the
    first device's thread is working on the next chunk. Whatever is launched on the attention device is launched
    under one lock: by the first device's thread a module at a time, by the others an attention call at a time
    including the transfers in and out. (Holding it only for the first device's attention calls was tried: the
    transients of a remote attention call then overlap those of the first device's MoE, which ran it out of memory.) Launches go to the device's one stream
    in the order they were made, so a call's kernels run as one block there, as they do in a serial forward.

    What the threads share on that device beyond the stream:
    - Attention state. A layer's rings and compressor buffers are only touched by that layer's own calls, which
      all come from the one thread its block belongs to, in chunk order. A pool that later layers read
      (kv_source_layer_ids) is append-only by position and read below the reader's own position, so a chunk
      behind the reader appending to it doesn't change what the reader sees; index selections and candidate
      blocks travel in the chunk's own params (dsv4_shared).
    - Static workspaces. Calls of more than GTensorCache.SCRATCH_STATIC_ROWS rows allocate theirs, apart from
      the attention's dsv4_stage, which is written and read inside one call. Shorter chunks don't come here.

    Before reaching into the attention device a thread waits for its own device to catch up: the copy in is
    ordered behind that device's work, and the attention device would otherwise sit waiting for it with the next
    chunk's work queued behind (one stream, first come, first served).
    """

    def __init__(self, hosts: list, first: torch.device):
        self.lock = threading.RLock()
        self.hosts = hosts
        self.first = first
        self.presync = _DS_PRESYNC
        self.min_rows = _DS_MIN_ROWS
        # Set to run every chunk serially (A/B tests)
        self.force_serial = False
        self.stats = {"pipelined": 0, "serial": 0, "settle": 0}

    def run_attn(self, block, y: torch.Tensor, params: dict) -> torch.Tensor:
        """A TransformerBlock's attention call during a pipelined prefill"""
        attn = block.attn
        if attn.device == block.device:
            # On the first device's thread, which holds the lock for the module, or on a device of its own
            return attn.forward(y, params)
        if self.presync:
            torch.cuda.current_stream(block.device).synchronize()
        with self.lock:
            y = attn.forward(attn.prepare_for_device(y, params), params)
            return to_device(y, block.device)

    def unsettled(self, model) -> bool:
        """Expert transfers between the worker and the devices are landing (promote-on-miss): the first MoE
        layer of the next forward applies them to every layer's placement map at once, which must not happen
        under a chunk in flight, so the next chunk is only issued once the route is clear"""
        reg = getattr(model.config.infer_params, "moe_cpu_swap_modules", None)
        return bool(reg) and any(getattr(m, "_split_pending", None) for m in reg)

    @staticmethod
    def layer_stages(model, cache, stages: list) -> dict | None:
        """Stage whose thread runs each recurrent layer of the cache, by layer key, which with a pinned
        attention device isn't the stage of the device the layer's state is on. None if a layer can't be
        placed"""
        layers = {id(m) for m in model.get_recurrent_layers()}
        out = {}
        for i, (module, instance, _) in enumerate(model.fwd_modules):
            stage = next((s for s, (_, a, b) in enumerate(stages) if a <= i < b), None)
            for m in module:
                if id(m) in layers:
                    if stage is None:
                        return None
                    out[(m.layer_idx, instance)] = stage
        if set(out) != set(cache.get_all_recurrent_layers()):
            return None
        return out


def _ds_pinned(module) -> set:
    """ids of the modules of a layer that the architecture pins to the attention device, with their children"""
    return {id(c) for m in module if m.caps.get("attn_device") for c in m}


def _worker(tasks: queue.SimpleQueue, local: threading.local):
    local.is_worker = True
    with torch.inference_mode():
        while (task := tasks.get()) is not None:
            task()


class _Item:

    def __init__(self, steps: list, owner, is_pass: bool):
        self.steps = steps
        self.owner = owner
        self.is_pass = is_pass
        self.aborted = False
        self.on_done = None
        self.started = None


class PrefillPipeline:

    @classmethod
    def plan(cls, model, cache, draft_model = None) -> tuple[list, int | None] | None:
        """
        The route through the devices for this model and draft model, as (stages, draft stage), each stage a
        (device, first module, end module) of the target's forward (draft stage: an index into the route, past the
        target's stages), or None if the split can't run this way.
        """
        modules = getattr(model, "fwd_modules", None)
        hosts = getattr(model.config, "moe_cpu_hosts", None)
        if not modules or model.loaded_tp or (hosts and not _DS):
            return None
        stages = []
        attn_devices = set()
        for i, (module, _, _) in enumerate(modules):
            device = torch.device(module.device)
            if device.type != "cuda":
                continue
            # Parts of a layer on another device would run on the wrong device's thread. Attention pinned to the
            # split's attention device is run there under _SharedAttnDevice's rules, if that is the first device
            pinned = _ds_pinned(module) if _DS else ()
            attn_devices |= {torch.device(m.device) for m in module if id(m) in pinned and m.device is not None}
            if any(m.device is not None and torch.device(m.device) != device for m in module if id(m) not in pinned):
                return None
            if not stages or stages[-1][0] != device:
                if any(s[0] == device for s in stages):
                    return None
                stages.append((device, i if stages else 0, None))
        if len(stages) < 2:
            return None
        ends = [s[1] for s in stages[1:]] + [len(modules)]
        stages = [(s[0], s[1], e) for s, e in zip(stages, ends)]
        devices = [s[0] for s in stages]
        ds = bool(hosts) or bool(attn_devices)
        if ds:
            if attn_devices and attn_devices != {devices[0]}:
                return None
            # The rows a prefetch stages are kept by its module until its forward takes them, so the next chunk's
            # prefetch has to come after that: on the first device's thread
            if any(m.caps.get("prefetch_ids") for module, _, _ in modules[stages[0][2]:] for m in module):
                return None
            if cache is not None and cache.recurrent_state_cls is not None \
                    and _SharedAttnDevice.layer_stages(model, cache, stages) is None:
                return None

        if cache is not None and cache.recurrent_state_cls is not None:
            layers = {torch.device(l.device) for l in cache.get_all_recurrent_layers().values()}
            if not layers <= set(devices):
                return None
            # Ring states decide their shift in the forward: the first section has to, so that the state can
            # advance for the next chunk before the section after it has run
            if getattr(cache.recurrent_state_cls, "guaranteed_rollback", 0) and devices[0] not in layers:
                return None

        draft_stage = None
        if draft_model is not None:
            if draft_model.loaded_tp or draft_model.caps.get("load_with_target") \
                    or draft_model.caps.get("draft_uses_target_cache"):
                return None
            draft_devices = {torch.device(m.device) for m, _, _ in draft_model.fwd_modules}
            draft_devices.discard(torch.device("cpu"))
            if len(draft_devices) != 1:
                return None
            draft_stage = len(stages)
            stages.append((draft_devices.pop(), 0, 0))
        return stages, draft_stage


    def __init__(self, model, stages: list, draft_stage: int | None, on_idle = None):
        self.model = model
        self.stages = stages
        self.route = [s[0] for s in stages]
        self.draft_stage = draft_stage
        self.on_idle = on_idle
        self.local = threading.local()
        self.cond = threading.Condition()
        self.in_flight = 0
        self.passes = 0
        self.issued = False
        # Chunks past this wait in the caller, so that a slow device can't collect the output of the ones before it
        self.max_passes = len(self.route) + 1
        # Pinned attention device and/or CPU-offloaded experts: see _SharedAttnDevice
        hosts = list((getattr(model.config, "moe_cpu_hosts", None) or {}).values())
        self.ds = None
        pinned = any(_ds_pinned(module) for module, _, _ in model.fwd_modules)
        if _DS and (hosts or pinned):
            self.ds = _SharedAttnDevice(hosts, self.route[0])
            depth = _DS_DEPTH or (_DS_PINNED_DEPTH if pinned else 0)
            if depth > 0:
                self.max_passes = max(min(self.max_passes, depth), 1)
        self.pending_stashes = set()
        self.failed_stashes = set()
        self.failures = []
        self.failed_owners = set()
        self.queues = {device: queue.SimpleQueue() for device in self.route}
        for device, q in self.queues.items():
            threading.Thread(target = _worker, args = (q, self.local), daemon = True, name = f"prefill-{device}").start()
        self.finalizer = weakref.finalize(self, PrefillPipeline._stop, list(self.queues.values()))


    @staticmethod
    def _stop(queues: list):
        for q in queues:
            q.put(None)


    def close(self):
        self.flush()
        self.finalizer()


    @property
    def busy(self) -> bool:
        return self.in_flight > 0


    def flush(self):
        """Wait for everything issued so far to reach the end of the route. No-op on a worker, whose own step is
        part of what would be waited for (a draft model's prefill runs on one)"""
        if not self.issued or getattr(self.local, "is_worker", False):
            return
        with self.cond:
            while self.in_flight:
                self.cond.wait()
        self.issued = False
        if self.on_idle is not None:
            self.on_idle()


    def take_failures(self) -> list:
        """(owner, exception) of the items that failed since the last call"""
        with self.cond:
            failures, self.failures = self.failures, []
            return failures


    def forget_owner(self, owner):
        self.failed_owners.discard(owner)


    def _issue(self, item: _Item):
        self.issued = True
        with self.cond:
            self.in_flight += 1
        self.queues[self.route[0]].put(partial(self._run, item, 0))


    def _run(self, item: _Item, s: int):
        step = item.steps[s]
        # An item of a job that failed is walked through without running, so that what's behind it isn't held up
        if item.owner in self.failed_owners:
            item.aborted = True
        if step is not None and not item.aborted:
            try:
                step()
            except BaseException as e:
                item.aborted = True
                with self.cond:
                    if item.owner not in self.failed_owners:
                        self.failed_owners.add(item.owner)
                        self.failures.append((item.owner, e))
        # The caller of a chunk that didn't run still has to get going again
        if item.started is not None:
            item.started.set()
        s += 1
        if s < len(self.route):
            self.queues[self.route[s]].put(partial(self._run, item, s))
            return
        item.steps = None
        if item.on_done is not None:
            item.on_done(item.aborted)
        with self.cond:
            self.in_flight -= 1
            self.passes -= item.is_pass
            self.cond.notify_all()


    def prefill(self, input_ids: torch.Tensor, params: dict, owner, to_output: bool = False, then = None):
        """
        Issue a prefill of input_ids, returning once the first device has it and the recurrent states have moved
        past it, which is what the next chunk needs. to_output runs the whole forward instead of stopping at the
        last cache-writing layer, for a drafter that reads the states after the final norm. then(params) runs at
        the draft stage with the params of the finished pass, so with the exported states.
        """
        model = self.model
        ds = self.ds
        if ds is not None:
            if ds.force_serial or input_ids.shape[-1] < ds.min_rows or input_ids.shape[0] != 1:
                # The serial path, which waits for the chunks in flight first
                ds.stats["serial"] += 1
                if to_output:
                    model.forward(input_ids = input_ids, params = params)
                else:
                    model.prefill(input_ids = input_ids, params = params)
                if then is not None:
                    then(params)
                return
            if ds.unsettled(model):
                ds.stats["settle"] += 1
                self.flush()
            ds.stats["pipelined"] += 1
            params["ds_attn_lock"] = ds

        with self.cond:
            while self.passes >= self.max_passes:
                self.cond.wait()
            self.passes += 1

        # The page table can move on before later devices get to this chunk
        if "block_table" in params:
            params["block_table"] = params["block_table"].clone()
        n = self.draft_stage if self.draft_stage is not None else len(self.route)
        end = len(model.fwd_modules) if to_output else model.last_kv_end
        run = dict(x = None, params = params)
        started = threading.Event()
        error = []

        def section(stage: int):
            a, b = self.stages[stage][1], min(self.stages[stage][2], end)
            if a < b and ds is not None:
                if stage and ds.presync:
                    # The copy of x from the previous device is ordered behind what this one still has queued
                    torch.cuda.current_stream(self.stages[stage][0]).synchronize()
                if stage == 0:
                    for i in range(a, b):
                        with ds.lock:
                            run["x"] = model.run_section(run["x"], run["params"], i, i + 1, not to_output)
                    return
            if a < b:
                run["x"] = model.run_section(run["x"], run["params"], a, b, not to_output)

        def first():
            try:
                for h in (ds.hosts if ds is not None else ()):
                    h.begin_pass()
                run["x"] = model.prepare_inputs(input_ids, params)
                for m in model._get_prefetch_layers:
                    m.prefetch(run["x"], params)
                section(0)
                # Later devices need the states as this chunk found them; the next chunk needs them past it
                run["params"] = dict(params)
                states = params.get("recurrent_states")
                if states:
                    run["params"]["recurrent_states"] = [copy.copy(r) for r in states]
                advance_recurrent_states(input_ids, params, model)
            except BaseException as e:
                error.append(e)
                item.aborted = True
            finally:
                started.set()

        def finish():
            run["x"] = None
            if ds is not None and self.route[self.draft_stage] == ds.first:
                # A drafter on the attention device takes the chunk there as one more call
                with ds.lock:
                    then(run["params"])
                return
            then(run["params"])

        steps = [first] + [partial(section, s) for s in range(1, n)]
        if self.draft_stage is not None:
            steps.append(finish if then is not None else None)
        item = _Item(steps, owner, True)
        item.started = started
        self._issue(item)
        started.wait()
        if error:
            raise error[0]


    def stash(self, state, owner) -> dict:
        """
        A checkpoint of a paired state, as of the last chunk issued. Returns the stash structure right away; its
        contents arrive as the checkpoint walks the route behind that chunk, each device's layers being copied
        once the device has finished the chunk. The position and shape of a state can be read here, while what the
        draft side has reached is only known at its own stage, behind the chunk's update.
        """
        stashed = {"position": state.position, "checkpoint_size": state.checkpoint_size}
        steps = [None] * len(self.route)
        if state.target is not None:
            snapshot = copy.copy(state.target)
            side = snapshot.stash_header()
            stashed["target"] = side
            by_stage = {}
            ds_stages = _SharedAttnDevice.layer_stages(self.model, snapshot.cache, self.stages[:self.draft_stage]) \
                if self.ds is not None else None
            for k, layer in snapshot.cache.get_all_recurrent_layers().items():
                # (a layer on the pinned attention device is left by the thread of the stage its block is in)
                s = ds_stages[k] if ds_stages is not None else self.route.index(torch.device(layer.device))
                by_stage.setdefault(s, {})[k] = layer
            def copy_layers(layers):
                if self.ds is not None:
                    with self.ds.lock:
                        return side.update(snapshot.stash_layers(layers))
                side.update(snapshot.stash_layers(layers))
            for s, layers in by_stage.items():
                steps[s] = partial(copy_layers, layers)
        if state.draft is not None:
            steps[self.draft_stage] = lambda: stashed.__setitem__("draft", state.draft.stash())
        item = _Item(steps, owner, False)
        self.pending_stashes.add(id(stashed))

        def done(aborted: bool):
            if aborted:
                self.failed_stashes.add(id(stashed))
            self.pending_stashes.discard(id(stashed))

        item.on_done = done
        self._issue(item)
        return stashed
