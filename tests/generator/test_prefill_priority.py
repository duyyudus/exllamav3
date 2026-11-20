"""
Prompt ingestion first (Generator prefill_priority, prefill_slice, decode_slice).

No GPU or model is required: Generator is instantiated via __new__ with stub jobs and stubbed model-facing
steps, so only the scheduling in iterate() is exercised. A fake clock advances by a fixed amount per prefill
chunk and per generation step.
"""

from types import SimpleNamespace
import pytest
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator import generator as generator_module
from exllamav3.generator.generator import Generator
from unittest.mock import Mock, patch

pytestmark = pytest.mark.nogpu

CHUNK = 1024


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class StubSeq:
    def __init__(self, prompt, cached_pages):
        self.sequence_ids = [0] * (prompt + 1)   # prefill stops one token short of the end
        self.kv_position = 0
        pages = prompt // PAGE_SIZE + 1
        self.allocated_pages = [
            SimpleNamespace(kv_position = PAGE_SIZE if i < cached_pages else 0) for i in range(pages)
        ]


class StubJob:
    """A generating job (prompt 0), or one with a prompt to ingest in CHUNK-token passes"""

    def __init__(self, name, log, clock, prompt = 0, cached_pages = 0):
        self.name = name
        self.log = log
        self.clock = clock
        self.sequences = [StubSeq(prompt, cached_pages)]
        self.chunks = []
        # A generating job has been streaming for a while; a prompt job gets its first token later
        self.time_first_token = 0.0 if prompt == 0 else None

    def is_prefill_done(self):
        seq = self.sequences[0]
        return seq.kv_position == len(seq.sequence_ids) - 1

    def prefill(self, results, chunk_size = None):
        seq = self.sequences[0]
        end = len(seq.sequence_ids) - 1
        if seq.kv_position == end:
            return
        # Cached pages are skipped without a pass, as in Job.prefill
        while seq.kv_position + PAGE_SIZE <= end and \
                seq.allocated_pages[seq.kv_position // PAGE_SIZE].kv_position == PAGE_SIZE:
            seq.kv_position += PAGE_SIZE
        seq.kv_position = min(end, seq.kv_position + (chunk_size or CHUNK))
        self.chunks.append(chunk_size)
        self.log.append(self.name)
        self.clock.now += 1.0


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(generator_module, "time", c)
    return c


def make_generator(jobs, log, clock, prefill_slice = 0.0, decode_slice = 2.0):
    g = Generator.__new__(Generator)
    g.cache = SimpleNamespace(initialized = True, owner_serial = None)
    g.draft_cache = None
    g.pending_jobs = []
    g.active_jobs = list(jobs)
    g.recurrent_cache = None
    g.pipeline = None
    g.draft_model = None
    g.dflash_draft = False
    g.mtp_draft = False
    g.ngram_match_min = 0
    g.hybrid_draft_tokens = 0
    g.visualizer = None
    g.disk_page_cache = None
    g.max_chunk_size = CHUNK
    g.shared_chunk_size = CHUNK // 2
    g.prefill_cooldown = 4
    g.prefill_cooldown_left = 0
    g.prefill_rotation = 0
    g.prefill_priority = True
    g.prefill_slice = prefill_slice
    g.decode_slice = decode_slice
    g.prefill_slice_start = None
    g.decode_turn_end = None
    g.decode_turn_pending = False
    g.iterate_start_jobs = lambda results: None

    def generate(results, draft_tokens = None):
        batch = [job for job in g.active_jobs if job.is_prefill_done()]
        if batch:
            log.append("G")
            for job in batch:
                if job.time_first_token is None:
                    job.time_first_token = clock.now
            clock.now += 0.5
    g.iterate_gen = generate
    return g


def run(g, iterations):
    for _ in range(iterations):
        g.iterate()


def test_generation_pauses_while_a_prompt_is_left_and_chunks_are_full(clock):
    log = []
    a, b = StubJob("a", log, clock), StubJob("b", log, clock, prompt = 3000)
    g = make_generator([a, b], log, clock)
    run(g, 5)
    # Three full chunks with no generation step between them, then generation resumes
    assert log == ["b", "b", "b", "G", "G"]
    assert b.chunks == [None] * 3


def test_paused_iteration_leaves_the_pipeline_running(clock):
    log = []
    a, b = StubJob("a", log, clock), StubJob("b", log, clock, prompt = 2000)
    g = make_generator([a, b], log, clock)
    g.pipeline = Mock(busy = True)
    g.reap_pipeline_failures = Mock()
    run(g, 2)
    g.pipeline.flush.assert_not_called()
    assert g.reap_pipeline_failures.call_count == 2
    run(g, 1)   # nothing left to ingest: the generation step waits for the chunks in flight
    g.pipeline.flush.assert_called_once()
    assert log == ["b", "b", "G"]


def test_one_prompt_at_a_time_fewest_tokens_first(clock):
    log = []
    b = StubJob("b", log, clock, prompt = 4000)
    c = StubJob("c", log, clock, prompt = 1500)
    g = make_generator([b, c], log, clock)
    run(g, 7)
    # c's first token comes right after its prompt; then c pauses while b's prompt goes in, and b's first
    # token (with c's next one) comes after b's
    assert log == ["c", "c", "G", "b", "b", "b", "b", "G"]


def test_finished_prompt_gets_its_first_token_before_the_next_prompt(clock):
    log = []
    a = StubJob("a", log, clock)
    b = StubJob("b", log, clock, prompt = 4000)
    c = StubJob("c", log, clock, prompt = 1500)
    g = make_generator([a, b, c], log, clock)
    run(g, 8)
    # With a generating, c's last chunk runs in a paused iteration; the next one is a step for c's
    # first token (and one for a), with b's pass deferred, then b's prompt goes in
    assert log == ["c", "c", "G", "b", "b", "b", "b", "G"]
    assert c.time_first_token == 2.0


def test_cached_pages_do_not_count_as_prompt_left(clock):
    log = []
    # 4000 tokens with 12 cached pages (3072) leaves 928; 2000 uncached is the longer one
    b = StubJob("b", log, clock, prompt = 4000, cached_pages = 12)
    c = StubJob("c", log, clock, prompt = 2000)
    assert Generator.prefill_left(b) == 4000 - 12 * PAGE_SIZE
    assert Generator.prefill_left(c) == 2000
    g = make_generator([c, b], log, clock)
    run(g, 4)
    assert log == ["b", "G", "c", "c", "G"]


def test_ties_keep_queue_order(clock):
    log = []
    b = StubJob("b", log, clock, prompt = 2000)
    c = StubJob("c", log, clock, prompt = 2000)
    g = make_generator([b, c], log, clock)
    run(g, 5)
    assert log == ["b", "b", "G", "c", "c", "G"]


def test_prefill_slice_gives_generating_jobs_a_turn(clock):
    log = []
    a, b = StubJob("a", log, clock), StubJob("b", log, clock, prompt = 100 * CHUNK)
    g = make_generator([a, b], log, clock, prefill_slice = 3.0, decode_slice = 2.0)
    run(g, 12)
    # Chunks at t = 0, 1, 2; at t = 3 the turn starts, and runs 2 s from the end of its first step (t = 3.5),
    # so steps start at 3, 3.5, 4, 4.5 and 5; at t = 5.5 prompt ingestion resumes for another 3 s
    assert log == ["b"] * 3 + ["G"] * 5 + ["b"] * 3 + ["G"]


def test_turn_state_resets_when_contention_ends(clock):
    log = []
    a, b = StubJob("a", log, clock), StubJob("b", log, clock, prompt = 100 * CHUNK)
    g = make_generator([a, b], log, clock, prefill_slice = 3.0, decode_slice = 2.0)
    run(g, 4)
    assert log == ["b"] * 3 + ["G"]
    g.active_jobs.remove(a)   # the generating job finishes during its turn
    run(g, 5)
    assert log == ["b"] * 3 + ["G"] + ["b"] * 5
    assert g.decode_turn_end is None and not g.decode_turn_pending
    g.active_jobs.append(StubJob("d", log, clock))
    run(g, 4)
    # A fresh contention starts a fresh pause of the full prefill_slice
    assert log[-4:] == ["b", "b", "b", "G"]


def test_no_generating_job_never_pauses_or_starts_a_turn(clock):
    log = []
    b = StubJob("b", log, clock, prompt = 10 * CHUNK)
    g = make_generator([b], log, clock, prefill_slice = 1.0)
    run(g, 5)
    assert log == ["b"] * 5
    assert g.prefill_slice_start is None


def test_prefill_failure_is_contained_and_generation_resumes(clock):
    log = []

    class Failing(StubJob):
        def prefill(self, results, chunk_size = None):
            raise RuntimeError("boom")

    a, b = StubJob("a", log, clock), Failing("b", log, clock, prompt = 2000)
    g = make_generator([a, b], log, clock)
    reaped = []

    def reap(job, e, results):
        reaped.append((job, str(e)))
        g.active_jobs.remove(job)
    g.reap_failed_job = reap
    run(g, 2)
    assert reaped == [(b, "boom")]
    assert log == ["G"]


def make_real_generator(**kwargs):
    model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = {})
    cache = SimpleNamespace(num_slots = 4, max_history = 0, reset_states = Mock())
    with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
         patch("exllamav3.generator.generator.ThreadPoolExecutor"):
        return Generator(model, cache, tokenizer = None, **kwargs)


def test_constructor_defaults_and_validation():
    g = make_real_generator()
    assert not g.prefill_priority and g.prefill_slice == 0 and g.decode_slice == 2.0
    g = make_real_generator(prefill_priority = True, prefill_slice = 10, decode_slice = 1.5)
    assert g.prefill_priority and g.prefill_slice == 10 and g.decode_slice == 1.5
    with pytest.raises(AssertionError):
        make_real_generator(prefill_priority = True, prefill_slice = -1)
    with pytest.raises(AssertionError):
        make_real_generator(prefill_priority = True, decode_slice = 0)
