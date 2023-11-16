from functools import wraps
from time import time


# decorator based on https://stackoverflow.com/a/27737385/7521429
def measure_time_old(f):
    @wraps(f)
    def timed(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()

        print(f"{f.__qualname__} took {te - ts:.8f} sec")

        return result

    return timed


def measure_time(f):
    @wraps(f)
    def timed(*args, **kw):
        with Timer(f.__qualname__):
            result = f(*args, **kw)

        return result

    return timed


def _safe_percentage(value, total):
    if total == 0:
        return 100
    else:
        return value / total * 100


class TimingContext:
    def __init__(self, name, parent_ctx=None):
        self.name = name
        self.duration = 0
        self.call_count = 0
        self.child_contexts = dict()
        self.parent_ctx = parent_ctx

    def __repr__(self):
        return f"{self.name}: {self.duration:.8f} sec"

    def __str__(self):
        """Prints the tree of timing contexts"""
        return self._str_helper(0)

    def _str_helper(self, indent, parent_duration=None):
        if parent_duration is None:
            parent_duration = self.duration

        indent_str = "  " * indent
        rem_indent_str = "  " * (indent + 1)
        result = f"{indent_str}{self.name}: {self.duration:.8f} sec ({_safe_percentage(self.duration, parent_duration):.2f}%) [{self.call_count}]\n"
        for child_ctx in self.child_contexts.values():
            result += child_ctx._str_helper(indent + 1, self.duration)

        if len(self.child_contexts) > 0:
            rem_duration = self._remaining_duration()
            if rem_duration > 0:
                remainder_str = f"{rem_indent_str}... {rem_duration:.8f} sec ({_safe_percentage(self._remaining_duration(), self.duration):.2f}%)\n"
                result += remainder_str

        return result

    def _remaining_duration(self):
        return self.duration - self._child_duration()

    def _child_duration(self):
        return sum(child_ctx.duration for child_ctx in self.child_contexts.values())


class Timer:
    global_ctx = TimingContext("global")
    current_ctx = global_ctx

    def __init__(self, name):
        self.name = name
        if name not in Timer.current_ctx.child_contexts:
            self.ctx = TimingContext(name, self.current_ctx)
            Timer.current_ctx.child_contexts[name] = self.ctx
        else:
            self.ctx = Timer.current_ctx.child_contexts[name]

    def __enter__(self):
        Timer.current_ctx = self.ctx
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        self.ctx.duration += self.end - self.start
        self.ctx.call_count += 1
        Timer.current_ctx = self.ctx.parent_ctx

    @classmethod
    def print(cls):
        print(cls)

    @classmethod
    def __str__(cls):
        cls.global_ctx.duration = cls.global_ctx._child_duration()
        return str(cls.global_ctx)

    @classmethod
    def reset(cls):
        cls.global_ctx = TimingContext("global")
        cls.current_ctx = cls.global_ctx
