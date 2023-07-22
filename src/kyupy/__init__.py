"""The kyupy package itself contains a miscellaneous utility functions.

In addition, it defines a ``numba`` and a ``cuda`` objects that point to the actual packages
if they are available and otherwise point to mocks.
"""

import time
import sys
from collections import defaultdict
import importlib.util
import gzip

import numpy as np


_pop_count_lut = np.asarray([bin(x).count('1') for x in range(256)])


def cdiv(x, y):
    return -(x // -y)


def popcount(a):
    """Returns the number of 1-bits in a given packed numpy array of type ``uint8``."""
    return np.sum(_pop_count_lut[a])


def readtext(file):
    """Reads and returns the text in a given file. Transparently decompresses \\*.gz files."""
    if hasattr(file, 'read'):
        return file.read().decode()
    if str(file).endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            return f.read()
    else:
        with open(file, 'rt') as f:
            return f.read()


def hr_sci(value):
    """Formats a value in a human-readible scientific notation."""
    multiplier = 0
    while abs(value) >= 1000:
        value /= 1000
        multiplier += 1
    while abs(value) < 1:
        value *= 1000
        multiplier -= 1
    return f'{value:.3f}{" kMGTPEafpnµm"[multiplier]}'


def hr_bytes(nbytes):
    """Formats a given number of bytes for human readability."""
    multiplier = 0
    while abs(nbytes) >= 1000:
        nbytes /= 1024
        multiplier += 1
    return f'{nbytes:.1f}{["", "ki", "Mi", "Gi", "Ti", "Pi"][multiplier]}B'


def hr_time(seconds):
    """Formats a given time interval for human readability."""
    s = ''
    if seconds >= 86400:
        d = seconds // 86400
        seconds -= d * 86400
        s += f'{int(d)}d'
    if seconds >= 3600:
        h = seconds // 3600
        seconds -= h * 3600
        s += f'{int(h)}h'
    if seconds >= 60:
        m = seconds // 60
        seconds -= m * 60
        if 'd' not in s:
            s += f'{int(m)}m'
    if 'h' not in s and 'd' not in s:
        s += f'{int(seconds)}s'
    return s


def batchrange(nitems, maxsize):
    """A simple generator that produces offsets and sizes for batch-loops."""
    for offset in range(0, nitems, maxsize):
        yield offset, min(nitems-offset, maxsize)


class Timer:
    def __init__(self, s=0): self.s = s
    def __enter__(self): self.start_time = time.perf_counter(); return self
    def __exit__(self, *args): self.s += time.perf_counter() - self.start_time
    @property
    def ms(self): return self.s*1e3
    @property
    def us(self): return self.s*1e6
    def __repr__(self): return f'{self.s:.3f}'
    def __add__(self, t):
        return Timer(self.s + t.s)


class Timers:
    def __init__(self, t={}): self.timers = defaultdict(Timer) | t
    def __getitem__(self, name): return self.timers[name]
    def __repr__(self): return '{' + ', '.join([f'{k}: {v}' for k, v in self.timers.items()]) + '}'
    def __add__(self, t):
        tmr = Timers(self.timers)
        for k, v in t.timers.items(): tmr.timers[k] += v
        return tmr
    def sum(self):
        return sum([v.s for v in self.timers.values()])
    def dict(self):
        return dict([(k, v.s) for k, v in self.timers.items()])


class Log:
    """A very simple logger that formats the messages with the number of seconds since
    program start.
    """

    def __init__(self):
        self.start = time.perf_counter()
        self.logfile = sys.stdout
        """When set to a file handle, log messages are written to it instead to standard output.
        """
        self.indent = 0
        self._limit = -1
        self.filtered = 0

    def limit(self, log_limit):
        class Limiter:
            def __init__(self, l): self.l = l
            def __enter__(self): self.l.start_limit(log_limit); return self
            def __exit__(self, *args): self.l.stop_limit()
        return Limiter(self)

    def start_limit(self, limit):
        self.filtered = 0
        self._limit = limit

    def stop_limit(self):
        if self.filtered > 0:
            log.info(f'{self.filtered} more messages (filtered).')
            self.filtered = 0
        self._limit = -1

    def __getstate__(self):
        return {'elapsed': time.perf_counter() - self.start}

    def __setstate__(self, state):
        self.logfile = sys.stdout
        self.indent = 0
        self.start = time.perf_counter() - state['elapsed']

    def write(self, s, indent=0):
        self.logfile.write(' '*indent + s + '\n')
        self.logfile.flush()

    def li(self, item): self.write('- ' + str(item).replace('\n', '\n'+' '*(self.indent+1)), self.indent)
    def lib(self): self.write('-', self.indent); self.indent += 1
    def lin(self): self.write('-', self.indent-1)
    def di(self, key, value): self.write(str(key) + ': ' + str(value).replace('\n', '\n'+' '*(self.indent+1)), self.indent)
    def dib(self, key): self.write(str(key) + ':', self.indent); self.indent += 1
    def din(self, key): self.write(str(key) + ':', self.indent-1)
    def ie(self, n=1): self.indent -= n

    def log(self, level, message):
        if self._limit == 0:
            self.filtered += 1
            return
        t = time.perf_counter() - self.start
        self.logfile.write(f'# {t:011.3f} {level} {message}\n')
        self.logfile.flush()
        self._limit -= 1

    def info(self, message):
        """Log an informational message."""
        self.log('-', message)

    def warn(self, message):
        """Log a warning message."""
        self.log('W', message)

    def error(self, message):
        """Log an error message."""
        self.log('E', message)

    def range(self, *args):
        """A generator that operates just like the ``range()`` built-in, and also occasionally logs the progress
        and compute time estimates."""
        elems = len(range(*args))
        start_time = time.perf_counter()
        lastlog_time = start_time
        log_interval = 5
        for elem, i in enumerate(range(*args)):
            yield i
            current_time = time.perf_counter()
            if current_time > lastlog_time + log_interval:
                done = (elem + 1) / elems
                elapsed_time = current_time - start_time
                total_time = elapsed_time / done
                rem_time = total_time - elapsed_time
                self.log(
                    ':', f'{done*100:.0f}% done {hr_time(elapsed_time)} elapsed {hr_time(rem_time)} remaining')
                log_interval = min(600, int(log_interval*1.5))
                lastlog_time = current_time


log = Log()
"""The standard logger instance."""


#
# Code below mocks basic numba and cuda functions for pure-python fallback.
#

class MockNumba:
    @staticmethod
    def njit(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


class MockCuda:

    def __init__(self):
        self.x = 0
        self.y = 0

    def jit(self, func=None, device=False):
        _ = device  # silence "not used" warning
        outer = self

        def make_launcher(func):
            class Launcher:
                def __init__(self, funcc):
                    self.func = funcc

                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)

                def __getitem__(self, item):
                    grid_dim, block_dim = item

                    def inner(*args, **kwargs):
                        for grid_x in range(grid_dim[0]):
                            for grid_y in range(grid_dim[1]):
                                for block_x in range(block_dim[0]):
                                    for block_y in range(block_dim[1]):
                                        outer.x = grid_x * \
                                            block_dim[0] + block_x
                                        outer.y = grid_y * \
                                            block_dim[1] + block_y
                                        self.func(*args, **kwargs)
                    return inner
            return Launcher(func)

        return make_launcher(func) if func else make_launcher

    @staticmethod
    def to_device(array, to=None):
        if to is not None:
            to[...] = array
            return to
        return array.copy()

    def synchronize(self):
        pass

    def grid(self, dims):
        _ = dims  # silence "not used" warning
        return self.x, self.y


if importlib.util.find_spec('numba') is not None:
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.error import CudaSupportError
    try:
        list(numba.cuda.gpus)
        from numba import cuda
        from numba.core import config
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    except CudaSupportError:
        log.warn('Cuda unavailable. Falling back to pure Python.')
        cuda = MockCuda()
else:
    numba = MockNumba()
    """If Numba is available on the system, it is the actual ``numba`` package.
    Otherwise, it simply defines an ``njit`` decorator that does nothing.
    """
    cuda = MockCuda()
    """If Numba is installed and Cuda GPUs are available, it is the actual ``numba.cuda`` package.
    Otherwise, it is an object that defines basic methods and decorators so that cuda-code can still
    run in the Python interpreter.
    """
    log.warn('Numba unavailable. Falling back to pure Python.')
