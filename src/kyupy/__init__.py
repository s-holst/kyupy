"""A package for processing and analysis of non-hierarchical gate-level VLSI designs.

The kyupy package itself contains a logger and other simple utility functions.
In addition, it defines a ``numba`` and a ``cuda`` objects that point to the actual packages
if they are available and otherwise point to mocks.
"""

import time
import importlib.util
import gzip

import numpy as np


_pop_count_lut = np.asarray([bin(x).count('1') for x in range(256)])


def popcount(a):
    """Returns the number of 1-bits in a given packed numpy array."""
    return np.sum(_pop_count_lut[a])


def readtext(file):
    """Reads and returns the text in a given file. Transparently decompresses \\*.gz files."""
    if hasattr(file, 'read'):
        return file.read()
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


class Log:
    """A very simple logger that formats the messages with the number of seconds since
    program start.
    """
    def __init__(self):
        self.start = time.perf_counter()
        self.logfile = None
        """When set to a file handle, log messages are written to it instead to standard output.
        After each write, ``flush()`` is called as well.
        """

    def log(self, level, message):
        t = time.perf_counter() - self.start
        if self.logfile is None:
            print(f'{t:011.3f} {level} {message}')
        else:
            self.logfile.write(f'{t:011.3f} {level} {message}\n')
            self.logfile.flush()

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
                self.log(':', f'{done*100:.0f}% done {hr_time(elapsed_time)} elapsed {hr_time(rem_time)} remaining')
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

    def jit(self, device=False):
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
                                        outer.x = grid_x * block_dim[0] + block_x
                                        outer.y = grid_y * block_dim[1] + block_y
                                        self.func(*args, **kwargs)
                    return inner
            return Launcher(func)

        return make_launcher

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
