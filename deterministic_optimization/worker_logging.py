"""Capture Python and native solver output inside an isolated worker process."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import sys


@contextmanager
def redirect_worker_output(log):
    """Redirect Python streams as well as the native stdout/stderr descriptors.

    On Windows, a console-backed TextIOWrapper uses WriteConsoleW. Redirecting
    its descriptor to a regular file with dup2 does not turn it into a file
    stream. Bind Python output directly to the log during native redirection,
    then restore descriptors before restoring the original Python streams.
    This is process-wide and must only be used in an isolated solver worker.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    saved = {}
    with redirect_stdout(log), redirect_stderr(log):
        try:
            for descriptor in (1, 2):
                saved[descriptor] = os.dup(descriptor)
                os.dup2(log.fileno(), descriptor)
            yield
        finally:
            try:
                log.flush()
            finally:
                for descriptor, backup in saved.items():
                    try:
                        os.dup2(backup, descriptor)
                    finally:
                        os.close(backup)
