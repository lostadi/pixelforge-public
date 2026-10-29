"""Linux pidfd operations for Android Python builds lacking the stdlib binding."""
import ctypes
import os
import signal


def _syscall(number, *arguments):
    if os.uname().machine != 'aarch64':
        raise RuntimeError('This Android syscall fallback is restricted to aarch64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number),
                          *(ctypes.c_long(value) for value in arguments))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    return _syscall(434, pid, 0)


def send_pidfd_signal(fd, signum):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, signum)
    return _syscall(424, fd, signum, 0, 0)
