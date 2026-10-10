"""Linux x86_64 computation isolation ported from the frozen SDE tool bridge."""

import ctypes
import errno
import os
import platform
import resource
import sys
from pathlib import Path


RUNTIME = [
    Path(p)
    for p in [
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc/ssl",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/ld.so.cache",
        "/etc/localtime",
        "/dev/null",
        "/dev/urandom",
        str(Path(sys.executable).absolute().parent.parent),
        str(Path(sys.executable).resolve().parent.parent),
    ]
    if Path(p).exists()
]


def restrict_fs(work: Path, extra_reads=()):
    """Fail closed when Landlock cannot enforce reads, writes and execution."""
    lib = ctypes.CDLL(None, use_errno=True)
    lib.syscall.restype = ctypes.c_long
    abi = lib.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Landlock ABI >=3 required")
    rights = (1 << 15) - 1  # EXECUTE through TRUNCATE (ABI 3).

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    mask = ctypes.c_uint64(rights)
    fd = lib.syscall(444, ctypes.byref(mask), ctypes.sizeof(mask), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "create Landlock ruleset")
    try:
        for path, access in [(p, 13) for p in RUNTIME + list(extra_reads)] + [
            (work.resolve(), rights)
        ]:
            # A file rule may not include directory-only access bits.
            bits = (
                access
                if path.is_dir()
                else access & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14))
            )
            parent = os.open(path, getattr(os, "O_PATH") | os.O_CLOEXEC)
            try:
                rule = Rule(bits, parent)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "add Landlock rule")
            finally:
                os.close(parent)
        if lib.prctl(38, 1, 0, 0, 0) != 0 or lib.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "apply Landlock ruleset")
    finally:
        os.close(fd)


def deny_network():
    """Commands need local computation; web access goes through the audited MCP tool."""
    lib = ctypes.CDLL(None, use_errno=True)

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    instructions = [
        Filter(0x20, 0, 0, 4),
        Filter(0x15, 1, 0, 0xC000003E),
        Filter(0x06, 0, 0, 0x80000000),
        Filter(0x20, 0, 0, 0),
        Filter(0x45, 0, 1, 0x40000000),
        Filter(0x06, 0, 0, 0x80000000),
    ]
    # x86_64 socket, namespace, mounting, tracing and foreign-process memory.
    for nr in [
        41,
        42,
        43,
        49,
        50,
        53,
        101,
        155,
        165,
        166,
        272,
        288,
        304,
        308,
        310,
        311,
    ]:
        instructions += [
            Filter(0x15, 0, 1, nr),
            Filter(0x06, 0, 0, 0x50000 | errno.EPERM),
        ]
    instructions += [Filter(0x06, 0, 0, 0x7FFF0000)]
    filters = (Filter * len(instructions))(*instructions)
    program = Program(len(filters), filters)
    if (
        lib.prctl(38, 1, 0, 0, 0) != 0
        or lib.prctl(22, 2, ctypes.byref(program), 0, 0) != 0
    ):
        raise OSError(ctypes.get_errno(), "apply seccomp")


def launch(work: Path, command: str) -> None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Scientific computation requires Linux x86_64 isolation")
    work = work.resolve()
    os.chdir(work)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024**2, 20 * 1024**2))
    restrict_fs(work)
    deny_network()
    os.execve(
        "/bin/bash",
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "HOME": str(work),
            "TMPDIR": str(work),
            "LANG": "C.UTF-8",
        },
    )


if __name__ == "__main__":
    launch(Path(sys.argv[1]), sys.argv[2])
