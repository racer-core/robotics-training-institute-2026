"""
robot_utils.py
RTI 2026 — Shared robot connection management.

Two functions:
  get_rvr()         - Returns the robot connection, creating it if needed.
                       Safe to call from multiple cells / re-run cells; it
                       will not open a second connection on top of an
                       existing one.
  close_if_exists()  - Releases the robot connection if one is open.
                       Safe to call even if nothing is connected.

Why this exists: opening more than one SpheroRvrObserver() on the same
serial port (either by re-running a cell, or by leaving one notebook's
kernel running while opening another notebook) causes competing threads
on /dev/ttyTHS1. These two functions centralize the fix so no individual
notebook has to hand-roll its own connection-safety logic.
"""

import time

from sphero_sdk import SpheroRvrObserver


# Module-level cache. Because Python caches imports per-kernel, this
# persists across cell re-runs within the same notebook/kernel automatically
# -- no reflection into the caller's variable names required.
_rvr = None


def get_rvr(wake=True):
    """Returns the robot connection, creating it if one doesn't already exist.

    If a connection is already open in this kernel, it's returned as-is --
    no new connection is attempted, and wake() is not called again.

    Args:
        wake (bool): If True (default), wakes the robot after connecting.
            Only applies the first time a connection is made.

    Returns:
        SpheroRvrObserver: The robot connection.
    """
    global _rvr

    if _rvr is not None:
        return _rvr

    try:
        candidate = SpheroRvrObserver()
    except Exception:
        print("⚠️  Couldn't connect to the robot.")
        print("    This usually means another notebook still has the robot's")
        print("    connection open. Go to that notebook, then Kernel menu ->")
        print("    'Shut Down Kernel', and try running this cell again.")
        raise

    if wake:
        time.sleep(2)
        candidate.wake()

    _rvr = candidate
    return _rvr


def close_if_exists():
    """Releases the robot connection if one is open. Safe to call even if
    nothing is connected -- does nothing in that case.

    Run this at the end of a notebook (or before switching to a different
    one) so the next notebook doesn't run into a busy serial port.
    """
    global _rvr

    if _rvr is None:
        return

    try:
        _rvr.close()
    finally:
        _rvr = None
