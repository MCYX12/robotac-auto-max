#!/usr/bin/env python3
"""Cancel one check/audit command without stopping the simulation or mission."""
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from common import atomic_json, read_json, process_start_ticks, supervisor_alive, stop_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'cancel'))
    parser.add_argument('token')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9-]{1,100}', args.token):
        parser.error('invalid observer token')
    record = Path('/tmp') / ('robotac-observer-' + args.token + '.json')
    request = record.with_suffix('.cancel')
    if args.action == 'cancel':
        request.touch()
        state = read_json(record)
        if supervisor_alive(state, str(Path(__file__).resolve())):
            try:os.kill(state['supervisor_pid'], signal.SIGTERM)
            except ProcessLookupError:pass
        return 0
    if not args.command:
        parser.error('observer command is required')
    cancelled = [False]
    child = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    try:
        atomic_json(record, {'supervisor_pid': os.getpid(), 'supervisor_start_ticks': process_start_ticks(os.getpid())}, durable=False)
        if request.exists() or cancelled[0]:
            return 130
        child = subprocess.Popen(args.command, start_new_session=True)
        while child.poll() is None:
            if request.exists() or cancelled[0]:
                return 130
            time.sleep(.1)
        return 130 if request.exists() or cancelled[0] else child.returncode
    finally:
        if child is not None and child.poll() is None:
            stop_group(child)
        record.unlink(missing_ok=True)
        request.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
