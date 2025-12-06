#!/usr/bin/env python3
"""
Send sheath-insert matrices to udp_recv_sheath_insert.py.

Example:
    python publish_insert_distance.py --distance 12.5 --rate 20
"""
from __future__ import annotations

import argparse
import socket
import time
from typing import Iterable

import numpy as np


def build_matrix(distance_mm: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[2, 3] = distance_mm  # encode insert distance along Z
    return matrix[:3, :]


def format_payload(matrix: np.ndarray) -> bytes:
    flat: Iterable[float] = matrix.reshape(-1)
    body = ",".join(f"{value:.6f}" for value in flat) + "\n"
    return body.encode("utf-8")


def publish(distance: float, host: str, port: int, rate_hz: float) -> None:
    payload = format_payload(build_matrix(distance))
    interval = 1.0 / max(rate_hz, 1e-9)
    addr = (host, port)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        print(f"Publishing insert distance {distance} mm to {host}:{port} @ {rate_hz} Hz")
        try:
            while True:
                sock.sendto(payload, addr)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped publisher.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP sheath insert distance publisher.")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9527, help="Receiver port (default 9527)")
    parser.add_argument("--distance", type=float, default=0.0, help="Insert distance in mm")
    parser.add_argument("--rate", type=float, default=30.0, help="Send rate in Hz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    publish(args.distance, args.host, args.port, args.rate)