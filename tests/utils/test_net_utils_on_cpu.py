# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket

import pytest

from verl.utils.net_utils import get_free_port, get_free_port_range


def _close(socks: list[socket.socket] | None) -> None:
    for sock in socks or []:
        sock.close()


def test_get_free_port_range_reserves_consecutive_ports():
    socks = None
    try:
        base, socks = get_free_port_range("127.0.0.1", count=3)
        assert [s.getsockname()[1] for s in socks] == [base, base + 1, base + 2]
    finally:
        _close(socks)


def test_get_free_port_range_honors_explicit_start():
    probe, probe_sock = get_free_port("127.0.0.1", with_alive_sock=True)
    probe_sock.close()
    socks = None
    try:
        base, socks = get_free_port_range("127.0.0.1", count=2, start=probe)
        assert base == probe
        assert [s.getsockname()[1] for s in socks] == [probe, probe + 1]
    finally:
        _close(socks)


def test_get_free_port_range_rejects_occupied_start():
    holder, holder_sock = get_free_port("127.0.0.1", with_alive_sock=True)
    try:
        with pytest.raises(OSError):
            get_free_port_range("127.0.0.1", count=1, start=holder)
    finally:
        holder_sock.close()


def test_get_free_port_range_rejects_bad_count_and_overflow():
    with pytest.raises(ValueError, match="count"):
        get_free_port_range("127.0.0.1", count=0)
    with pytest.raises(ValueError, match="65535"):
        get_free_port_range("127.0.0.1", count=4, start=65534)
