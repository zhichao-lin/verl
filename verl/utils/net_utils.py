# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import ipaddress
import socket


def is_ipv4(ip_str: str) -> bool:
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def _sock_family(address: str) -> int:
    return socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET


def _bind_tcp(
    address: str,
    port: int,
    family: int | None = None,
    *,
    reuse_addr: bool = False,
) -> socket.socket:
    family = _sock_family(address) if family is None else family
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    if reuse_addr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, port))
    return sock


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the socket open as a port reservation,
    preventing other calls from getting the same port. The caller is
    responsible for closing the socket before the port is actually bound
    by the target service (e.g. NCCL, uvicorn).
    """
    sock = _bind_tcp(address, 0, reuse_addr=True)
    port = sock.getsockname()[1]
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None


def get_free_port_range(
    address: str,
    count: int,
    start: int | None = None,
    *,
    max_attempts: int = 128,
) -> tuple[int, list[socket.socket]]:
    """Reserve ``count`` consecutive TCP ports on ``address``.

    Used by vLLM-Ascend ``MooncakeConnectorV1``, which binds
    ``handshake_port = kv_port + rank`` for ``rank`` in ``[0, tp)``.

    When ``start`` is set, bind exactly ``[start, start+count)``.
    Otherwise pick a free base and retry until the whole range is held.
    Binds without ``SO_REUSEADDR`` so the range is exclusive. Returned sockets
    stay open as reservations; close them before the target service binds.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    if start is not None:
        last = start + count - 1
        if start < 1 or last > 65535:
            raise ValueError(f"port range [{start}, {last}] is not inside 1..65535")
        return start, _bind_consecutive(address, start, count)

    last_error: OSError | None = None
    family = _sock_family(address)
    for _ in range(max_attempts):
        try:
            probe = _bind_tcp(address, 0, family)
        except OSError as exc:
            last_error = exc
            continue
        base = probe.getsockname()[1]
        if base + count - 1 > 65535:
            probe.close()
            continue
        socks = [probe]
        try:
            for offset in range(1, count):
                socks.append(_bind_tcp(address, base + offset, family))
            return base, socks
        except OSError as exc:
            last_error = exc
            for sock in socks:
                sock.close()
    raise RuntimeError(f"could not reserve {count} consecutive ports on {address}") from last_error


def _bind_consecutive(address: str, start: int, count: int) -> list[socket.socket]:
    family = _sock_family(address)
    socks: list[socket.socket] = []
    try:
        for offset in range(count):
            socks.append(_bind_tcp(address, start + offset, family))
        return socks
    except OSError:
        for sock in socks:
            sock.close()
        raise
