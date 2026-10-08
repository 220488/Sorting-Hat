"""Unit/integration tests permit asyncio loopback pipes, never external sockets."""
import socket
import pytest


@pytest.fixture(autouse=True)
def forbid_external_connections(monkeypatch):
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    def check(address):
        if not isinstance(address, tuple) or address[0] not in {'127.0.0.1', '::1', 'localhost'}:
            raise RuntimeError('External network forbidden in router tests')
    def guarded_connect(sock, address):
        check(address)
        return connect(sock, address)
    def guarded_connect_ex(sock, address):
        check(address)
        return connect_ex(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', guarded_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', guarded_connect_ex)
