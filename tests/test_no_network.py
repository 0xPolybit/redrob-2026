"""rank.py must be self-contained: no network libraries, no sockets at import.

We block socket creation, import the ranking module, and assert that no
LLM/embedding/HTTP library was pulled in. (The full ranking run is exercised
separately against the built artifacts.)
"""
import socket
import sys

import pytest

FORBIDDEN = ("anthropic", "sentence_transformers", "requests", "httpx", "openai")


def test_rank_imports_no_network_libs(monkeypatch):
    def _blocked(*a, **k):
        raise RuntimeError("network access attempted during rank import")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    # importing the module must not open a socket or pull in a network lib
    for mod in list(sys.modules):
        if mod.startswith("src.rank"):
            del sys.modules[mod]
    import src.rank  # noqa: F401

    for lib in FORBIDDEN:
        assert lib not in sys.modules, f"rank.py transitively imported {lib}"


def test_rank_source_has_no_network_imports():
    import src.rank as rank

    source = open(rank.__file__, encoding="utf-8").read()
    for lib in FORBIDDEN:
        assert f"import {lib}" not in source, f"rank.py references {lib}"
