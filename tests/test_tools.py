"""Behavior of the composite provider's advertised tools and backend routing."""

import json
from unittest.mock import Mock

import pytest
from conftest import make_provider

from mnemosyne import config


@pytest.fixture
def provider(monkeypatch):
    # Exercise provider defaults without reading local configuration or env vars.
    monkeypatch.setattr(config, "get", lambda *keys, default=None: default)
    instance = make_provider()
    instance._honcho = Mock()
    instance._hindsight = Mock()
    yield instance
    instance._executor.shutdown(wait=True)


def test_default_tool_order(provider):
    assert [tool["name"] for tool in provider.get_tool_schemas()] == [
        "memory_profile",
        "memory_reasoning",
        "memory_conclude",
        "memory_recall",
        "memory_reflect",
        "memory_forget",
    ]


@pytest.mark.parametrize("writes_allowed", [True, False])
def test_configured_exposure_preserves_order_and_duplicates(
    provider, monkeypatch, writes_allowed
):
    exposed = [
        "memory_forget",
        "unknown",
        "memory_reflect",
        "memory_profile",
        "memory_conclude",
        "memory_reflect",
    ]
    monkeypatch.setattr(
        config,
        "get",
        lambda *keys, default=None: exposed if keys == ("tools", "expose") else default,
    )
    provider._writes_allowed = writes_allowed
    expected = (
        [
            "memory_forget",
            "memory_reflect",
            "memory_profile",
            "memory_conclude",
            "memory_reflect",
        ]
        if writes_allowed
        else ["memory_reflect", "memory_profile", "memory_reflect"]
    )
    assert [tool["name"] for tool in provider.get_tool_schemas()] == expected


@pytest.mark.parametrize("timeout", [0, 5])
@pytest.mark.parametrize(
    "tool,backend,inner,args",
    [
        ("memory_profile", "honcho", "honcho_profile", {}),
        ("memory_reasoning", "honcho", "honcho_reasoning", {"query": "habits"}),
        ("memory_conclude", "honcho", "honcho_conclude", {"conclusion": "tea"}),
        ("memory_recall", "hindsight", "hindsight_recall", {"query": "deploy"}),
        ("memory_reflect", "hindsight", "hindsight_reflect", {"query": "deploy"}),
    ],
)
def test_tool_dispatch_forwards_arguments(
    provider, monkeypatch, timeout, tool, backend, inner, args
):
    monkeypatch.setattr(provider, "_timeout_for", lambda name: timeout)
    target = getattr(provider, "_" + backend)
    target.handle_tool_call.return_value = json.dumps({"result": "Deployment complete"})

    result = provider.handle_tool_call(tool, args, session_id="test-session")

    target.handle_tool_call.assert_called_once_with(
        inner, args, session_id="test-session"
    )
    other = provider._hindsight if backend == "honcho" else provider._honcho
    other.handle_tool_call.assert_not_called()
    assert "Deployment complete" in json.loads(result)["result"]


def test_forget_uses_local_handler(provider, monkeypatch):
    handler = Mock(return_value='{"forgotten": []}')
    monkeypatch.setattr(provider, "_handle_forget", handler)
    args = {"query": "tea"}

    assert provider.handle_tool_call("memory_forget", args) == '{"forgotten": []}'
    handler.assert_called_once_with(args)
    provider._honcho.handle_tool_call.assert_not_called()
    provider._hindsight.handle_tool_call.assert_not_called()


def test_profile_read_remains_available_without_writes(provider):
    provider._writes_allowed = False
    provider._honcho.handle_tool_call.return_value = '{"result": ["tea"]}'

    assert json.loads(provider.handle_tool_call("memory_profile", {})) == {
        "result": ["tea"]
    }
    provider._honcho.handle_tool_call.assert_called_once_with("honcho_profile", {})


@pytest.mark.parametrize(
    "tool,key",
    [
        ("memory_recall", "recall"),
        ("memory_reasoning", "reasoning"),
        ("memory_reflect", "reflect"),
        ("memory_profile", "profile"),
        ("memory_conclude", "conclude"),
        ("memory_forget", "forget"),
    ],
)
def test_tool_timeout_uses_configured_key(provider, monkeypatch, tool, key):
    monkeypatch.setattr(
        config,
        "get",
        lambda *keys, default=None: "7.5" if keys == ("timeouts", key) else default,
    )
    assert provider._timeout_for(tool) == 7.5


def test_unknown_tool_is_rejected(provider):
    with pytest.raises(NotImplementedError, match="does not handle tool unknown"):
        provider.handle_tool_call("unknown", {})
    assert provider._timeout_for("unknown") is None
    provider._honcho.handle_tool_call.assert_not_called()
    provider._hindsight.handle_tool_call.assert_not_called()
