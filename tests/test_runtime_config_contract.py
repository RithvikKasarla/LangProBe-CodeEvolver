"""Contract between the shipped runtime_config.json and the reader in common.py.

Every other config test (test_lm_fallback.py) monkeypatches `load_runtime_config`
with a hand-written dict, so nothing ever loads the real
codeevolver_benchmarks/runtime_config.json. A regression in the shipped file --
a renamed key, a dropped `routes` block, a `provider_preference` entry naming a
provider that no longer exists -- would sail through the suite untouched. This
file reads the real file and pins the invariants common.py actually relies on.

No network, no mocks: `_routes_for`, `resolve_provider`, and `resolve_fallback`
are pure functions over the config dict (they never touch an API key or make a
request), so they run against the shipped config directly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from codeevolver_benchmarks import common
from codeevolver_benchmarks.common import _routes_for, resolve_fallback, resolve_provider


def _load_full_config() -> dict[str, Any]:
    with common._CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


CONFIG = _load_full_config()
BLOCK_NAMES = sorted(CONFIG)
MULTI_ROUTE_BLOCK_NAMES = sorted(name for name in BLOCK_NAMES if CONFIG[name].get("routes"))

# Blocks with no `routes` key are legitimately single-provider (see
# `_routes_for`'s docstring in common.py). Name each one explicitly, with why,
# so a block that silently LOSES its routes -- rather than one deliberately
# designed without them -- fails test_routes_and_provider_preference_are_declared_together
# below instead of quietly passing as "single-provider by accident".
SINGLE_ROUTE_BLOCKS = {
    "alfworld_vision": (
        "OpenRouter is the only route; its own provider pool is the "
        "failover, so a second explicit route would just be OpenRouter twice."
    ),
}


def test_config_file_has_the_two_shipped_blocks():
    assert set(CONFIG) >= {"appworld", "alfworld_vision"}


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_load_runtime_config_matches_the_file_on_disk(name):
    """The loader `common.py` actually calls must agree with a direct read."""
    assert common.load_runtime_config(name) == CONFIG[name]


# --- structural shape --------------------------------------------------------


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_routes_and_provider_preference_are_declared_together(name):
    """One without the other is a silent misconfiguration.

    `routes` with no `provider_preference` means resolve_fallback's preference
    list defaults to empty, so no cover is ever selected even though routes
    exist to serve as one. `provider_preference` with no `routes` means
    _routes_for never builds a routing table for it in the first place. A
    block with neither is single-provider, allowed only via the named
    SINGLE_ROUTE_BLOCKS allowlist above.
    """
    block = CONFIG[name]
    has_routes = bool(block.get("routes"))
    has_preference = "provider_preference" in block
    assert has_routes == has_preference, (
        f"{name!r} declares one of routes/provider_preference without the other"
    )
    if not has_routes:
        assert name in SINGLE_ROUTE_BLOCKS, (
            f"{name!r} has no routes and is not in the SINGLE_ROUTE_BLOCKS "
            "allowlist -- either it silently lost its routes, or it's "
            "intentionally single-provider and needs a named, commented "
            "entry there"
        )


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_provider_preference_and_provider_preferences_have_the_right_types(name):
    """`provider_preference` and `provider_preferences` differ by one letter
    and are read four lines apart in common.py: the singular is the list
    resolve_fallback walks to pick a cover (`config.get("provider_preference",
    [])`); the plural is the dict _build_request forwards verbatim as
    OpenRouter's extra_body (`config["provider_preferences"]`). Swapping them,
    or writing the wrong type under either name, breaks routing or the
    passthrough with no error until a live call.
    """
    block = CONFIG[name]
    if "provider_preference" in block:
        assert isinstance(block["provider_preference"], list)
    if "provider_preferences" in block:
        assert isinstance(block["provider_preferences"], dict)


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_routes_build_without_a_missing_key(name):
    """_routes_for raises RuntimeError naming the exact missing key when a
    route (or, for a single-route block, the flat top-level fields) drops one
    of model/base_url/api_key_env. Building routes for every shipped block
    must not raise, and every built Route must have non-empty string fields.
    """
    routes = _routes_for(CONFIG[name])
    assert routes, f"{name!r} produced no routes at all"
    for route in routes.values():
        assert route.model and isinstance(route.model, str)
        assert route.base_url and isinstance(route.base_url, str)
        assert route.api_key_env and isinstance(route.api_key_env, str)


# --- routing -------------------------------------------------------------


@pytest.mark.parametrize("name", MULTI_ROUTE_BLOCK_NAMES)
def test_every_preferred_provider_has_a_route(name):
    block = CONFIG[name]
    routes = _routes_for(block)
    for provider in block["provider_preference"]:
        assert provider in routes, (
            f"{name!r} lists {provider!r} in provider_preference but it has no route"
        )


@pytest.mark.parametrize("name", MULTI_ROUTE_BLOCK_NAMES)
def test_primary_provider_is_routed_and_preferred(name):
    block = CONFIG[name]
    routes = _routes_for(block)
    assert block["provider"] in routes, f"{name!r}'s primary provider has no route"
    assert block["provider"] in block["provider_preference"], (
        f"{name!r}'s primary provider is missing from its own provider_preference"
    )


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_provider_resolves_without_env_overrides(monkeypatch, name):
    """resolve_provider/resolve_fallback against the shipped config, with no
    $LM_PROVIDER or $LM_FALLBACK set -- the defaults every real run starts from.
    """
    monkeypatch.delenv("LM_PROVIDER", raising=False)
    monkeypatch.delenv("LM_FALLBACK", raising=False)
    block = CONFIG[name]
    routes = _routes_for(block)
    primary = resolve_provider(block, routes)
    assert primary.provider == block["provider"]
    fallback = resolve_fallback(block, routes, primary)
    if name in SINGLE_ROUTE_BLOCKS:
        assert fallback is None
    else:
        assert fallback is not None, f"{name!r} has multiple routes but resolved no cover"
        assert fallback.provider != primary.provider


# --- request construction ---------------------------------------------------


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_request_keys_are_present_and_typed(name):
    """_build_request reads reasoning_effort and max_tokens unconditionally
    (a KeyError there is a crash, not a graceful degrade) and temperature/seed
    only if present, forwarding whatever type is in the config straight into
    the API request.
    """
    block = CONFIG[name]
    assert isinstance(block["reasoning_effort"], str) and block["reasoning_effort"]
    assert isinstance(block["max_tokens"], int) and block["max_tokens"] > 0
    if "temperature" in block:
        assert isinstance(block["temperature"], (int, float))
    if "seed" in block:
        assert isinstance(block["seed"], int)


# --- known-good shape (lock-in) ---------------------------------------------
#
# The generic checks above hold for any future block. These two pin the
# *current* shipped shape by name, per the incident this file exists to catch.


def test_appworld_is_multi_route_with_gmi_primary_and_a_three_way_preference():
    block = CONFIG["appworld"]
    assert block["provider"] == "gmi"
    assert set(block["provider_preference"]) == set(block["routes"]) == {
        "gmi",
        "deepseek",
        "deepinfra",
    }
    assert block["provider_preference"][0] == "gmi"


def test_alfworld_vision_is_intentionally_single_route_openrouter():
    block = CONFIG["alfworld_vision"]
    assert "alfworld_vision" in SINGLE_ROUTE_BLOCKS
    assert not block.get("routes")
    assert "provider_preference" not in block
    assert block["provider"] == "openrouter"
    assert isinstance(block["provider_preferences"], dict)
    # NB: `jpeg_quality` sits in this block but, as of this writing, no
    # codeevolver_benchmarks/*.py reads it (only `max_steps` is pulled off
    # the alfworld_vision config outside common.py). Not asserted on here --
    # an unused key isn't a routing contract violation -- but worth a look:
    # either something that should consume it was never wired up, or it is
    # dead and safe to delete.
    assert "jpeg_quality" in block
