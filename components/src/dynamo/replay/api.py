# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from contextlib import ExitStack
from types import ModuleType

from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.routing_policy import ScoreWorkers
from dynamo.replay.waiting_policy import ScoreOrder
from dynamo.replay.waiting_policy import load as _load_waiting_policy


def _normalize_trace_files(trace_files):
    if isinstance(trace_files, (str, os.PathLike)):
        return [trace_files]
    return list(trace_files)


def _waiting_policy_adapter(stack, waiting_policy, log_path):
    """``waiting_policy`` is a policy module or the path of one; the optional
    decision log stays open for the replay."""
    if waiting_policy is None:
        if log_path is not None:
            raise ValueError("waiting_policy_log needs waiting_policy")
        return None
    module = (
        waiting_policy
        if isinstance(waiting_policy, ModuleType)
        else _load_waiting_policy(waiting_policy)
    )
    log = None
    if log_path is not None:
        log = stack.enter_context(open(log_path, "w", encoding="utf-8"))
    return ScoreOrder(module, decision_log=log)


def _planner_config_arg(planner_config):
    """Normalize a planner config to the JSON-string form ``_prepare_planner_replay``
    expects: a dict is json-encoded; a str (path or inline JSON) passes through."""
    if isinstance(planner_config, dict):
        return json.dumps(planner_config)
    return planner_config


def run_trace_replay(
    trace_files,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    trace_block_size=None,
    trace_format="mooncake",
    trace_shared_prefix_ratio=0.0,
    trace_num_prefix_groups=0,
    report_jsonl_path=None,
    max_sim_time_ms=None,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    waiting_policy=None,
    waiting_policy_log=None,
    fpm_log=None,
    routing_policy=None,
    routing_policy_log=None,
):
    """Run trace replay.

    ``waiting_policy`` (a policy module or its path, see
    ``dynamo.replay.waiting_policy``) reorders each worker's waiting queue;
    offline aggregated vLLM replay of a Mooncake trace without
    ``replay_concurrency`` only (not mooncake-delta or agentic). The report then
    carries ``waiting_policy_stats``;
    ``waiting_policy_log`` writes one JSON line per policy call: the per-pass
    ``ctx`` fields, the waiting requests as the policy saw them (every field,
    native order) and the order it chose (``submit_seq``); the engine-argument
    ``ctx`` fields are not written; see ``ScoreOrder``. The log is opened once
    the policy has loaded, before the replay starts, so a call the binding or
    the scheduler rejects still truncates it; a policy that fails to load leaves
    it untouched.
    ``fpm_log`` writes one JSON line per worker iteration: the forward-pass
    snapshot a scaling policy sees per tick (what was prefilled and decoded,
    and what still waited at its start, after its admission) plus ``now_ms``,
    the simulated time the iteration ended.
    ``wall_time`` is the iteration's simulated duration in seconds, so it
    started at ``now_ms - 1000 * wall_time``; ``worker_id`` is the logical
    worker and ``dp_rank`` its attention-DP rank, and ``decode_worker_idx`` in
    the per-request records (like ``worker_id`` in the waiting-policy log) is
    ``worker_id * dp_size + dp_rank``. With attention DP, a rank with no work
    in its group's step logs a line with every count zero and the group's
    ``wall_time``. Offline aggregated replay of a standard Mooncake trace
    without ``replay_concurrency`` only, any engine type. With one worker the
    log, like a waiting policy, runs the multi-worker runtime: per-request
    records carry worker indices, the worker-seconds totals
    (``decode_worker_seconds``, ``gpu_hours``) can differ from the one-worker
    report in the last digits (with ``max_sim_time_ms`` by more: the one-worker
    runtime charges to its last completion, which can lie past the cap, the
    multi-worker runtime to the last event at or before the cap), and
    ``max_sim_time_ms`` stops at the first event past the cap.

    ``routing_policy`` is a Python module/path with ``score(request, worker, ctx)`` and
    optional ``on_route(request, worker, ctx)``. It replaces worker selection, using KV
    placement for tracking. Fixed-worker offline aggregated Mooncake replay,
    DP=1, without router queues, planner, concurrency or aic_perf_config only.
    ``routing_policy_log`` records pre-assignment snapshots, scores and destination.
    A path loads fresh module state per replay; a supplied module retains its state.

    ``wall_time_ms`` and derived throughput measure Rust runtime construction
    and execution. Planner creation and bootstrap happen before that boundary.
    """
    if routing_policy_log is not None and routing_policy is None:
        raise ValueError("routing_policy_log needs routing_policy")
    if routing_policy is not None:
        if planner_config is not None:
            raise ValueError("routing_policy does not support planner_config")
        # Custom scoring uses KV placement's observations and lifecycle accounting.
        router_mode = "kv_router"
    trace_files = _normalize_trace_files(trace_files)
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "trace_block_size": trace_block_size,
        "trace_format": trace_format,
        "trace_shared_prefix_ratio": trace_shared_prefix_ratio,
        "trace_num_prefix_groups": trace_num_prefix_groups,
        "report_jsonl_path": report_jsonl_path,
        "max_sim_time_ms": max_sim_time_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "fpm_log": fpm_log,
    }
    if planner_config is not None:
        # Planner replay is offline-only; reject controls the
        # planner path ignores so callers fail fast instead of silently getting an
        # offline planner run (matches the CLI's guardrails).
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        if trace_format not in ("mooncake", "dynamo"):
            raise ValueError(
                "planner_config replay only supports trace_format='mooncake' or 'dynamo'"
            )
        if report_jsonl_path is not None:
            raise ValueError("report_jsonl_path is not supported with planner_config")
        if max_sim_time_ms is not None:
            raise ValueError("max_sim_time_ms is not supported with planner_config")
        if trace_format != "dynamo" and len(trace_files) != 1:
            raise ValueError(
                f"planner_config replay with trace_format={trace_format!r} "
                "requires exactly one trace file"
            )
        if trace_format == "dynamo" and not trace_files:
            raise ValueError(
                "planner_config replay with trace_format='dynamo' "
                "requires at least one trace file"
            )
    with ExitStack() as stack:
        waiting = _waiting_policy_adapter(stack, waiting_policy, waiting_policy_log)
        routing = None
        if routing_policy is not None:
            log = None
            if routing_policy_log is not None:
                log = stack.enter_context(
                    open(routing_policy_log, "w", encoding="utf-8")
                )
            routing = ScoreWorkers(routing_policy, log)
        adapter = None
        if planner_config is not None:
            from dynamo.replay.main import _planner_replay_adapter

            adapter = stack.enter_context(
                _planner_replay_adapter(
                    extra_engine_args=extra_engine_args,
                    prefill_engine_args=prefill_engine_args,
                    decode_engine_args=decode_engine_args,
                    planner_config_arg=_planner_config_arg(planner_config),
                    benchmark_granularity=benchmark_granularity,
                )
            )
        trace_report = _run_mocker_trace_replay(
            trace_files,
            **replay_kwargs,
            scaling_policy=adapter,
            waiting_policy=waiting,
            **({"routing_policy": routing} if routing is not None else {}),
        )
        if routing is not None:
            trace_report["routing_policy_stats"] = routing.stats
        if waiting is not None:
            trace_report["waiting_policy_stats"] = waiting.stats
        if adapter is not None:
            return adapter.finalize(trace_report)
        return trace_report


def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    request_rate=None,
    arrival_interval_ms=None,
    arrival_seed=42,
    turns_per_session=1,
    shared_prefix_ratio=0.0,
    num_prefix_groups=0,
    inter_turn_delay_ms=0.0,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    waiting_policy=None,
    waiting_policy_log=None,
    fpm_log=None,
):
    """Run synthetic replay with the same timing boundary as trace replay.
    ``waiting_policy``, ``waiting_policy_log`` and ``fpm_log`` as in
    :func:`run_trace_replay`."""
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "request_rate": request_rate,
        "arrival_interval_ms": arrival_interval_ms,
        "arrival_seed": arrival_seed,
        "turns_per_session": turns_per_session,
        "shared_prefix_ratio": shared_prefix_ratio,
        "num_prefix_groups": num_prefix_groups,
        "inter_turn_delay_ms": inter_turn_delay_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "fpm_log": fpm_log,
    }
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
    with ExitStack() as stack:
        waiting = _waiting_policy_adapter(stack, waiting_policy, waiting_policy_log)
        adapter = None
        if planner_config is not None:
            from dynamo.replay.main import _planner_replay_adapter

            adapter = stack.enter_context(
                _planner_replay_adapter(
                    extra_engine_args=extra_engine_args,
                    prefill_engine_args=prefill_engine_args,
                    decode_engine_args=decode_engine_args,
                    planner_config_arg=_planner_config_arg(planner_config),
                    benchmark_granularity=benchmark_granularity,
                )
            )
        trace_report = _run_mocker_synthetic_trace_replay(
            input_tokens,
            output_tokens,
            request_count,
            **replay_kwargs,
            scaling_policy=adapter,
            waiting_policy=waiting,
        )
        if waiting is not None:
            trace_report["waiting_policy_stats"] = waiting.stats
        if adapter is not None:
            return adapter.finalize(trace_report)
        return trace_report
