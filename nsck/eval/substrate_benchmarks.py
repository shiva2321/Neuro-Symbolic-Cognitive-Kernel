"""
NSCK Substrate Benchmarks (STEP 6)
====================================
Evaluation harness for the modality-agnostic cognitive substrate.

Provides four benchmark functions:
- ``benchmark_learning_curve`` — per-episode success rate
- ``benchmark_transfer``        — zero-shot / N-shot transfer rate
- ``benchmark_lifelong``        — peak rates + forgetting ratio
- ``benchmark_efficiency``      — latency (avg, p95, max)
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def benchmark_learning_curve(
    engine: Any,
    task_tag: str,
    state_generator: Callable[[], Any],
    n_episodes: int,
    reward_fn: Optional[Callable[[str, Any], float]] = None,
    action_oracle: Optional[Callable[[Any], str]] = None,
) -> List[Tuple[int, float]]:
    """Measure per-episode success rate over *n_episodes*.

    Parameters
    ----------
    engine :
        A :class:`~python.core.reasoning.cognitive_engine.CognitiveEngine`
        instance with *task_tag* already registered.
    task_tag : str
        Registered task domain identifier.
    state_generator : callable
        ``() → state`` — generates a fresh state each episode.
    n_episodes : int
        Total number of episodes to run.
    reward_fn : callable, optional
        ``(action, state) → float`` — computes reward.  Defaults to a
        constant ``+1.0`` (always succeeds).
    action_oracle : callable, optional
        ``(state) → str`` — correct action for success detection.

    Returns
    -------
    list of (episode, success_rate) tuples
        Success rate is computed as a rolling average over the last 10
        episodes (or all episodes so far if < 10).
    """
    if reward_fn is None:
        reward_fn = lambda action, state: 1.0  # noqa: E731
    if action_oracle is None:
        action_oracle = lambda state: None  # noqa: E731

    results: List[Tuple[int, float]] = []
    recent_outcomes: List[float] = []

    for ep in range(n_episodes):
        state = state_generator()
        cs = engine.decide(state, task_tag)
        action = cs.chosen_action
        reward = reward_fn(action, state)
        correct = action_oracle(state)
        success = 1.0 if (correct is None or action == correct) else 0.0
        engine.learn(state, action, reward, task_tag,
                     outcome="success" if success else "failure")

        recent_outcomes.append(success)
        if len(recent_outcomes) > 10:
            recent_outcomes.pop(0)

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            rate = sum(recent_outcomes) / len(recent_outcomes)
            results.append((ep + 1, rate))

    return results


def benchmark_transfer(
    engine: Any,
    source_task: str,
    target_task: str,
    state_generator: Callable[[], Any],
    action_oracle: Optional[Callable[[Any], str]] = None,
    n_zero_shot: int = 20,
    n_few_shot: int = 50,
) -> Dict[str, float]:
    """Measure zero-shot and few-shot transfer rates.

    Parameters
    ----------
    engine :
        A :class:`~python.core.reasoning.cognitive_engine.CognitiveEngine`
        with *source_task* learned and *target_task* just registered.
    source_task, target_task : str
        Task domain identifiers.
    state_generator : callable
        ``() → state`` — generates a fresh state for *target_task*.
    action_oracle : callable, optional
        ``(state) → str`` — correct action.  If None, all actions count.
    n_zero_shot : int
        Number of episodes to evaluate before any fine-tuning (default 20).
    n_few_shot : int
        Number of additional fine-tuning episodes (default 50).

    Returns
    -------
    dict
        ``{"zero_shot_rate": float, "few_shot_rate": float,
           "delta": float}``
    """
    if action_oracle is None:
        action_oracle = lambda state: None  # noqa: E731

    def _eval_rate(n: int) -> float:
        successes = 0
        for _ in range(n):
            state = state_generator()
            cs = engine.decide(state, target_task)
            correct = action_oracle(state)
            if correct is None or cs.chosen_action == correct:
                successes += 1
        return successes / n if n else 0.0

    zero_shot_rate = _eval_rate(n_zero_shot)

    # Fine-tune with a few episodes
    for _ in range(n_few_shot):
        state = state_generator()
        cs = engine.decide(state, target_task)
        engine.learn(state, cs.chosen_action, 1.0, target_task, outcome="success")

    few_shot_rate = _eval_rate(n_zero_shot)  # re-evaluate same size sample

    return {
        "zero_shot_rate": zero_shot_rate,
        "few_shot_rate": few_shot_rate,
        "delta": few_shot_rate - zero_shot_rate,
    }


def benchmark_lifelong(
    engine: Any,
    task_sequence: List[str],
    state_generators: Dict[str, Callable[[], Any]],
    n_episodes_per_task: int = 100,
    n_eval_episodes: int = 20,
) -> Dict[str, Any]:
    """Measure lifelong learning: peak rates and catastrophic forgetting.

    Parameters
    ----------
    engine :
        A :class:`~python.core.reasoning.cognitive_engine.CognitiveEngine`.
    task_sequence : list of str
        Tasks to train in order — each must already be registered.
    state_generators : dict
        ``{task_tag: () → state}`` generators.
    n_episodes_per_task : int
        Training episodes per task (default 100).
    n_eval_episodes : int
        Evaluation episodes after each task switch (default 20).

    Returns
    -------
    dict
        ``{"peak_rates": {task: float},
           "final_rates": {task: float},
           "forgetting_ratio": float}``
    """
    peak_rates: Dict[str, float] = {}
    final_rates: Dict[str, float] = {}

    def _eval_task(tag: str) -> float:
        gen = state_generators.get(tag)
        if gen is None:
            return 0.0
        successes = sum(
            1 for _ in range(n_eval_episodes)
            if engine.decide(gen(), tag) is not None
        )
        return successes / n_eval_episodes

    for task in task_sequence:
        gen = state_generators.get(task)
        if gen is None:
            continue
        for _ in range(n_episodes_per_task):
            state = gen()
            cs = engine.decide(state, task)
            engine.learn(state, cs.chosen_action, 1.0, task, outcome="success")
        engine.sleep(task)
        peak_rates[task] = _eval_task(task)

    # Final evaluation of ALL tasks
    for task in task_sequence:
        final_rates[task] = _eval_task(task)

    # Forgetting ratio: average relative drop from peak to final
    drops = []
    for task in task_sequence:
        peak = peak_rates.get(task, 0.0)
        final = final_rates.get(task, 0.0)
        if peak > 0:
            drops.append(max(0.0, (peak - final) / peak))

    forgetting_ratio = sum(drops) / len(drops) if drops else 0.0

    return {
        "peak_rates": peak_rates,
        "final_rates": final_rates,
        "forgetting_ratio": forgetting_ratio,
    }


def benchmark_efficiency(
    engine: Any,
    task_tag: str,
    state_generator: Callable[[], Any],
    n: int = 100,
) -> Dict[str, float]:
    """Measure decision latency statistics over *n* calls.

    Parameters
    ----------
    engine :
        A :class:`~python.core.reasoning.cognitive_engine.CognitiveEngine`.
    task_tag : str
        Registered task domain identifier.
    state_generator : callable
        ``() → state`` — generates a fresh state each call.
    n : int
        Number of ``decide()`` calls to time (default 100).

    Returns
    -------
    dict
        ``{"avg_ms": float, "p95_ms": float, "max_ms": float}``
    """
    latencies: List[float] = []

    for _ in range(n):
        state = state_generator()
        t0 = time.perf_counter()
        engine.decide(state, task_tag)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_ms = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    max_ms = latencies[-1] if latencies else 0.0

    return {"avg_ms": avg_ms, "p95_ms": p95_ms, "max_ms": max_ms}
