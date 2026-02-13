"""
NSCK Transfer Learning Experiments
====================================
Rigorous test suite for validating cross-domain transfer learning.

6 Experiment Types:
  1. Full Transfer Matrix (all game pairs)
  2. Statistical Significance (multi-run t-test)
  3. Learning Curve (scaling)
  4. Ablation Studies (predicate removal, bin resolution)
  5. Negative Transfer Detection (adversarial + null)
  6. Curriculum Learning (chained training)

Usage:
  python transfer_experiments.py --quick            # Fast smoke test (~10s)
  python transfer_experiments.py --full             # Full suite (~1-2min)
  python transfer_experiments.py --exp 1,3,5        # Specific experiments
  python transfer_experiments.py --output report.md # Custom output
"""
import os
import sys
import json
import time
import random
import argparse
import math
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../')))

from python.benchmarks.benchmark import (
    ENVS, TeacherAgent, StudentAgent, RandomAgent, run_training, run_testing, run_random_baseline,
    SymbolicHasher, LearnedPolicy, NaturalLanguageTeacher, BenchmarkRunner,
    BalancerEnv, CatcherEnv
)


# =============================================================================
# Core Transfer Test Runner
# =============================================================================

def run_training(source_game: str, episodes: int, hasher=None, seed=None, policy_override=None):
    """Train a policy on source game. Returns (policy, avg_score)."""
    if seed is not None:
        random.seed(seed)
    env_cls = ENVS.get(source_game)
    if not env_cls:
        raise ValueError(f"Unknown game: {source_game}")
    env = env_cls()
    teacher = TeacherAgent()
    policy = policy_override if policy_override is not None else LearnedPolicy()
    h_fn = hasher.hash if hasher else SymbolicHasher.default_hash

    total_score = 0
    for _ in range(episodes):
        state = env.reset()
        for _ in range(600):
            action = teacher.choose_action(source_game, state)
            h = h_fn(source_game, state)
            policy.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break
        total_score += env.score

    return policy, total_score / max(episodes, 1)


def run_testing(target_game: str, policy, episodes: int, hasher=None, seed=None):
    """Test a trained policy on target game. Returns list of scores."""
    if seed is not None:
        random.seed(seed)
    env_cls = ENVS.get(target_game)
    if not env_cls:
        raise ValueError(f"Unknown game: {target_game}")
    env = env_cls()
    student = StudentAgent(policy)
    h_fn = hasher.hash if hasher else SymbolicHasher.default_hash

    scores = []
    for _ in range(episodes):
        state = env.reset()
        for _ in range(600):
            h = h_fn(target_game, state)
            learned = policy.predict(h)
            if learned:
                action = learned
            else:
                action = RandomAgent().choose_action(target_game, state)
            state, reward, done = env.step(action)
            if done:
                break
        scores.append(env.score)
    return scores


def run_random_baseline(target_game: str, episodes: int, seed=None):
    """Run random agent on target game. Returns list of scores."""
    if seed is not None:
        random.seed(seed)
    env_cls = ENVS.get(target_game)
    env = env_cls()
    agent = RandomAgent()
    scores = []
    for _ in range(episodes):
        state = env.reset()
        for _ in range(600):
            action = agent.choose_action(target_game, state)
            state, reward, done = env.step(action)
            if done:
                break
        scores.append(env.score)
    return scores


# =============================================================================
# Experiment 1: Full Transfer Matrix
# =============================================================================

def exp1_transfer_matrix(train_eps=50, test_eps=50, verbose=False):
    """Test all game->game transfer pairs."""
    games = ["snake", "maze", "collector", "balancer", "catcher"]
    results = {}

    for source in games:
        for target in games:
            if source == target:
                continue
            key = f"{source}->{target}"
            if verbose:
                print(f"  [{key}] training...", end="", flush=True)

            policy, train_score = run_training(source, train_eps)
            transfer_scores = run_testing(target, policy, test_eps)
            random_scores = run_random_baseline(target, test_eps)

            t_avg = sum(transfer_scores) / len(transfer_scores)
            r_avg = sum(random_scores) / len(random_scores)
            improvement = ((t_avg - r_avg) / max(r_avg, 0.01)) * 100

            results[key] = {
                "transfer_avg": round(t_avg, 2),
                "random_avg": round(r_avg, 2),
                "improvement_pct": round(improvement, 1),
                "policy_size": policy.size(),
            }
            if verbose:
                sign = "+" if improvement > 0 else ""
                print(f" transfer={t_avg:.2f} random={r_avg:.2f} ({sign}{improvement:.1f}%)")

    return results


# =============================================================================
# Experiment 2: Statistical Significance
# =============================================================================

def welch_ttest(a, b):
    """Welch's t-test for unequal variance. Returns (t_stat, p_value, cohens_d)."""
    import math
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / max(n1 - 1, 1)
    v2 = sum((x - m2) ** 2 for x in b) / max(n2 - 1, 1)
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 0.001

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / max(n1 - 1, 1) + (v2 / n2) ** 2 / max(n2 - 1, 1)
    df = num / max(denom, 0.001)

    # Approximate p-value using normal distribution for large df
    z = abs(t_stat)
    p_value = 2.0 * (1.0 - _normal_cdf(z))

    # Cohen's d
    pooled_std = math.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else 0.001
    cohens_d = (m1 - m2) / pooled_std

    return round(t_stat, 3), round(p_value, 4), round(cohens_d, 3)


def _normal_cdf(x):
    """Approximate standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def exp2_significance(pairs=None, runs=10, train_eps=50, test_eps=50, verbose=False):
    """Multi-run statistical significance test."""
    if pairs is None:
        pairs = [("balancer", "catcher"), ("snake", "maze"), ("catcher", "balancer")]

    results = {}
    for source, target in pairs:
        key = f"{source}->{target}"
        if verbose:
            print(f"  [{key}] {runs} runs...", end="", flush=True)

        all_transfer = []
        all_random = []
        for run in range(runs):
            seed_base = run * 1000
            policy, _ = run_training(source, train_eps, seed=seed_base)
            t_scores = run_testing(target, policy, test_eps, seed=seed_base + 500)
            r_scores = run_random_baseline(target, test_eps, seed=seed_base + 500)
            all_transfer.extend(t_scores)
            all_random.extend(r_scores)

        t_stat, p_value, cohens_d = welch_ttest(all_transfer, all_random)
        t_mean = sum(all_transfer) / len(all_transfer)
        r_mean = sum(all_random) / len(all_random)

        results[key] = {
            "transfer_mean": round(t_mean, 3),
            "random_mean": round(r_mean, 3),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "total_samples": len(all_transfer),
        }
        if verbose:
            sig = "SIGNIFICANT" if p_value < 0.05 else "not significant"
            print(f" p={p_value:.4f} d={cohens_d:.3f} ({sig})")

    return results


# =============================================================================
# Experiment 3: Learning Curve
# =============================================================================

def exp3_scaling(source="balancer", target="catcher",
                 steps=None, test_eps=50, verbose=False):
    """How does transfer improve with more training?"""
    if steps is None:
        steps = [10, 25, 50, 100, 200, 500]

    results = {}
    for n in steps:
        policy, train_score = run_training(source, n)
        transfer_scores = run_testing(target, policy, test_eps)
        random_scores = run_random_baseline(target, test_eps)

        t_avg = sum(transfer_scores) / len(transfer_scores)
        r_avg = sum(random_scores) / len(random_scores)

        results[n] = {
            "transfer_avg": round(t_avg, 2),
            "random_avg": round(r_avg, 2),
            "policy_size": policy.size(),
            "improvement_pct": round(((t_avg - r_avg) / max(r_avg, 0.01)) * 100, 1),
        }
        if verbose:
            print(f"  [{n} episodes] transfer={t_avg:.2f} random={r_avg:.2f} "
                  f"policy={policy.size()} entries")

    return results


# =============================================================================
# Experiment 4: Ablation Studies
# =============================================================================

def exp4_ablation(source="balancer", target="catcher",
                  train_eps=100, test_eps=50, verbose=False):
    """Test impact of removing predicates and varying bin resolution."""
    results = {"predicate_ablation": {}, "bin_resolution": {}}

    # 4a: Predicate ablation
    ablation_configs = {
        "full": None,  # all predicates
        "no_position": {"velocity", "danger"},
        "no_velocity": {"position", "danger"},
        "no_danger": {"position", "velocity"},
        "position_only": {"position"},
        "none": set(),  # no predicates
    }

    for name, preds in ablation_configs.items():
        hasher = SymbolicHasher(predicates=preds, bins=5)
        policy, _ = run_training(source, train_eps, hasher=hasher)
        transfer_scores = run_testing(target, policy, test_eps, hasher=hasher)
        random_scores = run_random_baseline(target, test_eps)

        t_avg = sum(transfer_scores) / len(transfer_scores)
        r_avg = sum(random_scores) / len(random_scores)

        results["predicate_ablation"][name] = {
            "transfer_avg": round(t_avg, 2),
            "random_avg": round(r_avg, 2),
            "policy_size": policy.size(),
        }
        if verbose:
            print(f"  [ablation/{name}] transfer={t_avg:.2f} random={r_avg:.2f} "
                  f"policy={policy.size()}")

    # 4b: Bin resolution
    for bins in [2, 5, 10, 20]:
        hasher = SymbolicHasher(bins=bins)
        policy, _ = run_training(source, train_eps, hasher=hasher)
        transfer_scores = run_testing(target, policy, test_eps, hasher=hasher)

        t_avg = sum(transfer_scores) / len(transfer_scores)
        results["bin_resolution"][bins] = {
            "transfer_avg": round(t_avg, 2),
            "policy_size": policy.size(),
        }
        if verbose:
            print(f"  [bins={bins}] transfer={t_avg:.2f} policy={policy.size()}")

    return results


# =============================================================================
# Experiment 5: Negative Transfer Detection
# =============================================================================

def exp5_negative_transfer(train_eps=100, test_eps=50, verbose=False):
    """Detect harmful or null transfer."""
    results = {}

    # 5a: Adversarial — inverted catcher
    if verbose:
        print("  [adversarial] training on balancer, testing on inverted_catcher...")
    policy, _ = run_training("balancer", train_eps)
    transfer_scores = run_testing("inverted_catcher", policy, test_eps)
    random_scores = run_random_baseline("inverted_catcher", test_eps)
    t_avg = sum(transfer_scores) / len(transfer_scores)
    r_avg = sum(random_scores) / len(random_scores)
    results["adversarial"] = {
        "transfer_avg": round(t_avg, 2),
        "random_avg": round(r_avg, 2),
        "is_negative": t_avg < r_avg,
    }
    if verbose:
        verdict = "NEGATIVE transfer" if t_avg < r_avg else "no negative transfer"
        print(f"    transfer={t_avg:.2f} random={r_avg:.2f} -> {verdict}")

    # 5b: Null transfer — train on random noise
    if verbose:
        print("  [null] training with random actions...")
    null_policy = LearnedPolicy()
    env = BalancerEnv()
    for _ in range(train_eps):
        state = env.reset()
        for _ in range(500):
            action = random.choice(["TILT_LEFT", "TILT_RIGHT", "HOLD"])
            h = SymbolicHasher.default_hash("balancer", state)
            null_policy.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break

    null_scores = run_testing("catcher", null_policy, test_eps)
    random_scores = run_random_baseline("catcher", test_eps)
    n_avg = sum(null_scores) / len(null_scores)
    r_avg = sum(random_scores) / len(random_scores)
    results["null_transfer"] = {
        "null_avg": round(n_avg, 2),
        "random_avg": round(r_avg, 2),
        "is_null": abs(n_avg - r_avg) < 0.5,
    }
    if verbose:
        print(f"    null={n_avg:.2f} random={r_avg:.2f}")

    # 5c: Incompatible — Pong -> Maze
    if verbose:
        print("  [incompatible] pong -> maze...")
    policy, _ = run_training("pong", train_eps)
    transfer_scores = run_testing("maze", policy, test_eps)
    random_scores = run_random_baseline("maze", test_eps)
    t_avg = sum(transfer_scores) / len(transfer_scores)
    r_avg = sum(random_scores) / len(random_scores)
    results["incompatible"] = {
        "transfer_avg": round(t_avg, 2),
        "random_avg": round(r_avg, 2),
        "no_harm": t_avg >= r_avg * 0.8,  # should be at least 80% of random
    }
    if verbose:
        print(f"    transfer={t_avg:.2f} random={r_avg:.2f}")

    return results


# =============================================================================
# Experiment 6: Curriculum Learning
# =============================================================================

def exp6_curriculum(test_eps=50, verbose=False):
    """Test chained training across multiple games."""
    results = {}

    # Pipeline 1: Snake -> Maze -> Collector
    if verbose:
        print("  [curriculum] Snake -> Maze -> Collector...")
    policy = LearnedPolicy()

    # Phase 1: Snake
    env = ENVS["snake"]()
    teacher = TeacherAgent()
    for _ in range(50):
        state = env.reset()
        for _ in range(200):
            action = teacher.choose_action("snake", state)
            h = SymbolicHasher.default_hash("snake", state)
            policy.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break

    # Phase 2: Maze
    env = ENVS["maze"]()
    for _ in range(50):
        state = env.reset()
        for _ in range(200):
            action = teacher.choose_action("maze", state)
            h = SymbolicHasher.default_hash("maze", state)
            policy.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break

    # Test on Collector
    curr_scores = run_testing("collector", policy, test_eps)
    random_scores = run_random_baseline("collector", test_eps)

    # Single-source comparison: train only on Snake
    single_policy, _ = run_training("snake", 100)
    single_scores = run_testing("collector", single_policy, test_eps)

    c_avg = sum(curr_scores) / len(curr_scores)
    s_avg = sum(single_scores) / len(single_scores)
    r_avg = sum(random_scores) / len(random_scores)

    results["grid_curriculum"] = {
        "curriculum_avg": round(c_avg, 2),
        "single_source_avg": round(s_avg, 2),
        "random_avg": round(r_avg, 2),
        "curriculum_better": c_avg > s_avg,
    }
    if verbose:
        print(f"    curriculum={c_avg:.2f} single={s_avg:.2f} random={r_avg:.2f}")

    # Pipeline 2: Balancer -> Easy Catcher -> Hard Catcher
    if verbose:
        print("  [curriculum] Balancer -> Easy Catcher -> Hard Catcher...")
    policy2 = LearnedPolicy()
    hasher = SymbolicHasher()

    # Phase 1: Balancer
    env = ENVS["balancer"]()
    for _ in range(50):
        state = env.reset()
        for _ in range(500):
            action = teacher.choose_action("balancer", state)
            h = hasher.hash("balancer", state)
            policy2.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break

    # Phase 2: Easy Catcher
    env = ENVS["easy_catcher"]()
    for _ in range(50):
        state = env.reset()
        for _ in range(600):
            action = teacher.choose_action("catcher", state)
            h = hasher.hash("easy_catcher", state)
            policy2.train(h, action)
            state, reward, done = env.step(action)
            if done:
                break

    # Test on Hard Catcher
    curr_scores2 = run_testing("hard_catcher", policy2, test_eps, hasher=hasher)
    random_scores2 = run_random_baseline("hard_catcher", test_eps)

    # Single-source comparison
    single_policy2, _ = run_training("balancer", 100)
    single_scores2 = run_testing("hard_catcher", single_policy2, test_eps)

    c2_avg = sum(curr_scores2) / len(curr_scores2)
    s2_avg = sum(single_scores2) / len(single_scores2)
    r2_avg = sum(random_scores2) / len(random_scores2)

    results["physics_curriculum"] = {
        "curriculum_avg": round(c2_avg, 2),
        "single_source_avg": round(s2_avg, 2),
        "random_avg": round(r2_avg, 2),
        "curriculum_better": c2_avg > s2_avg,
    }
    if verbose:
        print(f"    curriculum={c2_avg:.2f} single={s2_avg:.2f} random={r2_avg:.2f}")

    return results


# =============================================================================
# Experiment 7: Continual Learning (Retention & Backward Transfer)
# =============================================================================

def exp7_continual(train_eps=100, test_eps=50, verbose=False):
    """
    Test retention and backward transfer.
    1. Train A (Balancer) -> Measure Perf A1
    2. Train B (Catcher) [continual] -> Measure Perf B
    3. Retest A -> Measure Perf A2
    
    If A2 >= A1, we have retention (and possibly backward transfer).
    If A2 << A1, we have catastrophic forgetting.
    """
    results = {}
    
    # Setup
    policy = LearnedPolicy()
    hasher = SymbolicHasher()
    
    # 1. Train on Balancer (A)
    if verbose:
        print("  [Phase 1] Training on Balancer...")
    run_training("balancer", train_eps, hasher=hasher, policy_override=policy)
    score_a1_list = run_testing("balancer", policy, test_eps, hasher=hasher)
    score_a1 = sum(score_a1_list) / len(score_a1_list)
    
    # 2. Train on Catcher (B) - continuing with SAME policy
    if verbose:
        print(f"  [Phase 2] Continuing training on Catcher (Policy size: {policy.size()})...")
    run_training("catcher", train_eps, hasher=hasher, policy_override=policy)
    score_b_list = run_testing("catcher", policy, test_eps, hasher=hasher)
    score_b = sum(score_b_list) / len(score_b_list)
    
    # 3. Retest Balancer (A)
    if verbose:
        print(f"  [Phase 3] Retesting Balancer (Policy size: {policy.size()})...")
    score_a2_list = run_testing("balancer", policy, test_eps, hasher=hasher)
    score_a2 = sum(score_a2_list) / len(score_a2_list)
    
    # Random baselines for context
    rand_a = sum(run_random_baseline("balancer", test_eps)) / test_eps
    
    # Calculate metrics
    retention_ratio = score_a2 / max(score_a1, 0.01)
    forgetting = max(0, score_a1 - score_a2)
    backward_transfer = max(0, score_a2 - score_a1)
    
    results = {
        "score_a_initial": round(score_a1, 2),
        "score_b": round(score_b, 2),
        "score_a_final": round(score_a2, 2),
        "random_a": round(rand_a, 2),
        "retention_pct": round(retention_ratio * 100, 1),
        "backward_transfer": round(backward_transfer, 2),
        "forgetting": round(forgetting, 2),
        "policy_size": policy.size()
    }
    
    if verbose:
        print(f"  Results: A1={score_a1:.2f} -> B={score_b:.2f} -> A2={score_a2:.2f}")
        if score_a2 >= score_a1:
            print(f"  SUCCESS: No forgetting! ({results['backward_transfer']:+.2f} improvement)")
        else:
            print(f"  WARNING: Forgetting detected ({results['forgetting']:.2f} drop)")
            
    return results

# =============================================================================
# Experiment 8: Language Generalization (Zero-Shot & Transfer)
# =============================================================================

def exp8_language_transfer(test_eps=50, verbose=False):
    """
    Test zero-shot instruction following and cross-domain transfer.
    1. Zero-Shot: Teach "If object left, move left" -> Test Balancer (no training).
    2. Transfer: Test same rule on Catcher (no training).
    """
    results = {}
    
    # Setup Language Teacher
    lang_teacher = NaturalLanguageTeacher()
    
    # Teach rules (simulating user input)
    instructions = [
        "If object is to the left then move left",
        "If object is to the right then move right",
        "If object is to the left then tilt left", # Synonym test
        "If object is to the right then tilt right"
    ]
    
    if verbose:
        print(f"  [Teaching] Instructions: {instructions}")
        
    for instr in instructions:
        lang_teacher.teach(instr)
        
    # Test on Balancer (Zero-Shot)
    if verbose:
        print("  [Zero-Shot] Testing learned rules on Balancer...")
        
    env = BalancerEnv()
    scores_bal = []
    for _ in range(test_eps):
        state = env.reset()
        for _ in range(600):
            # Agent uses language-derived rules ZERO-SHOT
            action = lang_teacher.choose_action("balancer", state)
            state, reward, done = env.step(action)
            if done:
                break
        scores_bal.append(env.score)
        
    avg_bal = sum(scores_bal) / len(scores_bal)
    rand_bal = sum(run_random_baseline("balancer", test_eps)) / test_eps
    
    # Test on Catcher (Transfer)
    if verbose:
        print("  [Transfer] Testing same rules on Catcher...")
        
    env = CatcherEnv()
    scores_cat = []
    for _ in range(test_eps):
        state = env.reset()
        for _ in range(600):
            action = lang_teacher.choose_action("catcher", state)
            state, reward, done = env.step(action)
            if done:
                break
        scores_cat.append(env.score)
        
    avg_cat = sum(scores_cat) / len(scores_cat)
    rand_cat = sum(run_random_baseline("catcher", test_eps)) / test_eps
    
    results = {
        "balancer_zero_shot": round(avg_bal, 2),
        "balancer_random": round(rand_bal, 2),
        "catcher_transfer": round(avg_cat, 2),
        "catcher_random": round(rand_cat, 2),
        "improvement_bal": round((avg_bal - rand_bal)/max(rand_bal, 0.01) * 100, 1),
        "improvement_cat": round((avg_cat - rand_cat)/max(rand_cat, 0.01) * 100, 1)
    }
    
    if verbose:
        print(f"  Balancer: {avg_bal:.2f} (Random {rand_bal:.2f}) -> {results['improvement_bal']}%")
        print(f"  Catcher:  {avg_cat:.2f} (Random {rand_cat:.2f}) -> {results['improvement_cat']}%")
        
    return results


# =============================================================================
# Experiment 9: Cognitive Transfer (Complex Constraints)
# =============================================================================

def exp9_cognitive_transfer(test_eps=20, verbose=False):
    """
    Test transfer of abstract concepts (AVOID/APPROACH) via language.
    Constraint: "Catch Green, Avoid Red".
    
    Domains:
    1. Cognitive Catcher: Avoid=Dodge, Catch=Intercept.
    2. Cognitive Balancer: Avoid=Dump, Catch=Balance.
    """
    results = {}
    
    # Universal Instruction Set
    instructions = [
        "If object is red then avoid",
        "If object is green then catch",  # 'catch' maps to APPROACH -> Balance
    ]
    
    if verbose:
        print(f"  [Teaching] Instructions: {instructions}")
        
    teacher = NaturalLanguageTeacher()
    for rule in instructions:
        teacher.teach(rule)

    # 1. Test on Cognitive Catcher
    if verbose: print("  [Zero-Shot] Testing on Cognitive Catcher...")
    runner = BenchmarkRunner(verbose=False)
    
    # We need a custom agent that uses the teacher
    class LanguageAgent:
        def choose_action(self, game_type, state):
            return teacher.choose_action("cognitive_catcher", state)

    # Run Catcher
    res_catch = runner.run_game("cognitive_catcher", LanguageAgent(), "lang_agent", test_eps)
    
    # Run Random Baseline for Catcher
    base_catch = run_random_baseline("cognitive_catcher", test_eps)
    rand_catch = sum(base_catch)/len(base_catch)
    
    # 2. Test on Cognitive Balancer
    if verbose: print("  [Transfer] Testing on Cognitive Balancer...")
    
    class BalancerLanguageAgent:
        def choose_action(self, game_type, state):
            return teacher.choose_action("cognitive_balancer", state)
            
    res_bal = runner.run_game("cognitive_balancer", BalancerLanguageAgent(), "lang_agent", test_eps)
    
    # Run Random Baseline for Balancer
    base_bal = run_random_baseline("cognitive_balancer", test_eps)
    rand_bal = sum(base_bal)/len(base_bal)
    
    # Results
    score_catch = res_catch.avg_score
    score_bal = res_bal.avg_score
    
    pct_catch = (score_catch - rand_catch) / abs(rand_catch) * 100 if rand_catch != 0 else 0
    pct_bal = (score_bal - rand_bal) / abs(rand_bal) * 100 if rand_bal != 0 else 0
    
    if verbose:
        print(f"  Cognitive Catcher: {score_catch:.2f} (Random {rand_catch:.2f}) -> {pct_catch:.1f}%")
        print(f"  Cognitive Balancer: {score_bal:.2f} (Random {rand_bal:.2f}) -> {pct_bal:.1f}%")
    
    return {
        "catcher_score": score_catch,
        "catcher_random": rand_catch,
        "catcher_pct": pct_catch,
        "balancer_score": score_bal,
        "balancer_random": rand_bal,
        "balancer_pct": pct_bal
    }


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_results: Dict, elapsed: float) -> str:
    """Generate markdown report from experiment results."""
    lines = [
        "# NSCK Transfer Learning Experiment Report",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total runtime: {elapsed:.1f}s\n",
    ]

    if "exp1" in all_results:
        lines.append("## Experiment 1: Transfer Matrix\n")
        lines.append("| Source -> Target | Transfer | Random | Improvement |")
        lines.append("|-----------------|----------|--------|-------------|")
        for key, val in sorted(all_results["exp1"].items()):
            sign = "+" if val["improvement_pct"] > 0 else ""
            lines.append(f"| {key} | {val['transfer_avg']} | "
                         f"{val['random_avg']} | {sign}{val['improvement_pct']}% |")

    if "exp2" in all_results:
        lines.append("\n## Experiment 2: Statistical Significance\n")
        lines.append("| Pair | Transfer | Random | p-value | Cohen's d | Significant? |")
        lines.append("|------|----------|--------|---------|-----------|-------------|")
        for key, val in all_results["exp2"].items():
            sig = "YES" if val["significant"] else "no"
            lines.append(f"| {key} | {val['transfer_mean']:.3f} | "
                         f"{val['random_mean']:.3f} | {val['p_value']:.4f} | "
                         f"{val['cohens_d']:.3f} | {sig} |")

    if "exp3" in all_results:
        lines.append("\n## Experiment 3: Learning Curve\n")
        lines.append("| Training Episodes | Transfer | Random | Improvement | Policy Size |")
        lines.append("|-------------------|----------|--------|-------------|-------------|")
        for n, val in sorted(all_results["exp3"].items(), key=lambda x: int(x[0])):
            sign = "+" if val["improvement_pct"] > 0 else ""
            lines.append(f"| {n} | {val['transfer_avg']} | {val['random_avg']} | "
                         f"{sign}{val['improvement_pct']}% | {val['policy_size']} |")

    if "exp4" in all_results:
        lines.append("\n## Experiment 4: Ablation Studies\n")
        lines.append("### 4a: Predicate Ablation\n")
        lines.append("| Config | Transfer | Random | Policy Size |")
        lines.append("|--------|----------|--------|-------------|")
        for name, val in all_results["exp4"]["predicate_ablation"].items():
            lines.append(f"| {name} | {val['transfer_avg']} | "
                         f"{val['random_avg']} | {val['policy_size']} |")

        lines.append("\n### 4b: Bin Resolution\n")
        lines.append("| Bins | Transfer | Policy Size |")
        lines.append("|------|----------|-------------|")
        for bins, val in sorted(all_results["exp4"]["bin_resolution"].items(),
                                key=lambda x: int(x[0])):
            lines.append(f"| {bins} | {val['transfer_avg']} | {val['policy_size']} |")

    if "exp5" in all_results:
        lines.append("\n## Experiment 5: Negative Transfer\n")
        e5 = all_results["exp5"]
        if "adversarial" in e5:
            v = e5["adversarial"]
            lines.append(f"- **Adversarial (inverted):** transfer={v['transfer_avg']}, "
                         f"random={v['random_avg']} -> "
                         f"{'NEGATIVE transfer detected' if v['is_negative'] else 'No negative transfer'}")
        if "null_transfer" in e5:
            v = e5["null_transfer"]
            lines.append(f"- **Null (random training):** null={v['null_avg']}, "
                         f"random={v['random_avg']} -> "
                         f"{'No signal (as expected)' if v['is_null'] else 'Unexpected signal'}")
        if "incompatible" in e5:
            v = e5["incompatible"]
            lines.append(f"- **Incompatible (pong->maze):** transfer={v['transfer_avg']}, "
                         f"random={v['random_avg']} -> "
                         f"{'No harm' if v['no_harm'] else 'HARMFUL transfer'}")

    if "exp6" in all_results:
        lines.append("\n## Experiment 6: Curriculum Learning\n")
        for name, val in all_results["exp6"].items():
            better = "YES" if val["curriculum_better"] else "no"
            lines.append(f"- **{name}:** curriculum={val['curriculum_avg']}, "
                         f"single={val['single_source_avg']}, "
                         f"random={val['random_avg']} -> "
                         f"Curriculum better? {better}")

    if "exp7" in all_results:
        lines.append("\n## Experiment 7: Continual Learning (Balancer -> Catcher -> Balancer)\n")
        v = all_results["exp7"]
        lines.append(f"- **Initial Perf (A1):** {v['score_a_initial']}")
        lines.append(f"- **Final Perf (A2):** {v['score_a_final']}")
        lines.append(f"- **Retention:** {v['retention_pct']}%")
        lines.append(f"- **Backward Transfer:** {v['backward_transfer']:+}")
        
    if "exp8" in all_results:
        lines.append("\n## Experiment 8: Language Generalization\n")
        v = all_results["exp8"]
        lines.append(f"- **Balancer (Zero-Shot):** {v['balancer_zero_shot']} "
                     f"(Rand {v['balancer_random']}) -> {v['improvement_bal']}%")
        lines.append(f"- **Catcher (Transfer):** {v['catcher_transfer']} "
                     f"(Rand {v['catcher_random']}) -> {v['improvement_cat']}%")
        
    if "exp9" in all_results:
        lines.append("\n## Experiment 9: Cognitive Transfer\n")
        v = all_results["exp9"]
        lines.append(f"- **Cognitive Catcher:** {v['catcher_score']:.2f} "
                     f"(Random {v['catcher_random']:.2f}) -> {v['catcher_pct']:.1f}%")
        lines.append(f"- **Cognitive Balancer:** {v['balancer_score']:.2f} "
                     f"(Random {v['balancer_random']:.2f}) -> {v['balancer_pct']:.1f}%")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NSCK Transfer Learning Experiments")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--full", action="store_true", help="Full experiment suite")
    parser.add_argument("--exp", type=str, default=None,
                        help="Comma-separated experiment numbers (1-8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown report path")
    parser.add_argument("--json", type=str, default=None,
                        help="Output JSON report path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Determine which experiments to run
    if args.exp:
        exp_set = set(int(x) for x in args.exp.split(","))
    elif args.quick:
        exp_set = {1, 3, 5, 7, 8}
    elif args.full:
        exp_set = {1, 2, 3, 4, 5, 6, 7, 8}
    else:
        exp_set = {1, 2, 3, 4, 5, 6, 7, 8}

    # Scale parameters
    if args.quick:
        train_eps, test_eps, runs = 20, 20, 3
        scaling_steps = [10, 25, 50]
    else:
        train_eps, test_eps, runs = 100, 100, 10
        scaling_steps = [10, 25, 50, 100, 200, 500]

    print("=" * 60)
    print("  NSCK TRANSFER LEARNING EXPERIMENTS")
    print("=" * 60)
    print(f"  Experiments: {sorted(exp_set)}")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print("=" * 60)

    t0 = time.time()
    all_results = {}

    if 1 in exp_set:
        print("\n>>> Experiment 1: Transfer Matrix")
        all_results["exp1"] = exp1_transfer_matrix(
            train_eps=train_eps, test_eps=test_eps, verbose=args.verbose)

    if 2 in exp_set:
        print("\n>>> Experiment 2: Statistical Significance")
        all_results["exp2"] = exp2_significance(
            runs=runs, train_eps=train_eps, test_eps=test_eps, verbose=args.verbose)

    if 3 in exp_set:
        print("\n>>> Experiment 3: Learning Curve")
        all_results["exp3"] = exp3_scaling(
            steps=scaling_steps, test_eps=test_eps, verbose=args.verbose)

    if 4 in exp_set:
        print("\n>>> Experiment 4: Ablation Studies")
        all_results["exp4"] = exp4_ablation(
            train_eps=train_eps, test_eps=test_eps, verbose=args.verbose)

    if 5 in exp_set:
        print("\n>>> Experiment 5: Negative Transfer")
        all_results["exp5"] = exp5_negative_transfer(
            train_eps=train_eps, test_eps=test_eps, verbose=args.verbose)

    if 6 in exp_set:
        print("\n>>> Experiment 6: Curriculum Learning")
        all_results["exp6"] = exp6_curriculum(
            test_eps=test_eps, verbose=args.verbose)
            
    if 7 in exp_set:
        print("\n>>> Experiment 7: Continual Learning")
        all_results["exp7"] = exp7_continual(
            train_eps=train_eps, test_eps=test_eps, verbose=args.verbose)
            
    if 8 in exp_set:
        print("\n>>> Experiment 8: Language Generalization")
        all_results["exp8"] = exp8_language_transfer(test_eps=test_eps, verbose=args.verbose)

    if 9 in exp_set:
        print("\n>>> Experiment 9: Cognitive Transfer")
        all_results["exp9"] = exp9_cognitive_transfer(test_eps=test_eps, verbose=args.verbose)

    elapsed = time.time() - t0

    # Generate reports
    report = generate_report(all_results, elapsed)
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "transfer_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[*] Markdown report: {os.path.abspath(output_path)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"JSON report: {os.path.abspath(args.json)}")

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENTS COMPLETE -- {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
