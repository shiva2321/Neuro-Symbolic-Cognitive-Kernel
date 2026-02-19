# NSCK Benchmark Report

**Date:** 2026-02-12 21:20:11
**Duration:** 0.2s

## Results Summary

| Game | Agent | Episodes | Avg Score | Max | Success % | Avg Steps | Avg Reward |
|------|-------|----------|-----------|-----|-----------|-----------|------------|
| balancer | teacher | 100 | 7.95 | 49 | 100.0% | 104 | -0.02 |
| catcher | teacher | 100 | 20.00 | 20 | 100.0% | 340 | 24.50 |
| balancer | random | 100 | 43.08 | 49 | 100.0% | 500 | 48.55 |
| catcher | random | 100 | 1.23 | 6 | 72.0% | 126 | -8.77 |
| balancer | teacher | 100 | 10.91 | 49 | 100.0% | 132 | 4.00 |
| catcher | transfer(balancer->catcher) | 100 | 1.59 | 6 | 83.0% | 132 | -8.41 |
| catcher | random | 100 | 1.10 | 4 | 69.0% | 124 | -8.90 |

## Per-Game Details

### Balancer

**teacher** (100 episodes)
- Score: avg=7.95, median=3.0, max=49, min=2
- Steps: avg=104
- Reward: avg=-0.015
- Success rate: 100.0%
- First success: episode 0
- Score quartiles: Q1=2, Q3=5

**random** (100 episodes)
- Score: avg=43.08, median=43.0, max=49, min=34
- Steps: avg=500
- Reward: avg=48.551
- Success rate: 100.0%
- First success: episode 0
- Score quartiles: Q1=40, Q3=46

**teacher** (100 episodes)
- Score: avg=10.91, median=3.0, max=49, min=2
- Steps: avg=132
- Reward: avg=3.997
- Success rate: 100.0%
- First success: episode 0
- Score quartiles: Q1=2, Q3=6

### Catcher

**teacher** (100 episodes)
- Score: avg=20.00, median=20.0, max=20, min=20
- Steps: avg=340
- Reward: avg=24.500
- Success rate: 100.0%
- First success: episode 0
- Score quartiles: Q1=20, Q3=20

**random** (100 episodes)
- Score: avg=1.23, median=1.0, max=6, min=0
- Steps: avg=126
- Reward: avg=-8.770
- Success rate: 72.0%
- First success: episode 0
- Score quartiles: Q1=0, Q3=2

**transfer(balancer->catcher)** (100 episodes)
- Score: avg=1.59, median=1.0, max=6, min=0
- Steps: avg=132
- Reward: avg=-8.410
- Success rate: 83.0%
- First success: episode 2
- Score quartiles: Q1=1, Q3=2

**random** (100 episodes)
- Score: avg=1.10, median=1.0, max=4, min=0
- Steps: avg=124
- Reward: avg=-8.900
- Success rate: 69.0%
- First success: episode 1
- Score quartiles: Q1=0, Q3=2

## Transfer Learning Test

**Source:** balancer (100 training episodes)
**Target:** catcher (100 test episodes)

| Metric | Transfer | Random Baseline | Δ |
|--------|----------|-----------------|---|
| Avg Score | 1.59 | 1.10 | +0.49 |
| Success % | 83.0% | 69.0% | +14.0% |
| Avg Reward | -8.410 | -8.900 | +0.490 |

> ✅ Transfer shows **0.49** score improvement over random baseline
