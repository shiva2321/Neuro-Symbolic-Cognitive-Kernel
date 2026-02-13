# NSCK Transfer Learning Experiment Report

Generated: 2026-02-13 03:21 UTC
Total runtime: 7.4s

## Experiment 1: Transfer Matrix

| Source -> Target | Transfer | Random | Improvement |
|-----------------|----------|--------|-------------|
| balancer->catcher | 1.38 | 0.96 | +43.7% |
| balancer->collector | 0.98 | 0.93 | +5.4% |
| balancer->maze | 0.14 | 0.31 | -54.8% |
| balancer->snake | 0.05 | 0.07 | -28.6% |
| catcher->balancer | 34.98 | 7.86 | +344.7% |
| catcher->collector | 1.18 | 1.2 | -1.7% |
| catcher->maze | 0.29 | 0.29 | 0.0% |
| catcher->snake | 0.03 | 0.02 | +50.0% |
| collector->balancer | 6.86 | 7.58 | -9.5% |
| collector->catcher | 0.91 | 0.89 | +2.2% |
| collector->maze | 0.29 | 0.29 | 0.0% |
| collector->snake | 0.04 | 0.03 | +33.3% |
| maze->balancer | 7.93 | 7.36 | +7.7% |
| maze->catcher | 0.75 | 0.86 | -12.8% |
| maze->collector | 1.0 | 1.02 | -2.0% |
| maze->snake | 0.05 | 0.0 | +500.0% |
| snake->balancer | 7.24 | 7.74 | -6.4% |
| snake->catcher | 0.97 | 0.88 | +10.2% |
| snake->collector | 0.95 | 1.16 | -18.1% |
| snake->maze | 0.33 | 0.27 | +22.2% |

## Experiment 2: Statistical Significance

| Pair | Transfer | Random | p-value | Cohen's d | Significant? |
|------|----------|--------|---------|-----------|-------------|
| balancer->catcher | 1.451 | 0.874 | 0.0000 | 0.488 | YES |
| snake->maze | 0.242 | 0.242 | 1.0000 | 0.000 | no |
| catcher->balancer | 35.382 | 7.395 | 0.0000 | 2.406 | YES |

## Experiment 3: Learning Curve

| Training Episodes | Transfer | Random | Improvement | Policy Size |
|-------------------|----------|--------|-------------|-------------|
| 10 | 1.57 | 1.04 | +51.0% | 14 |
| 25 | 1.23 | 0.81 | +51.9% | 14 |
| 50 | 1.22 | 0.96 | +27.1% | 15 |
| 100 | 1.33 | 0.79 | +68.4% | 15 |
| 200 | 1.42 | 0.79 | +79.7% | 15 |
| 500 | 1.47 | 0.99 | +48.5% | 15 |

## Experiment 4: Ablation Studies

### 4a: Predicate Ablation

| Config | Transfer | Random | Policy Size |
|--------|----------|--------|-------------|
| full | 1.37 | 0.83 | 15 |
| no_position | 1.59 | 0.74 | 5 |
| no_velocity | 1.0 | 0.92 | 11 |
| no_danger | 1.46 | 0.84 | 15 |
| position_only | 0.98 | 0.81 | 11 |
| none | 1.09 | 0.9 | 1 |

### 4b: Bin Resolution

| Bins | Transfer | Policy Size |
|------|----------|-------------|
| 2 | 1.87 | 9 |
| 5 | 1.35 | 15 |
| 10 | 1.31 | 27 |
| 20 | 1.29 | 47 |

## Experiment 5: Negative Transfer

- **Adversarial (inverted):** transfer=0.83, random=0.95 -> NEGATIVE transfer detected
- **Null (random training):** null=1.11, random=0.98 -> No signal (as expected)
- **Incompatible (pong->maze):** transfer=0.27, random=0.25 -> No harm

## Experiment 6: Curriculum Learning

- **grid_curriculum:** curriculum=1.06, single=1.22, random=1.03 -> Curriculum better? no
- **physics_curriculum:** curriculum=1.29, single=1.03, random=1.19 -> Curriculum better? YES

## Experiment 7: Continual Learning (Balancer -> Catcher -> Balancer)

- **Initial Perf (A1):** 21.88
- **Final Perf (A2):** 36.98
- **Retention:** 169.0%
- **Backward Transfer:** +15.1

## Experiment 8: Language Generalization

- **Balancer (Zero-Shot):** 10.13 (Rand 7.89) -> 28.3%
- **Catcher (Transfer):** 20.0 (Rand 0.94) -> 2027.7%
