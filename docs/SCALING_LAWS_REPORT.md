# SCALING LAWS AUDIT: HUGE CORPUS STRESS TEST
Generated: 2026-02-13T14:37:38.151619

## 1. Capacity Metrics
| Metric | Value |
|---|---|
| Total Concepts | 50000 |
| Peak RAM usage | 882.02 MB |
| Precision (Exact) | 1.00 |
| Precision (10% Noise) | 1.0 |
| Precision (3-Bundle Member) | 0.92 |
| Cross-Modal Accuracy (T->I) | 1.00 |

## 2. The Saturation Floor (Information Floor)
Analyzes the RAM footprint and ingest speed as the semantic manifold grows.

| Concepts | RAM (MB) | Time (s) |
|---|---|---|
| 1000 | 649.7 | 0.0 |
| 2000 | 653.6 | 0.1 |
| 3000 | 658.2 | 0.2 |
| 4000 | 661.6 | 0.3 |
| 5000 | 669.2 | 0.4 |
| 6000 | 674.0 | 0.5 |
| 7000 | 677.6 | 0.6 |
| 8000 | 680.9 | 0.7 |
| 9000 | 686.2 | 0.8 |
| 10000 | 689.4 | 0.8 |
| 12000 | 830.6 | 7.7 |
| 14000 | 833.4 | 13.2 |
| 16000 | 835.4 | 18.5 |
| 18000 | 836.4 | 23.9 |
| 20000 | 839.4 | 29.2 |
| 22000 | 842.9 | 34.4 |
| 24000 | 845.4 | 39.8 |
| 26000 | 848.2 | 45.1 |
| 28000 | 851.0 | 50.3 |
| 30000 | 853.7 | 55.6 |
| 32000 | 856.2 | 60.9 |
| 34000 | 859.0 | 66.2 |
| 36000 | 861.8 | 71.6 |
| 38000 | 864.5 | 76.9 |
| 40000 | 867.3 | 82.2 |
| 42000 | 870.1 | 87.4 |
| 44000 | 874.2 | 92.8 |
| 46000 | 876.7 | 98.1 |
| 48000 | 879.5 | 103.9 |
| 50000 | 882.0 | 111.3 |