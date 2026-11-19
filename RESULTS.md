# Results

Evaluation with `evaluate.py` (goal 25 steps ahead, at most 50 actions, horizon 5, replanning every 5 actions,
FP32 model, RTX 4090), evaluation seeds 0/1/42 x 100 episodes. Each entry: mean +- SD over training seeds 3072/3073/3074
(per-seed values in brackets). Macro = mean of the four tasks, computed per training seed.

| Model | Training | Planner | PushT | Cube | Reacher | TwoRoom | Macro |
|---|---|---|---|---|---|---|---|
| INTACT | multi-task | Direct | 85.67 +- 0.88 [85 / 85.3 / 86.7] | 100.00 +- 0.00 [100 / 100 / 100] | 93.44 +- 1.71 [93 / 95.3 / 92] | 100.00 +- 0.00 [100 / 100 / 100] | 94.78 +- 0.35 |
| INTACT | multi-task | Guarded A | 88.11 +- 0.77 [87.7 / 87.7 / 89] | 99.56 +- 0.38 [100 / 99.3 / 99.3] | 93.11 +- 0.69 [92.3 / 93.3 / 93.7] | 99.78 +- 0.38 [99.3 / 100 / 100] | 95.14 +- 0.34 |
| INTACT | single-task | Direct | 85.11 +- 0.38 [85.3 / 84.7 / 85.3] | 100.00 +- 0.00 [100 / 100 / 100] | 94.89 +- 2.04 [96.7 / 95.3 / 92.7] | 99.67 +- 0.58 [100 / 99 / 100] | 94.92 +- 0.52 |
| INTACT | single-task | Guarded A | 89.78 +- 0.19 [90 / 89.7 / 89.7] | 99.56 +- 0.19 [99.7 / 99.7 / 99.3] | 93.67 +- 3.18 [96.7 / 94 / 90.3] | 99.33 +- 0.88 [99.7 / 98.3 / 100] | 95.58 +- 0.85 |
| LeWM | multi-task | Pure CEM 300x30 | 64.67 +- 18.48 [43.3 / 75.3 / 75.3] | 59.56 +- 1.26 [58.7 / 61 / 59] | 46.67 +- 26.12 [34.3 / 29 / 76.7] | 94.44 +- 0.96 [95 / 95 / 93.3] | 66.33 +- 9.19 |
| LeWM | single-task | Pure CEM 300x30 | 83.11 +- 1.58 [84.3 / 81.3 / 83.7] | 58.89 +- 1.39 [60 / 57.3 / 59.3] | 81.00 +- 4.26 [85.7 / 80 / 77.3] | 91.33 +- 1.45 [93 / 90.3 / 90.7] | 78.58 +- 1.89 |
