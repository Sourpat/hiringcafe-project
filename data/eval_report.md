# Offline Evaluation Report

Generated: 2026-02-09T07:37:43.930942+00:00


## Pass 1 (cold)

Cache hit rate: 0.25

| Query | Role hit rate | Remote precision | Mission precision | Diversity | t_embed (s) | t_vector (s) | t_rerank (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data science jobs | 1.00 | - | - | 0.80 | 0.001 | 0.080 | 0.007 |
| senior ML engineer | 1.00 | - | - | 0.90 | 0.000 | 0.065 | 0.010 |
| mlops engineer | 1.00 | - | - | 0.90 | 0.000 | 0.073 | 0.008 |
| product manager | 1.00 | - | - | 0.90 | 0.000 | 0.061 | 0.007 |
| business analyst | 1.00 | - | - | 0.90 | 0.000 | 0.052 | 0.010 |
| product owner remote | 1.00 | 1.00 | - | 0.90 | 0.000 | 0.048 | 0.009 |
| remote python developer | 1.00 | 1.00 | - | 1.00 | 0.000 | 0.049 | 0.008 |
| nonprofit climate impact jobs | 0.10 | - | 0.40 | 0.50 | 0.000 | 0.024 | 0.004 |
| mission-driven data scientist | 1.00 | - | 0.00 | 1.00 | 0.001 | 0.060 | 0.006 |
| electrician apprentice | 0.00 | - | - | 1.00 | 0.001 | 0.028 | 0.006 |
| registered nurse | 0.10 | - | - | 0.60 | 0.000 | 0.025 | 0.005 |
| warehouse associate | 0.00 | - | - | 0.70 | 0.000 | 0.027 | 0.005 |

### Aggregate Averages

- role_intent_hit_rate: 0.68
- remote_precision: 1.00
- mission_precision: 0.20
- diversity: 0.84
- t_embed_s: 0.000
- t_vector_s: 0.049
- t_rerank_s: 0.007

## Pass 2 (warm)

Cache hit rate: 1.00

| Query | Role hit rate | Remote precision | Mission precision | Diversity | t_embed (s) | t_vector (s) | t_rerank (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data science jobs | 1.00 | - | - | 0.80 | 0.000 | 0.063 | 1.534 |
| senior ML engineer | 1.00 | - | - | 0.90 | 0.000 | 0.064 | 0.007 |
| mlops engineer | 1.00 | - | - | 0.90 | 0.000 | 0.054 | 0.006 |
| product manager | 1.00 | - | - | 0.90 | 0.000 | 0.049 | 0.010 |
| business analyst | 1.00 | - | - | 0.90 | 0.000 | 0.056 | 0.015 |
| product owner remote | 1.00 | 1.00 | - | 0.90 | 0.000 | 0.050 | 0.009 |
| remote python developer | 1.00 | 1.00 | - | 1.00 | 0.000 | 0.057 | 0.013 |
| nonprofit climate impact jobs | 0.10 | - | 0.40 | 0.50 | 0.000 | 0.035 | 0.004 |
| mission-driven data scientist | 1.00 | - | 0.00 | 1.00 | 0.000 | 0.065 | 0.014 |
| electrician apprentice | 0.00 | - | - | 1.00 | 0.000 | 0.032 | 0.006 |
| registered nurse | 0.10 | - | - | 0.60 | 0.000 | 0.024 | 0.006 |
| warehouse associate | 0.00 | - | - | 0.70 | 0.000 | 0.029 | 0.004 |

### Aggregate Averages

- role_intent_hit_rate: 0.68
- remote_precision: 1.00
- mission_precision: 0.20
- diversity: 0.84
- t_embed_s: 0.000
- t_vector_s: 0.048
- t_rerank_s: 0.136
