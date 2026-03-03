---
title: Attention
---

## Self Attention

Before attention mechanism, the token embeddings were static, so the word "bank" would have the same embedding regardless of whether it referred to a riverbank or a financial institution.

Transformers, on the other hand, use attention to compute **context-aware embeddings**. By taking the dot product between queries and keys, attention measures the similarity between each token (or image patch, in the case of Vision Transformers). The resulting weights are then used to compute a weighted sum of the values for each token. As a result, each token’s representation incorporates information about its relationship to all other tokens in the sequence, capturing context and dependencies dynamically.