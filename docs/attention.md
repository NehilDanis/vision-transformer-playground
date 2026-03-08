---
title: Transformers
---


# Attention

Before attention mechanism, the token embeddings were static, so the word "bank" would have the same embedding regardless of whether it referred to a riverbank or a financial institution.

Transformers, on the other hand, use attention to compute **context-aware embeddings**. By taking the dot product between queries and keys, attention measures the similarity between each token (or image patch, in the case of Vision Transformers). The resulting weights are then used to compute a weighted sum of the values for each token. As a result, each token’s representation incorporates information about its relationship to all other tokens in the sequence, capturing context and dependencies dynamically. For given query, key and values the attention can be calculated with the following formula.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

There are different types of attentions. Altough the differences are subtle, they have huge effect on types of problems that can be addressed with each type of attention.


## Self Attention

Calculates the similarities between a token in a sequence with itself and every other word. Transformers that uses self attention are called **Encoder only Transformer**, they generate context aware encodings.

### What are context aware encodings

When using a token in a neural network, we first need to create an embedding consisting of numbers. Theoretically we can just assign a single random weight to each token, however this would then prevent us from capturing negative and positive meanings of a word, or in case of plural version of the word. Hence there are neural networks created to capture the meaning of the word, and encode it in a corrsponding word embedding. So when we visualiza the word embeddings in the end on a graph(assume that we have two dimensions in the word embedding), we could see after the training the words that have similar meaning will appear close to each other. 

However word embedding stage does not consider the position of the word on a sentence, so we would get the same input and output even if we changed the word other, hence in the transformers we also add the positional information to the word embeddings.

Later, using self attention, we create context aware embeddings, due to self attention encoding the scaled similarities between each token and all other tokens in the sequence.

Transformers that use self-attention can then be used to group similar documents together or relate different sentences. After applying self-attention within each sequence, a pooling step (e.g., using the `[CLS]` token or mean pooling) produces a fixed-size representation for the entire sequence. These pooled context-aware embeddings can then be compared across different documents or sentences to measure similarity.

For example, ChatGPT is a decoder only transformer, and it is called a generitive model. Because it was specifically trained, to generate the text that comes after a prompt.

In summary the decoder only transformers create generitive inputs that can be later plugged to a neural network that generates new tokens.

## Masked Self Attention

On the other hand masked self attention, takes into account the similarities of a token in a squence with itself and every other token that comes before it.

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

where $M$ is a mask matrix with $-\infty$ for future positions and $0$ for allowed positions.

The parts with -infinity values will get 0 after softmax. The attention percentage to those positions will be 0.

The transformers that use masked self attention, are called decoder only transformers. They are used in tasks, such a predicting the next word in a sequence, or generating an answer to the promt. 

They are trained in a **self-supervised manner**, meaning that during training, each token can see only itself and the tokens before it, and tries to predict the next token. There are no manual labels, the data itself provides the labels (the next token in the sequence).

## Encoder Decoder Attention (Cross Attention)

The first ever transformer model included an encoder that uses self attention, and a decoder that uses masked self attention. The transformer would use the encoder to create embeddings for keys and values, and use the decoder to create queries. Once the queries, keys and values are calculated the encoder-decoder attention is calculated just like self attention, using every similarity. This first transformer was based on something called **seq2seq** or encoder-decoder model.

Transformers with encoder-decoder attention are used in Multi-Modal Models. For example the encoder is trained on images or sound, and later the contex aware embeddings from the encoder could be fed into text based decoder, in order to generate captions.

## Multi-Head Attention

Transformers usually uses multiple attention units in parallel. This is needed to capture different relationships between the tokens in a sequence, simultaneously.

