# Positional Embeddings in PyTorch

## Nomenclature

Nobody likes it, but obviously this same things have many slightly different names.

It consists of two words, the first word can be "position" or "positional", and the 
second "embedding" or "encoding". In this pakcage, it is called **positional embedding**.

## In brief

Positional embedding is critical for a transformer to distinguish between permutations. 
However, the countless variants of positional embeddings make people dazzled. 
Positional embeddings can be awkward to understand and implement, sometimes taking the 
majority of space in your pytorch code.

The aim of this package is to build a collection of popular positional embedding modules 
and provide unified APIs. Ideally, positional embeddings could be isolated from the 
transformer architecture and be used in a plug-and-play manner, **i.e. outside
positional embedding modules, everything should be permutation invariant**.

Besides, this package is meant to provide plain, easy and naive implementation.

## APIs

After comparing several positional embeddings, I summarize the behavior of positional 
embedding into two APIs.

### `forward_input`

Positional embeddings can directly integrate positional information into input (e.g. 
word embedding). This API integrates positional information into the input.

```python
PositionalEmbedding.forward_input

"""Generate positional embedding to be added to the input.

Args:        

    input_: torch.Tensor: shape: [batch_size, max_length, embed_dim]
        The input tensor.

    positions: torch.Tensor, shape: [batch_size, max_length, input_]
        Absolute positions of input tokens.


Returns:
input_: torch.Tensor, shape: [batch_size, max_length, input_]
    A tensor with both input and positional information.
"""
```


### `forward_attn`

Some implementations (especially relative positional embeddings) directly generate 
attention matrix from positional embeddings and add to the qk attention matrix, i.e.
attention bias. Some implementations modify queries and keys before calculating 
attention matrix so that they possess positional information.

These facts represent the tight coupling between positional embedding and transformer 
and as a design choice, I decided to leave the responsibility of calculating attention 
matrix to positional embeddings.

```python
PositionalEmbedding.forward_attn

"""Generate attention logits from queries, keys and their positions.

Args:

    q: torch.Tensor, shape: [batch_size, num_heads, q_max_length, head_dim]
        The query tensor.

    k: torch.Tensor, shape: [batch_size, num_heads, k_max_length, head_dim]
        The key tensor.
    
    positions_q: torch.Tensor, shape: [batch_size, q_max_length]
        Absolute positions of query tokens.
    
    positions_k: torch.Tensor, shape: [batch_size, k_max_length]
        Absolute positions of key tokens.


Returns:
attn_logits: torch.Tensor, shape: [batch_size, q_max_length, k_max_length]
    Attention logits (before padding mask, before softmax, and before scaling)
"""
```

I know we generally regard calculating attention matrix (qk similarity) as a 
characteristic step of a transformer module and a lot architectural modifications happen 
here. However, to isolate positional embedding from transformer, I have to make this 
decision. This means the `O(n^2)` complexity now belongs to positional embedding instead 
of transformers, and architectural modifications, such as sparse attention, additive 
attention, now must be reflected in positional embedding modules.


## Basic usage

#To be added#

## Supported positional embeddings

Sinusoidal positional embedding (`SinusoidalPositionalEmbedding`) in "Attention is all you need".

Learnable positional embedding (`LearnedPositionalEmbedding`) in BERT and GPT.

Relative positional embedding (`TransformerXLPositionalEmbedding`) in Transformer-XL.

Relative positional embedding (`T5PositionalEmbedding`) in T5.

Unified positional embedding (`UnifiedPositionalEmbedding`) in TUPE.

Relative positional embedding (`EnformerPositionalEmbedding`) in Enformer.

## Installation

`pip install positional-embeddings-pytorch`

However, this package is highly experimental. It could serve as a reference for 
implementing and choosing positional embeddings, but I strongly discourage you directly 
throwing it into your code. Instead, users are expected to have prior knowledge about positional embeddings and check the code before using.

## Future work

- Positional embedding for decoder.
- Positional embedding with memory.
- Add support for \[CLS\] tokens.

(Current implementation only considers transformer encoder without memory, and does not 
support special tokens such as \[CLS\].)

## References

[pytorch/fairseq: Facebook AI Research Sequence-to-Sequence Toolkit written in Python. (github.com)](https://github.com/pytorch/fairseq)

[huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. (github.com)](https://github.com/huggingface/transformers)

[kimiyoung/transformer-xl (github.com)](https://github.com/kimiyoung/transformer-xl/)

[guolinke/TUPE: Transformer with Untied Positional Encoding (TUPE). Code of paper "Rethinking Positional Encoding in Language Pre-training". Improve existing models like BERT. (github.com)](https://github.com/guolinke/TUPE)

[T5 relative positional embedding (github.com)](https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16)

## Author(s)

Yiming Qu.

Undergraduate at Tsinghua University. Biology and Data Science.

Research Intern at Microsoft Research. Computational Biology.
