# Character-Level Language Model

This repository contains a PyTorch implementation of a character-level language model. The model is based on a Transformer architecture, featuring multi-head self-attention mechanisms.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Acknowledgements](#acknowledgements)

## Introduction

Character-level language models generate text one character at a time. These models are useful for generating text with fine-grained control over the output, such as programming code or structured documents.

## Model Architecture

The model consists of several key components:

- **Embedding Layers**: `nn.Embedding` layers to convert input tokens and their positions into embeddings.
- **Transformer Blocks**: Stacked blocks containing multi-head self-attention and feed-forward neural networks.
- **Layer Normalization**: Applied before each sub-layer in a Transformer block.
- **Linear Layer**: Projects the output embeddings to the vocabulary size for generating logits.

The model parameters can be adjusted via hyperparameters defined at the beginning of the script.

# Acknowledgements

This implementation is inspired by Andrej Karpathy's ["GPT from Scratch"](https://github.com/karpathy/ng-video-lecture). His clear and insightful explanations on building generative pre-trained transformers from scratch provided a strong foundation for this project.

Additionally, this work builds upon the Transformer architecture introduced in the seminal paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" by Vaswani et al. The concepts of self-attention and transformer blocks have been fundamental to the development of this model.
