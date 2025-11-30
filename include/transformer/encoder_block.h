#pragma once
#include <torch/torch.h>

namespace transformer
{

    struct EncoderBlockImpl : torch::nn::Module
    {
        int d_model_;
        int n_heads_;
        int mlp_dim_;

        torch::nn::MultiheadAttention mha{nullptr};
        torch::nn::LayerNorm norm1{nullptr};
        torch::nn::LayerNorm norm2{nullptr};
        torch::nn::Sequential mlp;

        EncoderBlockImpl(int d_model, int n_heads, int mlp_dim);

        // x: [B, N, D]
        torch::Tensor forward(const torch::Tensor &x);
    };

    TORCH_MODULE(EncoderBlock);

} // namespace transformer
