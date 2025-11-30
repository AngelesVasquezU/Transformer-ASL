#include "transformer/encoder_block.h"

namespace transformer
{

    EncoderBlockImpl::EncoderBlockImpl(int d_model, int n_heads, int mlp_dim)
        : d_model_(d_model),
          n_heads_(n_heads),
          mlp_dim_(mlp_dim)
    {
        mha = register_module(
            "mha",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d_model_, n_heads_).batch_first(true)));

        norm1 = register_module(
            "norm1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model_})));

        norm2 = register_module(
            "norm2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model_})));

        mlp = register_module(
            "mlp",
            torch::nn::Sequential(
                torch::nn::Linear(d_model_, mlp_dim),
                torch::nn::GELU(),
                torch::nn::Linear(mlp_dim, d_model_)));
    }

    torch::Tensor EncoderBlockImpl::forward(const torch::Tensor &x_in)
    {
        // x_in: [B, N, D]
        // Self-attention
        auto x = norm1->forward(x_in);
        auto attn_out_tuple = mha->forward(x, x, x);
        auto attn_out = std::get<0>(attn_out_tuple); // [B, N, D]
        auto x_res1 = x_in + attn_out;

        auto x2 = norm2->forward(x_res1);
        auto mlp_out = mlp->forward(x2);
        auto x_res2 = x_res1 + mlp_out;

        return x_res2;
    }

} // namespace transformer
