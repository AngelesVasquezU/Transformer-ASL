#include "transformer/encoder_block.h"

namespace transformer
{

    EncoderBlockImpl::EncoderBlockImpl(int d_model, int n_heads, int mlp_dim)
        : d_model_(d_model),
          n_heads_(n_heads),
          mlp_dim_(mlp_dim)
    {
        // MultiheadAttention SIN batch_first
        mha = register_module(
            "mha",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d_model_, n_heads_)));

        // LayerNorm sigue igual
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
        // x_in = [B, N, D]
        auto x = norm1->forward(x_in);

        // MHA antigua usa formato [N, B, D]
        auto q = x.transpose(0, 1); // [N, B, D]
        auto k = x.transpose(0, 1); // [N, B, D]
        auto v = x.transpose(0, 1); // [N, B, D]

        // attn_out = [N, B, D]
        auto attn_tuple = mha->forward(q, k, v);
        auto attn_out = std::get<0>(attn_tuple);

        // Volver a formato ViT: [B, N, D]
        attn_out = attn_out.transpose(0, 1);

        // Residual 1
        auto x_res1 = x_in + attn_out;

        auto x2 = norm2->forward(x_res1);
        auto mlp_out = mlp->forward(x2);

        // Residual 2
        auto x_res2 = x_res1 + mlp_out;

        return x_res2;
    }

} // namespace transformer
