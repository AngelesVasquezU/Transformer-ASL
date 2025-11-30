#include "transformer/vit.h"

namespace transformer
{

    VisionTransformerImpl::VisionTransformerImpl(int img_size,
                                                 int patch_size,
                                                 int d_model,
                                                 int n_heads,
                                                 int mlp_dim,
                                                 int num_layers,
                                                 int num_classes)
        : img_size_(img_size),
          patch_size_(patch_size),
          d_model_(d_model),
          n_heads_(n_heads),
          mlp_dim_(mlp_dim),
          num_layers_(num_layers),
          num_classes_(num_classes)
    {
        patch_embed = register_module(
            "patch_embed",
            PatchEmbedding(img_size_, patch_size_, d_model_));

        int num_patches_side = img_size_ / patch_size_;
        num_patches_ = num_patches_side * num_patches_side;

        // CLS token
        cls_token = register_parameter(
            "cls_token",
            torch::zeros({1, 1, d_model_}, torch::kFloat32));

        // Positional embedding (cls + patches)
        pos_embed = register_parameter(
            "pos_embed",
            torch::zeros({1, 1 + num_patches_, d_model_}, torch::kFloat32));

        for (int i = 0; i < num_layers_; ++i)
        {
            auto block = EncoderBlock(d_model_, n_heads_, mlp_dim_);
            blocks.push_back(register_module("block_" + std::to_string(i), block));
        }

        norm = register_module(
            "norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model_})));
        head = register_module(
            "head",
            torch::nn::Linear(d_model_, num_classes_));
    }

    torch::Tensor VisionTransformerImpl::forward(const torch::Tensor &x)
    {
        // x: [B, 3, H, W]
        auto B = x.size(0);

        auto x_patches = patch_embed->forward(x); // [B, N, D]

        auto cls_tok = cls_token.expand({B, 1, d_model_}); // [B, 1, D]
        auto x_cat = torch::cat({cls_tok, x_patches}, 1);  // [B, 1+N, D]

        auto pos = pos_embed;
        pos = pos.expand_as(x_cat); // [B, 1+N, D]
        auto x_pos = x_cat + pos;

        auto h = x_pos;
        for (auto &blk : blocks)
        {
            h = blk->forward(h);
        }

        h = norm->forward(h);                              // [B, 1+N, D]
        auto cls = h.index({torch::indexing::Slice(), 0}); // [B, D]
        auto logits = head->forward(cls);                  // [B, num_classes]

        return logits;
    }

} // namespace transformer
