#pragma once
#include <torch/torch.h>
#include <vector>
#include "transformer/patch_embedding.h"
#include "transformer/encoder_block.h"
#include "transformer/utils.h"

namespace transformer
{

    struct VisionTransformerImpl : torch::nn::Module
    {
        int img_size_;
        int patch_size_;
        int d_model_;
        int n_heads_;
        int mlp_dim_;
        int num_layers_;
        int num_classes_;
        int num_patches_;

        PatchEmbedding patch_embed{nullptr};
        std::vector<EncoderBlock> blocks;
        torch::nn::LayerNorm norm{nullptr};
        torch::nn::Linear head{nullptr};

        torch::Tensor cls_token; // [1, 1, D]
        torch::Tensor pos_embed; // [1, 1+num_patches, D]

        VisionTransformerImpl(int img_size,
                              int patch_size,
                              int d_model,
                              int n_heads,
                              int mlp_dim,
                              int num_layers,
                              int num_classes);

        // x: [B, 3, H, W]
        torch::Tensor forward(const torch::Tensor &x);
    };

    TORCH_MODULE(VisionTransformer);

} // namespace transformer
