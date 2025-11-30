#pragma once
#include <torch/torch.h>

namespace transformer
{

    struct PatchEmbeddingImpl : torch::nn::Module
    {
        int img_size_;
        int patch_size_;
        int d_model_;
        int num_patches_;

        torch::nn::Conv2d proj{nullptr};

        PatchEmbeddingImpl(int img_size, int patch_size, int d_model);
        torch::Tensor forward(const torch::Tensor &x);
    };

    TORCH_MODULE(PatchEmbedding);

} // namespace transformer
