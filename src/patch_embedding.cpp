#include "transformer/patch_embedding.h"

namespace transformer
{

    PatchEmbeddingImpl::PatchEmbeddingImpl(int img_size, int patch_size, int d_model)
        : img_size_(img_size),
          patch_size_(patch_size),
          d_model_(d_model)
    {
        int num_patches_side = img_size_ / patch_size_;
        num_patches_ = num_patches_side * num_patches_side;

        proj = register_module(
            "proj",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, d_model_, patch_size_).stride(patch_size_)));
    }

    torch::Tensor PatchEmbeddingImpl::forward(const torch::Tensor &x)
    {
        // x: [B, 3, H, W]
        auto y = proj->forward(x); // [B, D, H', W']
        y = y.flatten(2);          // [B, D, N]
        y = y.transpose(1, 2);     // [B, N, D]
        return y;
    }

} // namespace transformer
