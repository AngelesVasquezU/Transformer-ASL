#pragma once
#include <torch/torch.h>

namespace transformer
{

    inline torch::Tensor make_positional_encoding(int num_patches, int d_model)
    {
        // simple learnable pos embedding
        auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
        return torch::zeros({1, num_patches, d_model}, options);
    }

} // namespace transformer
