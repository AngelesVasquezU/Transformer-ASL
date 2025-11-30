#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "transformer.h"
#include "transformer/asl_dataset.h"
#include <iostream>

int main()
{
    torch::Device device(torch::kCUDA);

    std::string dataset_root = "./asl_alphabet_train"; // aquí extrajiste el dataset

    int img_size = 64;
    int patch_size = 8;
    int d_model = 64;
    int n_heads = 4;
    int mlp_dim = 128;
    int num_layers = 4;
    int num_classes = 5;
    int batch_size = 32;
    int epochs = 3;

    auto model = transformer::VisionTransformer(
        img_size, patch_size, d_model,
        n_heads, mlp_dim, num_layers, num_classes);
    model->to(device);

    // Dataset
    auto dataset = ASLDataset(dataset_root, img_size)
                       .map(torch::data::transforms::ShuffleDataset(/*buffer_size=*/5000))
                       .map(torch::data::transforms::Batch(batch_size))
                       .map(torch::data::transforms::Stack<>());

    auto dataloader = torch::data::make_data_loader(
        std::move(dataset), torch::data::DataLoaderOptions().workers(4));

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4));

    std::cout << "Entrenando ViT en lenguaje de señas (A–E)...\n";

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        int batch_idx = 0;
        for (auto &batch : *dataloader)
        {
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto logits =
