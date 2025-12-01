#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "transformer.h"
#include "transformer/asl_dataset.h"
#include <iostream>

int main()
{
    torch::Device device(torch::kCUDA);

    std::string dataset_root = "./asl_alphabet_train";

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
        n_heads, mlp_dim, num_layers, num_classes
    );
    model->to(device);

    // Dataset base (cada elemento: Example<tensor imagen, label>)
    auto dataset = ASLDataset(dataset_root, img_size)
        .map(torch::data::transforms::Stack<>());

    // DataLoader SIMPLE, sin shuffle
    torch::data::DataLoaderOptions options(batch_size);
    options.workers(4);

    auto dataloader = torch::data::make_data_loader(
        std::move(dataset),
        options
    );

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-4)
    );

    std::cout << "Entrenando ViT en lenguaje de señas (A–U)...\n";

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        int batch_idx = 0;
        long total_correct = 0;
        long total_samples = 0;

        for (auto& batch : *dataloader)
        {
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);

            auto logits = model->forward(images);
            auto loss = torch::nn::functional::cross_entropy(logits, labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // accuracy simple
            auto preds = logits.argmax(1);
            auto correct = (preds == labels).sum().item<int64_t>();
            total_correct += correct;
            total_samples += labels.size(0);

            if (batch_idx % 10 == 0)
            {
                std::cout << "Epoch " << epoch
                          << " Batch " << batch_idx
                          << " Loss: " << loss.item().toFloat()
                          << std::endl;
            }

            batch_idx++;
        }

        double acc = static_cast<double>(total_correct) / static_cast<double>(total_samples);
        std::cout << "===> Epoch " << epoch << " accuracy: " << acc << std::endl;
    }

    std::cout << "Entrenamiento terminado.\n";

    // ====== GUARDAR MODELO MANUALMENTE ======
    try {
        torch::serialize::OutputArchive archive;
        model->save(archive);                      // guarda weights al archive
        archive.save_to("vit_asl.pt");             // escribe archivo en disco
        std::cout << "Modelo guardado como vit_asl.pt\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error guardando el modelo: " << e.what() << std::endl;
    }

    return 0;
}