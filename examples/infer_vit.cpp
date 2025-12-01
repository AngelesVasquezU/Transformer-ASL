#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "transformer.h"
#include <iostream>
#include <array>

torch::Tensor load_image_to_tensor(const std::string& path, int img_size) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("No se pudo leer la imagen: " + path);
    }

    cv::resize(img, img, cv::Size(img_size, img_size));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        img.data, {img_size, img_size, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();

    tensor = tensor.permute({2, 0, 1});   // [3, H, W]
    tensor = tensor.unsqueeze(0);         // [1, 3, H, W]
    return tensor;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./infer_vit ruta_imagen\n";
        return 1;
    }

    std::string img_path = argv[1];
    int img_size = 64;
    int patch_size = 8;
    int d_model = 64;
    int n_heads = 4;
    int mlp_dim = 128;
    int num_layers = 4;
    int num_classes = 5;

    torch::Device device(torch::kCUDA);

    auto model = transformer::VisionTransformer(
        img_size, patch_size, d_model,
        n_heads, mlp_dim, num_layers, num_classes
    );

    torch::load(model, "vit_asl.pt");
    model->to(device);
    model->eval();

    auto input = load_image_to_tensor(img_path, img_size).to(device);

    auto logits = model->forward(input);
    auto pred = logits.argmax(1).item<int64_t>();

    std::array<char, 5> classes = {'A', 'B', 'C', 'D', 'E'};
    if (pred >= 0 && pred < classes.size()) {
        std::cout << "Prediccion: " << classes[pred] << std::endl;
    } else {
        std::cout << "Prediccion fuera de rango: " << pred << std::endl;
    }

    return 0;
}