#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

struct ASLDataset : torch::data::Dataset<ASLDataset>
{
    std::vector<std::string> image_paths;
    std::vector<int> labels;
    int img_size;

    ASLDataset(const std::string &root, int img_size_) : img_size(img_size_)
    {

        // SOLO 5 CLASES
        std::vector<std::string> classes = {"A", "E", "I", "O", "U"};

        for (int i = 0; i < classes.size(); i++)
        {
            std::string class_dir = root + "/" + classes[i];
            for (const auto &entry : fs::directory_iterator(class_dir))
            {
                image_paths.push_back(entry.path().string());
                labels.push_back(i);
            }
        }
    }

    // Convierte imagen → tensor
    torch::Tensor load_image(const std::string &path) const
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(img_size, img_size));

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);

        // Mat→Tensor [H, W, 3]
        auto tensor = torch::from_blob(
                          img.data, {img_size, img_size, 3},
                          torch::kFloat32)
                          .clone();

        // [3, H, W]
        tensor = tensor.permute({2, 0, 1});
        return tensor;
    }

    torch::data::Example<> get(size_t index) override
    {
        return {load_image(image_paths[index]),
                torch::tensor(labels[index], torch::kLong)};
    }

    torch::optional<size_t> size() const override
    {
        return image_paths.size();
    }
};
