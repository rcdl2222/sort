#include <iostream>
#include <fstream>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <yaml-cpp/yaml.h>

#include "utils.h"
#include "track_manager.h"

int main (int argc, char** argv) {
    // Get executable root path
    std::filesystem::path executablePath = std::filesystem::canonical(std::filesystem::path(argv[0])).parent_path();
    // Specify the relative path to the config file
    const std::string relativeConfigPath = "../config/config.yaml";
    // Construct the full path to the config file
    std::filesystem::path configFilePath = executablePath / relativeConfigPath;
    // Parse config file
    YAML::Node config = YAML::LoadFile(configFilePath);
    // Dataset to be loaded 
    const std::string dataset_name = config["dataset"].as<std::string>();
    // Load images
    std::vector<cv::String> images;
    cv::String path(config["image_path"].as<std::string>() + "/" + dataset_name + "/img1/*.jpg");
    cv::glob(path, images);
    size_t total_frames = images.size(); 
    if (total_frames == 0)
    {
        std::cerr << "Could not find any images; make sure they are in .jpg format!" << std::endl;
        return -1;
    }
    // Load labels
    const std::string detection_path = config["detections_path"].as<std::string>();
    std::string label_path;
    if (detection_path == "")
    {
        const std::string relativeDetectionPath = "/../det/" + dataset_name + "/det.txt";
        label_path = executablePath.string() + relativeDetectionPath;
    }
    else 
    {
        label_path = detection_path;
    }
    std::ifstream label_file(label_path);
    if (!label_file.is_open()) {
        std::cerr << "Could not open or find the labels!" << std::endl;
        return -1;
    }
    // Process bounding boxes
    BBoxList all_detections = getBBoxList(label_file);
    // Close label file
    label_file.close();
    // Initialize tracker
    TrackManager tracker;
    // Read images
    for (size_t i = 0; i < total_frames; i++) {
        cv::Mat img = cv::imread(images[i]);
        const auto &detections = all_detections[i];
        tracker.updateTracks(detections);
        for (const auto &t : tracker.getTracks()) {
            // Draw detections
            const auto& bbox = t.stateToBbox();
            cv::rectangle(img, bbox, t.getColor(), 3);
            cv::putText(img, std::to_string(t.getID()), cv::Point(bbox.tl().x, bbox.tl().y - 10),
                                    cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar(255, 255, 255), 2);
        }
        cv::imshow("Original", img);
        cv::waitKey(0);
    }
    return 0;
}
