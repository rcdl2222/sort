#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "utils.h"
#include "track_manager.h"

int main (int argc, char** argv) {
    // Load images
    std::vector<cv::String> images;
    cv::String path("./MOT15/train/ADL-Rundle-6/img1/*.jpg");
    cv::glob(path, images);
    size_t total_frames = images.size(); 
    std::string label_path = "./MOT15/train/ADL-Rundle-6/det/det.txt";
    std::ifstream label_file(label_path);
    if (!label_file.is_open()) {
        std::cerr << "Could not open or find the label!" << std::endl;
        return -1;
    }
    BBoxList all_detections = getBBoxList(label_file);
    // Close label file
    label_file.close();
    TrackManager tracker;

    for (size_t i = 0; i < total_frames; i++) {
        cv::Mat img = cv::imread(images[i]);
        const auto &detections = all_detections[i];
        tracker.updateTracks(detections);
        for (const auto &t : tracker.getTracks()) {
            // Draw detections in red bounding box
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
