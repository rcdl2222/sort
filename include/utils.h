#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>

using BBoxList = std::vector<std::vector<cv::Rect>>;

inline BBoxList getBBoxList(std::ifstream& label_file)
{
    // Process labels - group bounding boxes by frame index
    BBoxList bbox;
    std::vector<cv::Rect> bbox_per_frame;
    // Label index starts from 1
    int current_frame_index = 1;
    std::string line;
    while (std::getline(label_file, line)) {
        std::stringstream ss(line);
        // Label format <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        std::vector<float> label;
        std::string data;
        while (std::getline(ss , data, ',')) 
        {
            label.push_back(std::stof(data));
        }
        if (static_cast<int>(label[0]) != current_frame_index) 
        {
            current_frame_index = static_cast<int>(label[0]);
            bbox.push_back(bbox_per_frame);
            bbox_per_frame.clear();
        }
        bbox_per_frame.emplace_back(label[2], label[3], label[4], label[5]);
    }
    // Add bounding boxes from last frame
    bbox.push_back(bbox_per_frame);
    return bbox;
}

inline float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2)
{
    auto xx1 = std::max(rect1.tl().x, rect2.tl().x);
    auto yy1 = std::max(rect1.tl().y, rect2.tl().y);
    auto xx2 = std::min(rect1.br().x, rect2.br().x);
    auto yy2 = std::min(rect1.br().y, rect2.br().y);
    auto w = std::max(0, xx2 - xx1);
    auto h = std::max(0, yy2 - yy1);

    // calculate area of intersection and union
    float det_area = rect1.area();
    float rect2_area = rect2.area();
    auto intersection_area = w * h;
    float union_area = det_area + rect2_area - intersection_area;
    auto iou = intersection_area / union_area;
    return iou > 1 ? 1.0 : iou;
}