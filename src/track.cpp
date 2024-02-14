#include <cstdlib>
#include <iostream>
#include "track.h"

Track::Track(const cv::Rect& bbox, const size_t id):
id_(id),
death_counter_(0),
color(std::rand() % 256, std::rand() % 256, std::rand() % 256)
{   
    kf.init(8, 4, 0);

    // Transition matrix (A)
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(0, 4) = 1;
    kf.transitionMatrix.at<float>(1, 5) = 1;
    kf.transitionMatrix.at<float>(2, 6) = 1;
    kf.transitionMatrix.at<float>(3, 7) = 1;

    // Initial state, x
    float x = bbox.x + static_cast<float>(bbox.width) / 2; 
    float y = bbox.y + static_cast<float>(bbox.height) / 2;
    float width = static_cast<float>(bbox.width);
    float height = static_cast<float>(bbox.height);

    state = (cv::Mat_<float>(8, 1) << x, y, width, height, 0.0, 0.0, 0.0, 0.0);

    kf.statePre.at<float>(0) = x;
    kf.statePre.at<float>(1) = y;
    kf.statePre.at<float>(2) = width;
    kf.statePre.at<float>(3) = height;

    kf.statePost.at<float>(0) = x;
    kf.statePost.at<float>(1) = y;
    kf.statePost.at<float>(2) = width;
    kf.statePost.at<float>(3) = height;

    // Q matrix
    kf.processNoiseCov = (cv::Mat_<float>(8, 8) <<
                            1,   0,   0,  0,  0,   0,  0, 0,
                            0,   1,   0,  0,  0,   0,  0, 0,
                            0,   0,   1,  0,  0,   0,  0, 0,
                            0,   0,   0,  1,  0,   0,  0, 0,  
                            0,   0,   0,  0,  0.01, 0,  0, 0, 
                            0,   0,   0,  0,  0,  0.01, 0, 0,
                            0,   0,   0,  0,  0,  0, 0.0001, 0,
                            0,   0,   0,  0,  0,  0, 0, 0.0001);

    // H matrix
    cv::setIdentity(kf.measurementMatrix);
    // R matrix
    cv::setIdentity(kf.measurementNoiseCov);
    kf.measurementNoiseCov.at<float>(2, 2) = 10;
    kf.measurementNoiseCov.at<float>(3, 3) = 10;
    // P matrix
    kf.errorCovPost = (cv::Mat_<float>(8, 8) <<
           10, 0, 0, 0, 0, 0, 0, 0,
            0, 10, 0, 0, 0, 0, 0, 0,
            0, 0, 10, 0, 0, 0, 0, 0,
            0, 0, 0, 10, 0, 0, 0, 0,
            0, 0, 0, 0, 10000, 0, 0, 0,
            0, 0, 0, 0, 0, 10000, 0, 0,
            0, 0, 0, 0, 0, 0, 10000, 0,
            0, 0, 0, 0, 0, 0, 0, 10000);
    // P' matrix
    kf.errorCovPre = cv::Mat::zeros(8, 8, CV_32F);
}

void Track::predictState()
{
    // Internally, cv::KalmanFilter updates statePre and errorCovPre
    // and returns statePre
    state = kf.predict();
}

void Track::update(const cv::Rect& rect)
{    
    // Initialize measurement matrix
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << rect.x + static_cast<float>(rect.width) / 2,
     rect.y + static_cast<float>(rect.height) / 2, 
     static_cast<float>(rect.width),
     static_cast<float>(rect.height));
     // cv::KalmanFilter returns statePost
    state = kf.correct(measurement);
    // reset death counter
    death_counter_ = 0;
}

void Track::increaseDeathCounter()
{
    death_counter_++;
}

cv::Rect Track::stateToBbox() const 
{
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    auto width = std::max(0, static_cast<int>(state.at<float>(2)));
    auto height = std::max(0, static_cast<int>(state.at<float>(3)));
    auto tl_x = static_cast<int>(state.at<float>(0) - width / 2.0);
    auto tl_y = static_cast<int>(state.at<float>(1) - height / 2.0);
    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
    return rect;
}

size_t Track::getID() const
{
    return id_;
}
cv::Scalar Track::getColor() const
{
    return color;
}

int Track::getDeathCounter() const
{
    return death_counter_;
}
