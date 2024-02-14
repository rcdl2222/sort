#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>

class Track {

    public:
        Track(const cv::Rect& bbox, const size_t id);
        ~Track() = default;
        void predictState();
        void update(const cv::Rect& rect);
        void increaseDeathCounter();
        cv::Rect stateToBbox() const;
        size_t getID() const;
        cv::Scalar getColor() const;
        int getDeathCounter() const;
    private:
        size_t id_;
        size_t death_counter_;
        cv::Scalar color;
        cv::KalmanFilter kf;
        cv::Mat state;
};
