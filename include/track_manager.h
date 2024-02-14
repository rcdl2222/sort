#include <opencv2/core.hpp>
#include "track.h"
#include "utils.h"
#include "Hungarian.h"

class TrackManager {

    public:
        TrackManager();
        ~TrackManager() = default;
        void updateTracks(const std::vector<cv::Rect>& incoming_tracks);
        std::vector<Track> getTracks() const;

    private:
        size_t id;
        bool init;
        std::vector<Track> tracks;
        std::vector<std::vector<double>> initCostMatrix(const std::vector<cv::Rect>& incoming);
        void initTracks(const std::vector<cv::Rect>& incoming_tracks);

};