#include "track_manager.h"

TrackManager::TrackManager():
id(0),
init(false)
{
    tracks.reserve(100); // track up to 100 tracks
};

void TrackManager::initTracks(const std::vector<cv::Rect>& incoming_tracks)
{
    for (const cv::Rect& bbox: incoming_tracks)
    {
        Track track(bbox, id);
        id++;
        tracks.push_back(track);
    }
}

void TrackManager::updateTracks(const std::vector<cv::Rect>& incoming_tracks)
{
    if (!init)
    {
        initTracks(incoming_tracks);
        init = true;
        return;
    }
    for (auto& track: tracks)
    {
        track.predictState();
    }
    std::vector<int> association_vector;
    std::vector<std::vector<double>> cost_matrix = initCostMatrix(incoming_tracks);
    HungarianAlgorithm algorithm;
    double cost = algorithm.Solve(cost_matrix, association_vector);
    // Check if existing track was associated
    for (int i = 0; i < tracks.size(); i++)
    {
        if (association_vector[i] == -1 || cost_matrix[i][association_vector[i]] > 0.9)
        {
            // Increase death counter
            tracks[i].increaseDeathCounter();
        }
        else 
        {
            // Update track KF state
            tracks[i].update(incoming_tracks[association_vector[i]]);
        }
    }
    // Delete existing dead tracks
     for (auto it = tracks.begin(); it != tracks.end();) {
        if (it->getDeathCounter() > 3) 
        {
            it = tracks.erase(it);
        } else 
        {
            it++;
        }
    }
    // Initialize new tracks from incoming detections
    std::vector<cv::Rect> new_tracks;
    for (int j = 0; j < incoming_tracks.size(); j++){
        if (std::find(std::begin(association_vector), std::end(association_vector), j) == std::end(association_vector)) 
        {
            new_tracks.push_back(incoming_tracks[j]);
        }
    }
    initTracks(new_tracks);
}

std::vector<std::vector<double>> TrackManager::initCostMatrix(const std::vector<cv::Rect>& incoming)
{
    std::vector<std::vector<double>> cost_matrix(tracks.size(), std::vector<double>(incoming.size()));
    for (size_t i = 0; i < tracks.size(); i++)
    {
        for (size_t j = 0; j < incoming.size(); j++)
        {
            cost_matrix[i].at(j) = 1 - calculateIOU(tracks[i].stateToBbox(), incoming[j]);   
        }
    }
    return cost_matrix;
}

std::vector<Track> TrackManager::getTracks() const
{
    return tracks;
}
