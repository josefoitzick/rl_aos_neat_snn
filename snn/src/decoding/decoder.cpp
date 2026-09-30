#include "../../include/decoding/decoder.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

RLDecoder::RLDecoder(DecodingType type, double window_duration, int vote_windows, int neurons_per_group)
    : type(type), window_duration(window_duration), vote_windows(vote_windows), neurons_per_group(neurons_per_group) {
}

int RLDecoder::decodeDiscreteAction(
    const std::vector<std::vector<double>>& output_spikes) const {

    switch (type) {
        case DecodingType::SPIKE_COUNT: {
            size_t max_spikes = 0;
            int best_action = 0;
            for (size_t i = 0; i < output_spikes.size(); i++) {
                if (output_spikes[i].size() > max_spikes) {
                    max_spikes = output_spikes[i].size();
                    best_action = static_cast<int>(i);
                }
            }
            return best_action;
        }

        case DecodingType::FIRST_SPIKE: {
            double earliest_spike = std::numeric_limits<double>::max();
            int best_action = 0;
            for (size_t i = 0; i < output_spikes.size(); i++) {
                if (!output_spikes[i].empty() && output_spikes[i][0] < earliest_spike) {
                    earliest_spike = output_spikes[i][0];
                    best_action = static_cast<int>(i);
                }
            }
            return best_action;
        }

        case DecodingType::VOTING: {
            std::vector<int> votes(output_spikes.size(), 0);
            double sub_window = window_duration / vote_windows;
            for (int w = 0; w < vote_windows; w++) {
                double t_start = w * sub_window;
                double t_end   = t_start + sub_window;
                int winner = -1;
                int max_count = 0;
                for (size_t i = 0; i < output_spikes.size(); i++) {
                    int count = 0;
                    for (double t : output_spikes[i]) {
                        if (t >= t_start && t < t_end) count++;
                    }
                    if (count > max_count) { max_count = count; winner = static_cast<int>(i); }
                }
                if (winner >= 0) votes[winner]++;
            }
            return static_cast<int>(
                std::max_element(votes.begin(), votes.end()) - votes.begin());
        }

        case DecodingType::GROUP_SPIKE_COUNT:
            return decodeGroupedAction(output_spikes, neurons_per_group);

        case DecodingType::WINNER_TAKES_ALL:
        default: {
            double max_rate_val = 0.0;
            int best_action = 0;
            for (size_t i = 0; i < output_spikes.size(); i++) {
                double rate = output_spikes[i].size() / window_duration * 1000.0;
                if (rate > max_rate_val) { max_rate_val = rate; best_action = static_cast<int>(i); }
            }
            return best_action;
        }
    }
}

// Suma spikes de grupos de neuronas; gana el grupo con más spikes totales
// Equivale al decoder de cartpole3: left_spikes vs right_spikes sobre 20 neuronas cada uno
int RLDecoder::decodeGroupedAction(
    const std::vector<std::vector<double>>& output_spikes,
    int npg) const {

    if (npg <= 0 || output_spikes.empty()) return 0;

    int num_groups = static_cast<int>(output_spikes.size()) / npg;
    if (num_groups == 0) num_groups = 1;

    int best_group = 0;
    int max_total = -1;

    for (int g = 0; g < num_groups; g++) {
        int total = 0;
        int start = g * npg;
        int end   = std::min(start + npg, static_cast<int>(output_spikes.size()));
        for (int i = start; i < end; i++) {
            total += static_cast<int>(output_spikes[i].size());
        }
        if (total > max_total) {
            max_total  = total;
            best_group = g;
        }
    }

    return best_group;
}

double RLDecoder::decodeContinuousAction(const std::vector<double>& spike_times) const {
    if (spike_times.empty()) {
        return 0.0;
    }

    switch (type) {
        // TODO: Revisar SPIKE_COUNT y determinar la diferencia con RATE
        case DecodingType::SPIKE_COUNT:
            return std::min(1.0, spike_times.size() / 10.0) * 2.0 - 1.0;

        case DecodingType::FIRST_SPIKE: {
            double latency = spike_times[0] / window_duration;
            return 1.0 - latency;
        }

        case DecodingType::RATE: {
            double rate = spike_times.size() / window_duration * 1000.0;
            return std::min(1.0, rate / 100.0);
        }

        default:
            return 0.0;
    }
}

double RLDecoder::decodeContinuousAction(const std::vector<double>& spike_times, double max_spikes) const {
    if (spike_times.empty()) return -1.0;

    const double n = static_cast<double>(spike_times.size());
    switch (type) {
        case DecodingType::FIRST_SPIKE: {
            double latency = spike_times[0] / window_duration;
            return std::clamp(1.0 - 2.0 * latency, -1.0, 1.0);
        }
        case DecodingType::SPIKE_COUNT:
            return std::clamp(std::min(1.0, n / max_spikes) * 2.0 - 1.0, -1.0, 1.0);
        default:
            return std::clamp(n / max_spikes * 2.0 - 1.0, -1.0, 1.0);
    }
}

std::vector<double> RLDecoder::decodeContinuousActions(
    const std::vector<std::vector<double>>& output_spikes) const {

    std::vector<double> actions;
    actions.reserve(output_spikes.size());
    for (const auto& spikes : output_spikes) {
        actions.push_back(decodeContinuousAction(spikes));
    }
    return actions;
}
