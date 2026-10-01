#include "adaptive_pursuit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace aos {

using neat::kNumOperators;

AdaptivePursuitSelector::AdaptivePursuitSelector(const AdaptivePursuitConfig& cfg)
    : cfg_(cfg),
      q_(kNumOperators, cfg.q_init),
      last_credit_(kNumOperators, std::numeric_limits<double>::quiet_NaN()) {
    if (cfg_.p_min < 0.0 || kNumOperators * cfg_.p_min >= 1.0)
        throw std::invalid_argument("AdaptivePursuit: se requiere 0 <= K·p_min < 1");
    if (cfg_.p_init.empty()) {
        p_.assign(kNumOperators, 1.0 / kNumOperators);
    } else {
        if (static_cast<int>(cfg_.p_init.size()) != kNumOperators)
            throw std::invalid_argument("AdaptivePursuit: p_init debe tener K elementos");
        p_ = cfg_.p_init;
        double sum = std::accumulate(p_.begin(), p_.end(), 0.0);
        for (double& p : p_) p = std::max(p / sum, cfg_.p_min);   // respetar la cota inferior
        sum = std::accumulate(p_.begin(), p_.end(), 0.0);
        for (double& p : p_) p /= sum;
    }
}

double AdaptivePursuitSelector::childReward(const neat::OperatorOutcome& o, double f_best, double f_worst) {
    const double range = f_best - f_worst;
    const double eps = 1e-12;
    return std::max(0.0, o.child_fitness - o.parent_fitness) / (range + eps);
}

void AdaptivePursuitSelector::pursue(int star) {
    const double p_max = 1.0 - (kNumOperators - 1) * cfg_.p_min;
    for (int i = 0; i < kNumOperators; i++)
        p_[i] += cfg_.beta * (((i == star) ? p_max : cfg_.p_min) - p_[i]);
}

void AdaptivePursuitSelector::updateSnap(const neat::GenerationReport& report) {
    const double range = report.stats.best - report.stats.worst;
    auto reward = [&](double f) { return range > 0.0 ? (f - report.stats.worst) / range : 0.0; };
    std::vector<double> sum(kNumOperators, 0.0);
    std::vector<int> n(kNumOperators, 0);

    if (generations_seen_ < cfg_.init_generations) {          // período inicial: P uniforme fijo
        if (init_rewards_.empty()) init_rewards_.assign(kNumOperators, {});
        for (const auto& [op, f] : report.last_op_scores) {
            int i = static_cast<int>(op);
            double r = reward(f);
            init_rewards_[i].push_back(r);
            sum[i] += r; n[i]++;
        }
        p_.assign(kNumOperators, 1.0 / kNumOperators);
        if (++generations_seen_ == cfg_.init_generations) {  // Q = media + 1 desv. est.
            for (int i = 0; i < kNumOperators; i++) {
                const auto& v = init_rewards_[i];
                if (v.empty()) continue;
                double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), var = 0.0;
                for (double x : v) var += (x - m) * (x - m);
                q_[i] = m + std::sqrt(v.size() > 1 ? var / (v.size() - 1) : 0.0);
            }
            pursue(static_cast<int>(std::max_element(q_.begin(), q_.end()) - q_.begin()));
        }
    } else {                                                 // actualización tras cada evaluación
        generations_seen_++;
        for (const auto& [op, f] : report.last_op_scores) {
            int i = static_cast<int>(op);
            double r = reward(f);
            q_[i] += cfg_.alpha * (r - q_[i]);
            pursue(static_cast<int>(std::max_element(q_.begin(), q_.end()) - q_.begin()));
            sum[i] += r; n[i]++;
        }
    }
    for (int i = 0; i < kNumOperators; i++)
        last_credit_[i] = n[i] ? sum[i] / n[i] : std::numeric_limits<double>::quiet_NaN();
}

void AdaptivePursuitSelector::update(const neat::GenerationReport& report) {
    if (cfg_.snap) { updateSnap(report); return; }

    std::vector<double> sum(kNumOperators, 0.0), best(kNumOperators, 0.0);
    std::vector<int> n(kNumOperators, 0);
    for (const auto& o : report.outcomes) {
        int i = static_cast<int>(o.op);
        double r = childReward(o, report.stats.best, report.stats.worst);
        sum[i] += r;
        best[i] = std::max(best[i], r);
        n[i]++;
    }

    bool any = false;
    for (int i = 0; i < kNumOperators; i++) {
        if (n[i] == 0) { last_credit_[i] = std::numeric_limits<double>::quiet_NaN(); continue; }
        double R = cfg_.extreme_credit ? best[i] : sum[i] / n[i];
        last_credit_[i] = R;
        q_[i] += cfg_.alpha * (R - q_[i]);
        any = true;
    }
    if (!any) return;

    pursue(static_cast<int>(std::max_element(q_.begin(), q_.end()) - q_.begin()));
}

}  // namespace aos
