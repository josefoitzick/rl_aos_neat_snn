#include "selector.hpp"

#include <numeric>
#include <stdexcept>

namespace neat {

StaticSelector::StaticSelector(std::vector<double> weights) : p_(std::move(weights)) {
    if (p_.empty() || static_cast<int>(p_.size()) > kNumOperatorTypes)
        throw std::invalid_argument("StaticSelector: se espera un peso por operador del catálogo");
    double sum = std::accumulate(p_.begin(), p_.end(), 0.0);
    if (sum <= 0.0) throw std::invalid_argument("StaticSelector: pesos no positivos");
    for (double& p : p_) p /= sum;
}

StaticSelector StaticSelector::stanley2002() {
    return StaticSelector({0.8, 0.05, 0.03});
}

}  // namespace neat
