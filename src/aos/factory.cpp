#include "factory.hpp"

#include <stdexcept>

namespace aos {

std::unique_ptr<neat::OperatorSelector> makeSelector(const std::string& name,
                                                     const AdaptivePursuitConfig& apc, bool init_uniform,
                                                     const std::vector<neat::Operator>& catalog) {
    const bool classic = catalog == neat::defaultCatalog();
    const int num_operators = static_cast<int>(catalog.size());
    if (name == "static")
        return std::make_unique<neat::StaticSelector>(
            classic ? neat::StaticSelector::stanley2002() : neat::StaticSelector(std::vector<double>(num_operators, 1.0)));
    if (name == "ap") {
        AdaptivePursuitConfig c = apc;
        c.num_operators = num_operators;
        if (c.p_init.empty() && !init_uniform && !c.snap && classic)   // SNAP-NEAT parte uniforme
            c.p_init = neat::StaticSelector::stanley2002().probabilities();
        return std::make_unique<AdaptivePursuitSelector>(c);
    }
    throw std::invalid_argument("makeSelector: selector desconocido '" + name + "'");
}

}  // namespace aos
