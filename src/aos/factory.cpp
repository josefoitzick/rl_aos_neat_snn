#include "factory.hpp"

#include <stdexcept>

namespace aos {

std::unique_ptr<neat::OperatorSelector> makeSelector(const std::string& name,
                                                     const AdaptivePursuitConfig& apc, bool init_uniform) {
    if (name == "static")
        return std::make_unique<neat::StaticSelector>(neat::StaticSelector::stanley2002());
    if (name == "ap") {
        AdaptivePursuitConfig c = apc;
        if (c.p_init.empty() && !init_uniform && !c.snap)   // SNAP-NEAT parte uniforme
            c.p_init = neat::StaticSelector::stanley2002().probabilities();
        return std::make_unique<AdaptivePursuitSelector>(c);
    }
    throw std::invalid_argument("makeSelector: selector desconocido '" + name + "'");
}

}  // namespace aos
