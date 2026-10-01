#ifndef AOS_FACTORY_HPP
#define AOS_FACTORY_HPP

#include <memory>
#include <string>
#include "adaptive_pursuit.hpp"
#include "neat/selector.hpp"

namespace aos {

// "static" → StaticSelector::stanley2002()
// "ap"     → AdaptivePursuitSelector; si apc.p_init está vacío parte del P de Stanley
//            (salvo init_uniform = true, que parte de 1/K como en Thierens 2005)
std::unique_ptr<neat::OperatorSelector> makeSelector(const std::string& name,
                                                     const AdaptivePursuitConfig& apc = AdaptivePursuitConfig(),
                                                     bool init_uniform = false);

}  // namespace aos

#endif
