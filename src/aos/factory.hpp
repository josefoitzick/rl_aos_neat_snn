#ifndef AOS_FACTORY_HPP
#define AOS_FACTORY_HPP

#include <memory>
#include <string>
#include "adaptive_pursuit.hpp"
#include "neat/selector.hpp"

namespace aos {

// Con el catálogo por defecto (los 3 operadores de Stanley):
// "static" → StaticSelector::stanley2002()
// "ap"     → AdaptivePursuitSelector; si apc.p_init está vacío parte del P de Stanley
//            (salvo init_uniform = true, que parte de 1/K como en Thierens 2005)
// Con otro catálogo no hay P de Stanley: ambos parten uniformes salvo que se dé P (--p o apc.p_init).
std::unique_ptr<neat::OperatorSelector> makeSelector(const std::string& name,
                                                     const AdaptivePursuitConfig& apc = AdaptivePursuitConfig(),
                                                     bool init_uniform = false,
                                                     const std::vector<neat::Operator>& catalog = neat::defaultCatalog());

}  // namespace aos

#endif
