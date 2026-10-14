#include "selector.hpp"

#include <algorithm>
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

std::vector<OpRank> rankCredit(const std::vector<Genome>& G, size_t K) {
    std::vector<OpRank> r(K);
    const int n = static_cast<int>(G.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return G[a].fitness < G[b].fitness; });
    std::vector<double> pct(n, 0.0);
    for (int lo = 0; lo < n;) {   // empates: percentil medio del grupo
        int hi = lo;
        while (hi + 1 < n && G[order[hi + 1]].fitness == G[order[lo]].fitness) hi++;
        const double q = n > 1 ? 0.5 * (lo + hi) / (n - 1) : 0.5;
        for (int k = lo; k <= hi; k++) pct[order[k]] = q;
        lo = hi + 1;
    }
    for (int i = 0; i < n; i++) {
        const Origin& og = G[i].origin;
        if (!og.has_operator || og.slot < 0 || og.slot >= static_cast<int>(K)) continue;
        OpRank& o = r[og.slot];
        o.n++;
        o.rank_sum += pct[i];
        if (pct[i] >= 0.8) o.top++;
        if (!og.applied) o.fail++;
    }
    return r;
}

StaticSelector StaticSelector::stanley2002() {
    return StaticSelector({0.8, 0.05, 0.03});
}

}  // namespace neat
