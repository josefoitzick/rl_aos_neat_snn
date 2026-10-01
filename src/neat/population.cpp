#include "population.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace neat {

Population::Population(const NeatConfig& cfg, int num_inputs, int num_outputs, uint64_t seed)
    : cfg_(cfg), rng_(seed), innov_(num_inputs + num_outputs),
      compat_threshold_(cfg.compat_threshold),
      best_ever_fitness_(-std::numeric_limits<double>::infinity()) {
    genomes_.reserve(cfg_.population_size);
    for (int k = 0; k < cfg_.population_size; k++)
        genomes_.push_back(Genome::minimal(next_genome_id_++, num_inputs, num_outputs, cfg_, innov_, rng_));
}

// Stanley 2002 §3.3: cada genoma va a la primera especie cuyo representante está a δ < δt.
void Population::speciate() {
    for (Species& s : species_) s.members.clear();
    for (int i = 0; i < static_cast<int>(genomes_.size()); i++) {
        bool placed = false;
        for (Species& s : species_) {
            if (Genome::compatibility(genomes_[i], s.representative, cfg_) < compat_threshold_) {
                s.members.push_back(i);
                placed = true;
                break;
            }
        }
        if (!placed)
            species_.push_back({next_species_id_++, genomes_[i], {i},
                                -std::numeric_limits<double>::infinity(), generation_});
    }
    species_.erase(std::remove_if(species_.begin(), species_.end(),
                                  [](const Species& s) { return s.members.empty(); }),
                   species_.end());
}

void Population::adaptThreshold() {
    if (cfg_.target_species <= 0) return;
    const int n = static_cast<int>(species_.size());
    if (n < cfg_.target_species)
        compat_threshold_ = std::max(cfg_.compat_threshold_min, compat_threshold_ - cfg_.compat_threshold_step);
    else if (n > cfg_.target_species)
        compat_threshold_ += cfg_.compat_threshold_step;
}

GenerationReport Population::endGeneration() {
    speciate();

    GenerationReport report;
    PopulationStats& st = report.stats;
    st.generation = generation_;
    st.num_species = static_cast<int>(species_.size());
    st.compat_threshold = compat_threshold_;
    adaptThreshold();   // δt para la próxima especiación

    const int n = static_cast<int>(genomes_.size());
    int champ = 0;
    double sum = 0.0, sum2 = 0.0, sum_hidden = 0.0, sum_conns = 0.0;
    st.worst = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; i++) {
        const Genome& g = genomes_[i];
        if (g.fitness > genomes_[champ].fitness) champ = i;
        sum += g.fitness; sum2 += g.fitness * g.fitness;
        st.worst = std::min(st.worst, g.fitness);
        sum_hidden += g.numHidden(); sum_conns += g.numEnabledConns();
        if (g.origin.has_operator)
            report.outcomes.push_back({g.origin.op, g.fitness, g.origin.parent_fitness});
        if (g.origin.last_op >= 0)
            report.last_op_scores.push_back({static_cast<Operator>(g.origin.last_op), g.fitness});
    }
    st.best = genomes_[champ].fitness;
    st.mean = sum / n;
    st.stdev = std::sqrt(std::max(0.0, sum2 / n - st.mean * st.mean));
    st.champion_hidden = genomes_[champ].numHidden();
    st.champion_conns = genomes_[champ].numEnabledConns();
    st.mean_hidden = sum_hidden / n;
    st.mean_conns = sum_conns / n;

    for (Species& s : species_) {
        double best = -std::numeric_limits<double>::infinity();
        for (int i : s.members) best = std::max(best, genomes_[i].fitness);
        if (best > s.best_fitness) { s.best_fitness = best; s.last_improved = generation_; }
    }
    if (st.best > best_ever_fitness_) {
        best_ever_fitness_ = st.best;
        best_ever_ = genomes_[champ];
        last_global_improvement_ = generation_;
    }

    ended_ = true;
    return report;
}

// Cuota de hijos por especie ∝ suma del fitness compartido (fitness desplazado a >= 0,
// porque los retornos de control suelen ser negativos), con resto mayor para sumar exacto.
std::vector<int> Population::allocateOffspring() {
    const int S = static_cast<int>(species_.size());
    double fmin = std::numeric_limits<double>::infinity(), fmax = -fmin;
    int champ = 0;
    for (int i = 0; i < static_cast<int>(genomes_.size()); i++) {
        fmin = std::min(fmin, genomes_[i].fitness);
        fmax = std::max(fmax, genomes_[i].fitness);
        if (genomes_[i].fitness > genomes_[champ].fitness) champ = i;
    }
    const double eps = 1e-6 * std::max(1.0, fmax - fmin);

    std::vector<bool> eligible(S, false);
    for (int k = 0; k < S; k++) {
        const Species& s = species_[k];
        bool has_champ = std::find(s.members.begin(), s.members.end(), champ) != s.members.end();
        eligible[k] = has_champ || (generation_ - s.last_improved) < cfg_.species_stagnation;
    }
    // Estancamiento global (nota 2 del paper): solo las 2 mejores especies se reproducen.
    if (generation_ - last_global_improvement_ >= cfg_.global_stagnation && S > 2) {
        std::vector<int> order(S);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return species_[a].best_fitness > species_[b].best_fitness; });
        std::fill(eligible.begin(), eligible.end(), false);
        eligible[order[0]] = eligible[order[1]] = true;
        last_global_improvement_ = generation_;
    }

    std::vector<double> share(S, 0.0);
    double total = 0.0;
    for (int k = 0; k < S; k++) {
        if (!eligible[k]) continue;
        const Species& s = species_[k];
        for (int i : s.members) share[k] += (genomes_[i].fitness - fmin + eps) / s.members.size();
        total += share[k];
    }

    const int P = cfg_.population_size;
    std::vector<int> count(S, 0);
    std::vector<std::pair<double, int>> remainders;
    int assigned = 0;
    for (int k = 0; k < S; k++) {
        if (!eligible[k] || total <= 0.0) continue;
        double quota = share[k] / total * P;
        count[k] = static_cast<int>(std::floor(quota));
        assigned += count[k];
        remainders.emplace_back(quota - count[k], k);
    }
    std::sort(remainders.begin(), remainders.end(), std::greater<>());
    for (size_t r = 0; assigned < P && !remainders.empty(); r = (r + 1) % remainders.size()) {
        count[remainders[r].second]++;
        assigned++;
    }
    if (assigned == 0) {   // nadie elegible: todo a la especie del campeón
        for (int k = 0; k < S; k++)
            if (std::find(species_[k].members.begin(), species_[k].members.end(), champ) != species_[k].members.end())
                count[k] = P;
    }
    return count;
}

void Population::reproduce(const std::vector<double>& op_probs) {
    if (!ended_) throw std::logic_error("Population::reproduce antes de endGeneration");
    if (static_cast<int>(op_probs.size()) != kNumOperators)
        throw std::invalid_argument("Population::reproduce: P debe tener kNumOperators elementos");

    innov_.newGeneration();
    std::discrete_distribution<int> pick_op(op_probs.begin(), op_probs.end());
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto randomIndex = [&](size_t n) { return std::uniform_int_distribution<size_t>(0, n - 1)(rng_); };

    const std::vector<int> count = allocateOffspring();
    std::vector<Genome> next;
    next.reserve(cfg_.population_size);
    std::vector<size_t> kept;   // especies que sobreviven (se mueven al final: el cruce
                                // interespecies puede leer cualquier especie durante el bucle)

    for (size_t k = 0; k < species_.size(); k++) {
        if (count[k] == 0) continue;
        Species& s = species_[k];
        std::vector<int> m = s.members;
        std::sort(m.begin(), m.end(),
                  [&](int a, int b) { return genomes_[a].fitness > genomes_[b].fitness; });

        int to_make = count[k];
        if (static_cast<int>(m.size()) >= cfg_.elitism_min_size) {     // élite sin cambios
            Genome elite = genomes_[m[0]];
            elite.id = next_genome_id_++;
            Origin keep;
            keep.last_op = elite.origin.last_op;   // la élite conserva su último operador
            elite.origin = keep;
            next.push_back(std::move(elite));
            to_make--;
        }

        int survivors = std::max(1, static_cast<int>(std::ceil(cfg_.survival_fraction * m.size())));
        survivors = std::max(survivors, std::min(2, static_cast<int>(m.size())));

        for (int c = 0; c < to_make; c++) {
            const Genome& p1 = genomes_[m[randomIndex(survivors)]];
            Genome child;
            double parent_ref;
            if (survivors == 1 || U(rng_) < cfg_.mutation_only_fraction) {
                child = p1;
                parent_ref = p1.fitness;
            } else {
                const Genome* p2 = nullptr;
                if (species_.size() > 1 && U(rng_) < cfg_.interspecies_rate) {
                    size_t other = randomIndex(species_.size() - 1);
                    if (other >= k) other++;
                    const auto& om = species_[other].members;
                    p2 = &genomes_[om[randomIndex(om.size())]];
                } else {
                    int j = m[randomIndex(survivors)];
                    for (int tries = 0; tries < 5 && &genomes_[j] == &p1; tries++) j = m[randomIndex(survivors)];
                    p2 = &genomes_[j];
                }
                bool p1_fitter = p1.fitness > p2->fitness || (p1.fitness == p2->fitness && U(rng_) < 0.5);
                child = p1_fitter ? Genome::crossover(p1, *p2, cfg_, rng_)
                                  : Genome::crossover(*p2, p1, cfg_, rng_);
                parent_ref = std::max(p1.fitness, p2->fitness);
            }
            Operator op = static_cast<Operator>(pick_op(rng_));
            child.apply(op, cfg_, innov_, rng_);          // un operador por hijo (esquema categórico)
            child.id = next_genome_id_++;
            child.fitness = 0.0;
            child.origin = Origin{true, op, parent_ref, static_cast<int>(op)};
            next.push_back(std::move(child));
        }

        kept.push_back(k);
    }

    std::vector<Species> survivors_species;
    for (size_t k : kept) {
        Species& s = species_[k];
        s.representative = genomes_[s.members[randomIndex(s.members.size())]];
        survivors_species.push_back(std::move(s));
    }
    species_ = std::move(survivors_species);
    genomes_ = std::move(next);
    generation_++;
    ended_ = false;
}

}  // namespace neat
