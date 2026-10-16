#include "features.hpp"

#include <cmath>
#include <stdexcept>

namespace aos {

FeatureExtractor::FeatureExtractor(const FeatureConfig& cfg)
    : cfg_(cfg), win_rank_(cfg.K, 0.0), win_n_(cfg.K, 0.0), win_fail_(cfg.K, 0.0), credit_(cfg.K, 0.0), fail_(cfg.K, 0.0),
      potencia_(cfg.potencia_base) {
    if (cfg_.K < 1) throw std::invalid_argument("FeatureExtractor: K < 1");
    if (cfg_.budget < 1) throw std::invalid_argument("FeatureExtractor: presupuesto < 1");
    if (!(cfg_.ref_rl != cfg_.ref_trivial)) throw std::invalid_argument("FeatureExtractor: J_RL igual a J_trivial");
    if (!(cfg_.compat_threshold0 > 0.0)) throw std::invalid_argument("FeatureExtractor: δt inicial no positivo");
    if (cfg_.target_species < 1) throw std::invalid_argument("FeatureExtractor: especies objetivo < 1");
    if (cfg_.op_pesos >= cfg_.K) throw std::invalid_argument("FeatureExtractor: índice de mutate_weights fuera del catálogo");
    if (!(cfg_.potencia_base > 0.0)) throw std::invalid_argument("FeatureExtractor: potencia base no positiva");
    if (cfg_.snn && cfg_.compacto) throw std::invalid_argument("FeatureExtractor: estado SNN y compacto a la vez");
}

void FeatureExtractor::addActividad(const Actividad& a) {
    act_sum_.ocultas_mudas += a.ocultas_mudas;
    act_sum_.ocultas_saturadas += a.ocultas_saturadas;
    act_sum_.entradas_mudas += a.entradas_mudas;
    act_sum_.salidas_mudas += a.salidas_mudas;
    act_sum_.accion_saturada += a.accion_saturada;
    act_n_++;
}

void FeatureExtractor::addGeneration(const neat::GenerationReport& rep) {
    if (!std::isfinite(rep.champion_validation) || !std::isfinite(rep.best_validation))
        throw std::logic_error("FeatureExtractor: el reporte no trae las validaciones (las llena el ciclo de NEAT)");
    if (rep.evaluations < 1) throw std::logic_error("FeatureExtractor: el reporte no trae las evaluaciones usadas");
    if (static_cast<int>(rep.op_rank.size()) != cfg_.K)
        throw std::logic_error("FeatureExtractor: crédito por rango de tamaño distinto de K");
    tau_ = static_cast<double>(rep.evaluations) / cfg_.budget;
    best_val_ = rep.best_validation;
    if (!any_best_ || rep.best_validation > best_seen_) {
        best_seen_ = rep.best_validation;
        tau_improve_ = tau_;
        any_best_ = true;
    }
    vgen_sum_ += sigma(rep.champion_validation);
    vgen_n_++;
    for (int k = 0; k < cfg_.K; k++) {
        win_rank_[k] += rep.op_rank[k].rank_sum;
        win_n_[k] += rep.op_rank[k].n;
        win_fail_[k] += rep.op_rank[k].fail;
    }
    last_ = rep.stats;
    have_gen_ = true;
}

std::vector<float> FeatureExtractor::closeWindow() {
    if (!have_gen_ || vgen_n_ == 0) throw std::logic_error("FeatureExtractor: ventana sin generaciones");
    const double vstar = sigma(best_val_);
    const double vgen = vgen_sum_ / vgen_n_;
    const double dvstar = first_close_ ? 0.0 : vstar - vbest_prev_close_;
    first_close_ = false;
    vbest_prev_close_ = vstar;
    for (int k = 0; k < cfg_.K; k++) {
        if (win_n_[k] >= 1.0) {
            credit_[k] = (1.0 - cfg_.ema) * credit_[k] + cfg_.ema * (win_rank_[k] / win_n_[k] - 0.5);
            fail_[k] = (1.0 - cfg_.ema) * fail_[k] + cfg_.ema * (win_fail_[k] / win_n_[k]);
        }
        win_rank_[k] = win_n_[k] = win_fail_[k] = 0.0;
    }
    vgen_sum_ = 0.0;
    vgen_n_ = 0;

    std::vector<float> o;
    o.reserve(dim());
    const double conns0 = cfg_.conns0 + 1.0;
    if (cfg_.snn) {
        if (act_n_ == 0) throw std::logic_error("FeatureExtractor: estado SNN sin actividad medida en la ventana");
        o.push_back(static_cast<float>(tau_));
        o.push_back(static_cast<float>(vstar));
        o.push_back(static_cast<float>(dvstar));
        o.push_back(static_cast<float>(tau_ - tau_improve_));
        o.push_back(static_cast<float>(std::log((last_.champion_conns + 1.0) / conns0)));
        o.push_back(static_cast<float>(act_sum_.ocultas_mudas / act_n_));
        o.push_back(static_cast<float>(act_sum_.ocultas_saturadas / act_n_));
        o.push_back(static_cast<float>(act_sum_.entradas_mudas / act_n_));
        o.push_back(static_cast<float>(act_sum_.salidas_mudas / act_n_));
        o.push_back(static_cast<float>(act_sum_.accion_saturada / act_n_));
        act_sum_ = Actividad{};
        act_n_ = 0;
        for (float v : o)
            if (!std::isfinite(v)) throw std::runtime_error("FeatureExtractor: rasgo no finito");
        return o;
    }
    if (cfg_.compacto) {
        o.push_back(static_cast<float>(tau_));
        o.push_back(static_cast<float>(vstar));
        o.push_back(static_cast<float>(dvstar));
        o.push_back(static_cast<float>(tau_ - tau_improve_));
        o.push_back(static_cast<float>(last_.stdev / (cfg_.ref_rl - cfg_.ref_trivial)));
        o.push_back(static_cast<float>(std::log(last_.compat_threshold / cfg_.compat_threshold0)));
        o.push_back(static_cast<float>(std::log((last_.champion_conns + 1.0) / conns0)));
        o.push_back(static_cast<float>(cfg_.op_pesos >= 0 ? credit_[cfg_.op_pesos] : 0.0));
        o.push_back(static_cast<float>(std::log(potencia_ / cfg_.potencia_base)));
        for (float v : o)
            if (!std::isfinite(v)) throw std::runtime_error("FeatureExtractor: rasgo no finito");
        return o;
    }
    o.push_back(static_cast<float>(tau_));
    o.push_back(static_cast<float>(vstar));
    o.push_back(static_cast<float>(vgen));
    o.push_back(static_cast<float>(vstar - vgen));
    o.push_back(static_cast<float>(dvstar));
    o.push_back(static_cast<float>(tau_ - tau_improve_));
    o.push_back(static_cast<float>(last_.stdev / (cfg_.ref_rl - cfg_.ref_trivial)));
    o.push_back(static_cast<float>(static_cast<double>(last_.num_species) / cfg_.target_species));
    o.push_back(static_cast<float>(last_.largest_species_share));
    o.push_back(static_cast<float>(std::log(last_.compat_threshold / cfg_.compat_threshold0)));
    o.push_back(static_cast<float>(std::log((last_.mean_conns + 1.0) / conns0)));
    o.push_back(static_cast<float>(std::log1p(last_.mean_hidden)));
    o.push_back(static_cast<float>(std::log((last_.champion_conns + 1.0) / conns0)));
    o.push_back(static_cast<float>(std::log1p(static_cast<double>(last_.champion_hidden))));
    for (double c : credit_) o.push_back(static_cast<float>(c));
    for (double f : fail_) o.push_back(static_cast<float>(f));
    for (float v : o)
        if (!std::isfinite(v)) throw std::runtime_error("FeatureExtractor: rasgo no finito");
    return o;
}

}  // namespace aos
