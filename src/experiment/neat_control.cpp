// NEAT + SNN en tareas de control (RLtools), con selector de operadores estático o Adaptive Pursuit.
//
// Protocolo (docs/diseno_neat_ap.md §8–§9):
//  - Entrenamiento con números aleatorios comunes: todos los genomas de la corrida se evalúan en
//    los mismos E episodios (semillas derivadas de --seed, stream 0).
//  - El campeón de cada generación se mide en M episodios de test no vistos (stream 1). Esas
//    evaluaciones de medición no cuentan en el presupuesto.
//  - Al final, el mejor genoma histórico se mide en F episodios nuevos (stream 2).
//  - Con --resample-train (docs/diseno_neat_ap.md §15.4, paso 5): cada generación usa E episodios nuevos
//    (stream 1000 + gen), comunes a toda la población, y el genoma final es el campeón de generación con
//    mejor test (stream 1, validación). Sin la opción, el comportamiento es idéntico al anterior.
//  - Sonda de P (docs/diseno_neat_ap.md §16): --p fija el P del selector estático (se normaliza); con
//    --p2 y --p-switch G, las generaciones G en adelante se producen con P2. Sin estas opciones, el
//    comportamiento es idéntico al anterior (P de Stanley).
//  - Catálogo ampliado (docs/operadores_candidatos.md): --catalog elige los operadores (P y los selectores trabajan
//    sobre ese orden); --credito-rango agrega al CSV, por operador, el percentil medio de sus hijos en la generación,
//    la fracción de ellos en el 20 % superior y la fracción de aplicaciones fallidas. Sin estas opciones, el
//    comportamiento y las salidas son idénticos a los anteriores.
//  - CSV por generación en --out/generations.csv; resumen en --out/summary.txt.
#include <CLI/CLI.hpp>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "aos/factory.hpp"
#include "eval/evaluator.hpp"
#include "neat/population.hpp"

using namespace neat;

// "a,b,c" → vector de K pesos no negativos (uno por operador del catálogo), normalizado a suma 1.
static std::vector<double> parseP(const std::string& txt, size_t K) {
    std::vector<double> w;
    std::stringstream ss(txt);
    std::string tok;
    while (std::getline(ss, tok, ',')) w.push_back(std::stod(tok));
    if (w.size() != K)
        throw std::invalid_argument("--p/--p2: se esperan " + std::to_string(K) + " valores separados por coma");
    for (double x : w) if (x < 0.0) throw std::invalid_argument("--p/--p2: valores negativos");
    return StaticSelector(w).probabilities();
}

static std::string joinP(const std::vector<double>& P, const char* fmt, const char* sep) {
    std::string out;
    char buf[64];
    for (size_t i = 0; i < P.size(); i++) {
        std::snprintf(buf, sizeof buf, fmt, P[i]);
        out += (i ? sep : "") + std::string(buf);
    }
    return out;
}

int main(int argc, char** argv) {
    CLI::App app{"NEAT + SNN (Izhikevich) en control con RLtools"};
    std::string env_name = "pendulum", selector_name = "static", out_dir;
    envs::EnvOptions env_opt;
    eval::EvalConfig ecfg;
    NeatConfig ncfg;
    aos::AdaptivePursuitConfig apc;
    bool ap_uniform = false;
    uint64_t seed = 1;
    int generations = 100, episodes = 5, test_episodes = 10, final_episodes = 100, print_every = 10;

    app.add_option("--env", env_name, "Entorno (docs/entornos_rltools.md)")->check(CLI::IsMember(envs::envNames()));
    app.add_option("--torque-scale", env_opt.torque_scale, "Escala del torque máximo (acrobot 0.2 → ±1 como gymnasium)");
    app.add_option("--selector", selector_name, "static | ap")->check(CLI::IsMember({"static", "ap"}));
    app.add_option("--ap-alpha", apc.alpha);
    app.add_option("--ap-beta", apc.beta);
    app.add_option("--ap-pmin", apc.p_min);
    app.add_flag("--ap-extreme", apc.extreme_credit, "Crédito extremo (máximo)");
    app.add_flag("--ap-uniform", ap_uniform, "P inicial uniforme (en vez del P de Stanley)");
    app.add_flag("--ap-snap", apc.snap, "Crédito de SNAP-NEAT (fitness normalizado del último operador, secuencial)");
    app.add_option("--ap-init-gens", apc.init_generations, "SNAP: generaciones del período inicial uniforme");
    app.add_option("--seed", seed, "Semilla de la réplica (población inicial y episodios)");
    app.add_option("--generations", generations);
    app.add_option("--pop", ncfg.population_size);
    app.add_option("--episodes", episodes, "Episodios de entrenamiento por evaluación (E)");
    app.add_option("--test-episodes", test_episodes, "Episodios de test del campeón por generación (M)");
    app.add_option("--final-episodes", final_episodes, "Episodios de test final del mejor genoma (F)");
    app.add_option("--step-ms", ecfg.step_ms, "Ventana SNN por paso de control (ms)");
    app.add_option("--input-gain", ecfg.input_gain, "Corriente para |obs| = 1");
    app.add_option("--spikes-full", ecfg.spikes_full_action, "Diferencia de spikes para |acción| = 1");
    app.add_flag("--keep-state", ecfg.keep_state, "Reiniciar la SNN solo al comienzo del episodio (política con memoria)");
    std::string p_txt, p2_txt;
    int p_switch = 0;
    app.add_option("--p", p_txt, "P fijo del selector estático: pesos,conexión,nodo (se normaliza)");
    app.add_option("--p2", p2_txt, "P desde la generación --p-switch en adelante (requiere --p-switch)");
    app.add_option("--p-switch", p_switch, "Generación desde la que se usa --p2");
    bool resample_train = false;
    app.add_flag("--resample-train", resample_train,
                 "Episodios de entrenamiento nuevos en cada generación (los mismos para toda la población); "
                 "el campeón final se elige por el test de validación de cada generación");
    // Calibración de NEAT
    double weight_replace_prob = 1.0 - ncfg.weight_perturb_prob;
    app.add_option("--compat-threshold", ncfg.compat_threshold, "δt inicial");
    app.add_option("--target-species", ncfg.target_species, "Especies objetivo (0 = δt fijo)");
    app.add_option("--compat-weight-scale", ncfg.compat_weight_scale);
    app.add_option("--weight-power", ncfg.weight_perturb_power, "Perturbación de pesos ~ U(-p, p)");
    app.add_option("--weight-replace-prob", weight_replace_prob, "Prob. por peso de valor nuevo (Stanley: 0,1)");
    app.add_option("--add-node-weight", ncfg.add_node_in_weight);
    app.add_option("--survival", ncfg.survival_fraction);
    app.add_option("--weight-init-range", ncfg.weight_init_range, "Peso inicial / nuevo ~ U(-r, r)");
    app.add_option("--weight-max", ncfg.weight_max, "Límite |w|");
    app.add_option("--mutation-only", ncfg.mutation_only_fraction, "Fracción de hijos sin cruce");
    app.add_option("--species-stagnation", ncfg.species_stagnation, "Generaciones sin mejora antes de que una especie deje de reproducirse");
    // Catálogo ampliado (docs/operadores_candidatos.md)
    std::string catalog_txt;
    bool credito_rango = false;
    app.add_option("--catalog", catalog_txt, "Operadores del catálogo, separados por coma (por defecto los 3 de Stanley)");
    app.add_flag("--credito-rango", credito_rango, "CSV: percentil, 20 % superior y fallas de los hijos de cada operador");
    app.add_option("--c4", ncfg.c4, "Peso de la diferencia de parámetros de nodo en δ");
    app.add_option("--fine-sigma", ncfg.fine_weight_sigma, "weights_fine: σ de la perturbación");
    app.add_flag("--fine-cauchy", ncfg.fine_weight_cauchy, "weights_fine con perturbación de Cauchy");
    app.add_option("--directed-iso", ncfg.directed_iso_sigma, "weights_directed: σ isotrópico");
    app.add_option("--directed-line", ncfg.directed_line_sigma, "weights_directed: σ sobre la línea entre élites");
    app.add_option("--bias-sigma", ncfg.bias_sigma, "neuron_bias: σ");
    app.add_option("--bias-max", ncfg.bias_max, "neuron_bias: |sesgo| máximo");
    app.add_option("--gain-log-sigma", ncfg.gain_log_sigma, "input_gain: σ en escala logarítmica");
    app.add_option("--multi-k", ncfg.add_node_multi_k, "add_node_multi: nodos por aplicación");
    app.add_option("--out", out_dir, "Directorio de salida (CSV y resumen)");
    app.add_option("--print-every", print_every);
    CLI11_PARSE(app, argc, argv);
    ncfg.weight_perturb_prob = 1.0 - weight_replace_prob;
    if (!catalog_txt.empty()) ncfg.catalog = parseCatalog(catalog_txt);
    const bool default_catalog = ncfg.catalog == defaultCatalog();
    const size_t K = ncfg.catalog.size();
    if (!default_catalog && selector_name == "static" && p_txt.empty())
        throw std::invalid_argument("--catalog distinto del de Stanley con selector estático: dar --p");

    // El campeón se mide en episodios de test; en entornos con escenas (l2f-visual), además en escenas de test
    envs::EnvOptions test_opt = env_opt;
    test_opt.test_split = true;
    auto probe = envs::makeEnv(env_name, test_opt);
    const int n_in = eval::numInputs(*probe), n_out = eval::numOutputs(*probe);
    auto sel = aos::makeSelector(selector_name, apc, ap_uniform, ncfg.catalog);
    std::vector<double> P2;
    if (!p_txt.empty() || !p2_txt.empty() || p_switch != 0) {
        if (selector_name != "static") throw std::invalid_argument("--p/--p2/--p-switch solo con --selector static");
        if (!p_txt.empty()) sel = std::make_unique<StaticSelector>(parseP(p_txt, K));
        if (p2_txt.empty() != (p_switch <= 0)) throw std::invalid_argument("--p2 y --p-switch van juntos (p-switch > 0)");
        if (!p2_txt.empty()) P2 = parseP(p2_txt, K);
    }

    const auto train_seeds = eval::episodeSeeds(seed, episodes, 0);
    const auto test_seeds  = eval::episodeSeeds(seed, test_episodes, 1);
    const auto final_seeds = eval::episodeSeeds(seed, final_episodes, 2);

    std::ofstream csv;
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
        csv.open(out_dir + "/generations.csv");
        csv << "gen,evaluations,best_train,mean_train,std_train,champion_test,species,compat_threshold,"
               "champion_hidden,champion_conns,mean_hidden,mean_conns,seconds";
        for (Operator o : ncfg.catalog) {
            const char* op = operatorName(o);
            csv << ",p_" << op << ",n_" << op << ",improved_" << op;
        }
        if (credito_rango)
            for (Operator o : ncfg.catalog) {
                const char* op = operatorName(o);
                csv << ",rank_" << op << ",top_" << op << ",fail_" << op;
            }
        csv << "\n";
    }

    std::printf("env=%s (%d agente(s); obs %d → %d entradas, acción %d → %d salidas)  torque_scale=%.2f  keep_state=%d  selector=%s  "
                "seed=%llu  pop=%d  gens=%d  E=%d  hilos=%d\n",
                env_name.c_str(), probe->numAgents(), probe->obsDim(), n_in, probe->actionDim(), n_out, env_opt.torque_scale,
                static_cast<int>(ecfg.keep_state),
                sel->name().c_str(), static_cast<unsigned long long>(seed), ncfg.population_size, generations,
                episodes, omp_get_max_threads());

    if (!default_catalog) {
        std::printf("catálogo:");
        for (Operator o : ncfg.catalog) std::printf(" %s", operatorName(o));
        std::printf("\n");
    }
    if (!p_txt.empty() || !P2.empty()) {
        const auto P1 = sel->probabilities();
        std::printf("P fijo = (%s)", joinP(P1, "%.3f", " ").c_str());
        if (!P2.empty()) std::printf("; desde la generación %d, P2 = (%s)", p_switch, joinP(P2, "%.3f", " ").c_str());
        std::printf("\n");
    }
    if (resample_train)
        std::printf("resample_train=1: episodios de entrenamiento distintos en cada generación (stream 1000 + gen); "
                    "campeón final = el de mejor test de validación\n");

    Population pop(ncfg, n_in, n_out, seed);
    // Con --resample-train el máximo histórico de fitness de entrenamiento está sesgado (generaciones con
    // episodios "fáciles"): el campeón final es el campeón de generación con mejor test (episodios del
    // stream 1, validación). El test final usa el stream 2, que no interviene en la elección.
    Genome best_val_genome;
    double best_val = -std::numeric_limits<double>::infinity();
    std::vector<double> P = sel->probabilities();   // P con el que se generó la generación actual
    auto t_start = std::chrono::steady_clock::now();

    for (int gen = 0; gen < generations; gen++) {
        auto t0 = std::chrono::steady_clock::now();
        const auto gen_seeds = resample_train
            ? eval::episodeSeeds(seed, episodes, 1000 + static_cast<uint64_t>(gen))
            : train_seeds;
        eval::evaluatePopulation(pop.genomes(), ncfg, env_name, env_opt, gen_seeds, ecfg);

        const auto& G = pop.genomes();
        size_t champ = 0;
        for (size_t i = 1; i < G.size(); i++) if (G[i].fitness > G[champ].fitness) champ = i;
        const double champ_test = eval::evaluateGenome(G[champ], ncfg, *probe, test_seeds, ecfg);
        if (resample_train && champ_test > best_val) {
            best_val = champ_test;
            best_val_genome = G[champ];
        }

        // Crédito por rango (--credito-rango): percentil del hijo entre todos los genomas de su generación
        std::vector<double> rank_sum(K, 0.0);
        std::vector<int> top_n(K, 0), fail_n(K, 0);
        if (credito_rango) {
            const int n = static_cast<int>(G.size());
            std::vector<int> order(n);
            for (int i = 0; i < n; i++) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return G[a].fitness < G[b].fitness; });
            std::vector<double> pct(n, 0.0);
            for (int lo = 0; lo < n;) {              // empates: percentil medio del grupo
                int hi = lo;
                while (hi + 1 < n && G[order[hi + 1]].fitness == G[order[lo]].fitness) hi++;
                const double r = n > 1 ? 0.5 * (lo + hi) / (n - 1) : 0.5;
                for (int k = lo; k <= hi; k++) pct[order[k]] = r;
                lo = hi + 1;
            }
            for (int i = 0; i < n; i++) {
                const Origin& og = G[i].origin;
                if (!og.has_operator || og.slot < 0) continue;
                rank_sum[og.slot] += pct[i];
                if (pct[i] >= 0.8) top_n[og.slot]++;
                if (!og.applied) fail_n[og.slot]++;
            }
        }

        GenerationReport rep = pop.endGeneration();
        std::vector<int> n_op(K, 0), improved(K, 0);
        for (const auto& o : rep.outcomes) {
            n_op[o.slotIndex()]++;
            if (o.child_fitness > o.parent_fitness) improved[o.slotIndex()]++;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto& st = rep.stats;

        if (csv.is_open()) {
            csv << gen << "," << static_cast<long>(gen + 1) * ncfg.population_size << "," << st.best << ","
                << st.mean << "," << st.stdev << "," << champ_test << "," << st.num_species << ","
                << st.compat_threshold << "," << st.champion_hidden << "," << st.champion_conns << "," << st.mean_hidden << ","
                << st.mean_conns << "," << secs;
            for (size_t i = 0; i < K; i++)
                csv << "," << P[i] << "," << n_op[i] << ","
                    << (n_op[i] ? static_cast<double>(improved[i]) / n_op[i] : 0.0);
            if (credito_rango)
                for (size_t i = 0; i < K; i++) {
                    if (n_op[i]) csv << "," << rank_sum[i] / n_op[i] << "," << static_cast<double>(top_n[i]) / n_op[i]
                                     << "," << static_cast<double>(fail_n[i]) / n_op[i];
                    else csv << ",,,";
                }
            csv << "\n";
            csv.flush();
        }
        if (gen % print_every == 0 || gen == generations - 1)
            std::printf("gen %4d  best %9.2f  mean %9.2f  test %9.2f  especies %3d  campeón %2d ocultas/%3d conex  "
                        "P=(%s)  %.2fs\n",
                        gen, st.best, st.mean, champ_test, st.num_species, st.champion_hidden, st.champion_conns,
                        joinP(P, "%.3f", " ").c_str(), secs);

        if (gen == generations - 1) break;
        sel->update(rep);
        P = sel->probabilities();
        if (!P2.empty() && gen + 1 >= p_switch) P = P2;   // la generación gen + 1 se produce con P2
        pop.reproduce(P);
    }

    const Genome& best = resample_train ? best_val_genome : pop.bestEver();
    const double final_test = eval::evaluateGenome(best, ncfg, *probe, final_seeds, ecfg);
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("\nMejor genoma histórico: fitness entrenamiento %.2f · test final (%d episodios) %.2f · "
                "%d ocultas, %d conexiones · %.1f s en total\n",
                best.fitness, final_episodes, final_test, best.numHidden(), best.numEnabledConns(), total);

    if (!out_dir.empty()) {
        std::ofstream sum(out_dir + "/summary.txt");
        sum << "env=" << env_name << "\ntorque_scale=" << env_opt.torque_scale << "\nkeep_state=" << ecfg.keep_state << "\nselector=" << sel->name()
            << "\nseed=" << seed << "\npop=" << ncfg.population_size << "\ngenerations=" << generations
            << "\nepisodes=" << episodes << "\ntarget_species=" << ncfg.target_species
            << "\ncompat_threshold=" << ncfg.compat_threshold << "\nweight_power=" << ncfg.weight_perturb_power
            << "\nweight_replace_prob=" << 1.0 - ncfg.weight_perturb_prob << "\nevaluations=" << static_cast<long>(generations) * ncfg.population_size
            << "\nbest_train=" << best.fitness << "\nfinal_test=" << final_test
            << "\nfinal_test_episodes=" << final_episodes << "\nbest_hidden=" << best.numHidden()
            << "\nbest_conns=" << best.numEnabledConns() << "\nseconds=" << total << "\n";
        if (resample_train) sum << "resample_train=1\nbest_validation_test=" << best_val << "\n";
        {
            const NeatConfig d;
            const eval::EvalConfig de;
            if (ncfg.population_size != d.population_size || ncfg.compat_weight_scale != d.compat_weight_scale ||
                ncfg.survival_fraction != d.survival_fraction || ncfg.add_node_in_weight != d.add_node_in_weight ||
                ncfg.weight_init_range != d.weight_init_range || ncfg.weight_max != d.weight_max ||
                ncfg.mutation_only_fraction != d.mutation_only_fraction || ncfg.species_stagnation != d.species_stagnation ||
                ecfg.input_gain != de.input_gain || ecfg.spikes_full_action != de.spikes_full_action || ecfg.step_ms != de.step_ms)
                sum << "compat_weight_scale=" << ncfg.compat_weight_scale << "\nsurvival=" << ncfg.survival_fraction
                    << "\nadd_node_weight=" << ncfg.add_node_in_weight << "\nweight_init_range=" << ncfg.weight_init_range
                    << "\nweight_max=" << ncfg.weight_max << "\nmutation_only=" << ncfg.mutation_only_fraction
                    << "\nspecies_stagnation=" << ncfg.species_stagnation << "\ninput_gain=" << ecfg.input_gain
                    << "\nspikes_full=" << ecfg.spikes_full_action << "\nstep_ms=" << ecfg.step_ms << "\n";
        }
        if (!p_txt.empty() || !P2.empty()) {
            auto join = [](const std::vector<double>& v) {
                std::ostringstream os;
                for (size_t i = 0; i < v.size(); i++) os << (i ? "," : "") << v[i];
                return os.str();
            };
            sum << "p=" << join(sel->probabilities()) << "\n";
            if (!P2.empty()) sum << "p2=" << join(P2) << "\np_switch=" << p_switch << "\n";
        }
        if (!default_catalog) {
            sum << "catalog=";
            for (size_t i = 0; i < K; i++) sum << (i ? "," : "") << operatorName(ncfg.catalog[i]);
            sum << "\nc4=" << ncfg.c4 << "\nfine_sigma=" << ncfg.fine_weight_sigma << "\nfine_cauchy=" << ncfg.fine_weight_cauchy
                << "\ndirected_iso=" << ncfg.directed_iso_sigma << "\ndirected_line=" << ncfg.directed_line_sigma
                << "\nbias_sigma=" << ncfg.bias_sigma << "\nbias_max=" << ncfg.bias_max
                << "\ngain_log_sigma=" << ncfg.gain_log_sigma << "\nmulti_k=" << ncfg.add_node_multi_k << "\n";
        }
    }
    return 0;
}
