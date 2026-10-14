// Verificación V1 de NEAT como entorno del agente PPO (docs/diseno_ppo.md §5).
//
// Corre NeatRun (una corrida, secuencial) o NeatPool (--paralelo N: N corridas con semillas consecutivas, en
// paralelo, por el mismo camino que usará el entrenamiento) con las mismas opciones que neat_control y una P fija o
// una secuencia de dos P (--p, --p2, --p-switch, como neat_control). Escribe generations.csv y summary.txt con el
// formato de neat_control; scripts/verificar_neat_run.sh compara ambos bit a bit, salvo la columna de segundos.
//
// En NeatRun la P solo cambia en las fronteras de decisión: --p-switch debe ser g_t + 1 para alguna frontera g_t.
#include <CLI/CLI.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "aos/action_map.hpp"
#include "aos/factory.hpp"
#include "aos/neat_run.hpp"
#include "experiment/opciones_neat.hpp"
#include "neat/selector.hpp"

namespace {

// Igual que parseP de neat_control: K pesos no negativos, normalizados por StaticSelector
std::vector<double> parseP(const std::string& txt, size_t K) {
    std::vector<double> w;
    std::stringstream ss(txt);
    std::string tok;
    while (std::getline(ss, tok, ',')) w.push_back(std::stod(tok));
    if (w.size() != K) throw std::invalid_argument("--p/--p2: se esperan " + std::to_string(K) + " valores separados por coma");
    for (double x : w)
        if (x < 0.0) throw std::invalid_argument("--p/--p2: valores negativos");
    return neat::StaticSelector(w).probabilities();
}

void escribirResumen(const std::string& dir, double final_test, double best_val) {
    std::filesystem::create_directories(dir);
    std::ofstream sum(dir + "/summary.txt");
    sum << "final_test=" << final_test << "\nbest_validation_test=" << best_val << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Verificación V1: NEAT como entorno (NeatRun / NeatPool) frente a neat_control"};
    opciones_neat::Opciones on;
    opciones_neat::agregar(app, on);
    int T = 10, paralelo = 0, p_switch = 0, print_every = 0;
    std::string p_txt, p2_txt, out_dir, selector = "static";
    bool credito_rango = false;
    app.add_option("--T", T, "Decisiones por corrida");
    app.add_option("--p", p_txt, "P fija (como neat_control)");
    app.add_option("--p2", p2_txt, "P desde la generación --p-switch");
    app.add_option("--p-switch", p_switch, "Generación desde la que se usa --p2 (debe ser una frontera + 1)");
    app.add_flag("--credito-rango", credito_rango, "Columnas de crédito por rango en el CSV");
    app.add_option("--out", out_dir, "Directorio de salida")->required();
    app.add_option("--paralelo", paralelo, "N corridas (semillas seed … seed+N−1) en NeatPool, en paralelo");
    // Aceptadas para poder pasar la línea de comando de neat_control tal cual
    app.add_option("--selector", selector)->check(CLI::IsMember({"static"}));
    app.add_option("--print-every", print_every);
    CLI11_PARSE(app, argc, argv);
    try {
        opciones_neat::finalizar(on);
        if (!on.resample_train) throw std::invalid_argument("NeatRun usa rotación de episodios: falta --resample-train");
        const size_t K = on.ncfg.catalog.size();
        const bool default_catalog = on.ncfg.catalog == neat::defaultCatalog();
        std::vector<double> P1, P2;
        if (p_txt.empty()) {
            if (!default_catalog) throw std::invalid_argument("--catalog distinto del de Stanley: dar --p");
            aos::AdaptivePursuitConfig apc;
            P1 = aos::makeSelector("static", apc, false, on.ncfg.catalog)->probabilities();
        } else {
            P1 = neat::StaticSelector(parseP(p_txt, K)).probabilities();   // como neat_control: parseP y StaticSelector
        }
        if (p2_txt.empty() != (p_switch <= 0)) throw std::invalid_argument("--p2 y --p-switch van juntos (p-switch > 0)");
        if (!p2_txt.empty()) P2 = parseP(p2_txt, K);

        aos::NeatRunConfig cfg;
        cfg.env_name = on.env_name;
        cfg.env_opt = on.env_opt;
        cfg.ecfg = on.ecfg;
        cfg.ncfg = on.ncfg;
        cfg.generations = on.generations;
        cfg.episodes = on.episodes;
        cfg.test_episodes = on.test_episodes;
        cfg.final_episodes = on.final_episodes;
        cfg.reward_episodes = 0;
        cfg.T = T;
        const auto bounds = aos::decisionBoundaries(cfg.generations, T);
        if (!P2.empty()) {
            bool alineado = false;
            for (int t = 0; t < T; t++) alineado |= bounds[t] + 1 == p_switch;
            if (!alineado) throw std::invalid_argument("--p-switch no coincide con una frontera de decisión + 1");
        }
        auto Pde = [&](int t) { return (!P2.empty() && bounds[t] + 1 >= p_switch) ? P2 : P1; };

        if (paralelo <= 0) {
            aos::NeatRun run(cfg, on.seed);
            std::filesystem::create_directories(out_dir);
            run.enableCsv(out_dir + "/generations.csv", credito_rango, P1);
            run.startSequential();
            for (int t = 0; t < T; t++) run.advanceSequential(Pde(t));
            const double ft = run.finalTest();
            escribirResumen(out_dir, ft, run.bestValidation());
            std::printf("semilla %llu: %d generaciones, test final %.6g\n", static_cast<unsigned long long>(on.seed), run.generation() + 1, ft);
        } else {
            aos::NeatPool pool(cfg, paralelo, on.seed, 0.0, 1.0, true);
            pool.enableCsv(out_dir, credito_rango, P1);
            std::vector<int> all(paralelo);
            for (int s = 0; s < paralelo; s++) all[s] = s;
            pool.start(all);
            for (int t = 0; t < T; t++) pool.advance(std::vector<std::vector<double>>(paralelo, Pde(t)));
            for (const auto& f : pool.takeFinished()) {
                char sub[32];
                std::snprintf(sub, sizeof sub, "seed_%02llu", static_cast<unsigned long long>(f.seed));
                // best_validation en crudo: se recupera de la corrida (Finished guarda su σ)
                double bv = 0.0;
                for (int s = 0; s < paralelo; s++)
                    if (pool.run(s).seed() == f.seed) bv = pool.run(s).bestValidation();
                escribirResumen(out_dir + "/" + sub, f.final_test, bv);
                std::printf("semilla %llu: test final %.6g\n", static_cast<unsigned long long>(f.seed), f.final_test);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    return 0;
}
