#include <CLI/CLI.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "../../include/core/neuron.hpp"

struct NeuronTypeInfo {
    NeuronType type;
    const char* name;
};

static const std::vector<NeuronTypeInfo> kNeuronTypes = {
    {NeuronType::REGULAR_SPIKING,        "REGULAR_SPIKING"},
    {NeuronType::FAST_SPIKING,           "FAST_SPIKING"},
    {NeuronType::INTRINSICALLY_BURSTING, "INTRINSICALLY_BURSTING"},
    {NeuronType::CHATTERING,             "CHATTERING"},
    {NeuronType::LOW_THRESHOLD_SPIKING,  "LOW_THRESHOLD_SPIKING"},
    {NeuronType::RESONATOR,              "RESONATOR"},
};

int main(int argc, char** argv) {
    CLI::App app{"Inyección de corriente directa en neuronas individuales (una por tipo)"};

    int    steps      = 100;
    double dt         = 1.0;
    double current    = 10.0;
    int    start_step = 10;
    std::string output = "current_injection.csv";

    app.add_option("-n,--steps",      steps,      "Cantidad de steps a simular")->check(CLI::PositiveNumber);
    app.add_option("-d,--timestep",   dt,         "Timestep (ms)")->check(CLI::PositiveNumber);
    app.add_option("-c,--current",    current,    "Corriente directa inyectada (mA)");
    app.add_option("-s,--start-step", start_step, "Step en el que comienza la inyección de corriente")->check(CLI::NonNegativeNumber);
    app.add_option("-o,--output",     output,     "Archivo CSV de salida");

    CLI11_PARSE(app, argc, argv);

    std::vector<IzhikevichNeuron> neurons;
    for (const auto& info : kNeuronTypes)
        neurons.emplace_back(info.type);

    std::ofstream csv(output);
    if (!csv.is_open()) {
        std::cerr << "No se pudo abrir el archivo de salida: " << output << "\n";
        return 1;
    }

    csv << "step,current";
    for (const auto& info : kNeuronTypes)
        csv << "," << info.name;
    csv << "\n";

    for (int step = 0; step < steps; step++) {
        double injected_current = (step >= start_step) ? current : 0.0;

        for (auto& neuron : neurons) {
            neuron.setIOffset(injected_current);
            neuron.step(dt);
        }

        csv << step << "," << injected_current;
        for (const auto& neuron : neurons)
            csv << "," << neuron.getVoltage();
        csv << "\n";
    }

    csv.close();
    std::cout << "Simulación completada. Resultados guardados en " << output << "\n";

    return 0;
}
