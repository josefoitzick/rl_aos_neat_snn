#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include "../../include/core/neuron.hpp"

// Simula una neurona con corriente de offset constante (como en el paper de Izhikevich).
// Devuelve los tiempos de spike en ms.
static std::vector<double> simulate(NeuronType type, double i_offset,
                                    double duration, double dt = 1.0) {
    IzhikevichNeuron neuron(type, i_offset);
    std::vector<double> spike_times;
    for (double t = 0; t < duration; t += dt) {
        neuron.step(dt);
        if (neuron.hasSpiked()) spike_times.push_back(t);
    }
    return spike_times;
}

// Calcula los intervalos inter-spike (ISI) a partir de los tiempos de spike.
static std::vector<double> computeISIs(const std::vector<double>& spikes) {
    std::vector<double> isis;
    isis.reserve(spikes.size());
    for (size_t i = 1; i < spikes.size(); i++)
        isis.push_back(spikes[i] - spikes[i - 1]);
    return isis;
}

// ─── 1. Disparo tónico ────────────────────────────────────────────────────────

TEST_CASE("RS: disparo tónico con corriente suficiente") {
    auto spikes = simulate(NeuronType::REGULAR_SPIKING, 10.0, 500.0);
    REQUIRE(spikes.size() >= 5);
}

TEST_CASE("FS: disparo tónico con corriente suficiente") {
    auto spikes = simulate(NeuronType::FAST_SPIKING, 10.0, 500.0);
    REQUIRE(spikes.size() >= 5);
}

TEST_CASE("IB: disparo con corriente suficiente") {
    auto spikes = simulate(NeuronType::INTRINSICALLY_BURSTING, 10.0, 500.0);
    REQUIRE(spikes.size() >= 5);
}

TEST_CASE("CH: disparo con corriente suficiente") {
    auto spikes = simulate(NeuronType::CHATTERING, 10.0, 500.0);
    REQUIRE(spikes.size() >= 5);
}

// ─── 2. Silencio sin corriente ────────────────────────────────────────────────

TEST_CASE("RS: sin corriente no hay spikes") {
    auto spikes = simulate(NeuronType::REGULAR_SPIKING, 0.0, 500.0);
    REQUIRE(spikes.empty());
}

TEST_CASE("FS: sin corriente no hay spikes") {
    auto spikes = simulate(NeuronType::FAST_SPIKING, 0.0, 500.0);
    REQUIRE(spikes.empty());
}

// ─── 3. Adaptación de frecuencia en RS ───────────────────────────────────────
// El parámetro d=8 (alto) aumenta u tras cada spike, reduciendo la tasa de disparo.
// → Los ISIs deben crecer con el tiempo (último ISI > primer ISI).

TEST_CASE("RS: adaptación de frecuencia (ISIs crecientes)") {
    auto spikes = simulate(NeuronType::REGULAR_SPIKING, 10.0, 1000.0);
    REQUIRE(spikes.size() >= 6);

    auto isis = computeISIs(spikes);

    // Comparar la media de los primeros 3 ISIs con los 3 últimos
    double early_mean = (isis[0] + isis[1] + isis[2]) / 3.0;
    double late_mean  = (isis[isis.size()-3] + isis[isis.size()-2] + isis[isis.size()-1]) / 3.0;

    INFO("ISI early mean: " << early_mean << " ms, late mean: " << late_mean << " ms");
    REQUIRE(late_mean > early_mean);
}

// ─── 4. Sin adaptación en FS ─────────────────────────────────────────────────
// FS: a=0.1 (recuperación rápida), d=2 (poca adaptación por spike).
// → Los ISIs deben mantenerse estables: cociente último/primero < 1.5

TEST_CASE("FS: sin adaptación significativa de frecuencia") {
    auto spikes = simulate(NeuronType::FAST_SPIKING, 10.0, 1000.0);
    REQUIRE(spikes.size() >= 10);

    auto isis = computeISIs(spikes);
    double ratio = isis.back() / isis.front();

    INFO("ISI ratio (last/first): " << ratio);
    REQUIRE(ratio < 1.5);
}

// ─── 5. FS dispara más rápido que RS ─────────────────────────────────────────
// FS tiene a=0.1 y d=2 vs RS a=0.02 y d=8 → mayor tasa de disparo sostenida.

TEST_CASE("FS dispara más rápido que RS con la misma corriente") {
    auto rs_spikes = simulate(NeuronType::REGULAR_SPIKING, 10.0, 500.0);
    auto fs_spikes = simulate(NeuronType::FAST_SPIKING,    10.0, 500.0);

    INFO("RS spikes: " << rs_spikes.size() << ", FS spikes: " << fs_spikes.size());
    REQUIRE(fs_spikes.size() > rs_spikes.size());
}

// ─── 6. IB: burst inicial ────────────────────────────────────────────────────
// IB: c=-55 → el voltaje se resetea más alto tras cada spike, lo que causa
// que los primeros spikes del episodio estén muy próximos entre sí (burst).
// → Al menos uno de los primeros ISIs debe ser muy corto (< 10 ms).

TEST_CASE("IB: burst inicial con ISIs cortos") {
    auto spikes = simulate(NeuronType::INTRINSICALLY_BURSTING, 10.0, 500.0);
    REQUIRE(spikes.size() >= 4);

    auto isis = computeISIs(spikes);

    // Buscar al menos un ISI muy corto en los primeros 5 intervalos
    size_t n = std::min(isis.size(), size_t(5));
    double min_early_isi = *std::min_element(isis.begin(), isis.begin() + n);

    INFO("Mínimo ISI temprano: " << min_early_isi << " ms");
    REQUIRE(min_early_isi < 10.0);
}

// ─── 7. CH: bursting rítmico ─────────────────────────────────────────────────
// CH: c=-50 → reset aún más alto → bursts rítmicos.
// El patrón alterna ISIs intra-burst (cortos) e inter-burst (largos).
// → El cociente entre el ISI máximo y el mínimo debe ser alto (distribución bimodal).

TEST_CASE("CH: bursting rítmico (ISIs bimodales)") {
    auto spikes = simulate(NeuronType::CHATTERING, 10.0, 1000.0);
    REQUIRE(spikes.size() >= 8);

    auto isis = computeISIs(spikes);
    double min_isi = *std::min_element(isis.begin(), isis.end());
    double max_isi = *std::max_element(isis.begin(), isis.end());

    INFO("ISI min: " << min_isi << " ms, max: " << max_isi << " ms, ratio: " << max_isi / min_isi);
    REQUIRE(max_isi / min_isi > 3.0);
}

// ─── 8. IB: adaptación hacia disparo tónico ──────────────────────────────────
// Tras el burst inicial, IB se estabiliza en disparo tónico regular.
// → Los ISIs tardíos deben ser mayores que los primeros (burst → tónico).

TEST_CASE("IB: transición de burst inicial a disparo tónico") {
    auto spikes = simulate(NeuronType::INTRINSICALLY_BURSTING, 10.0, 1000.0);
    REQUIRE(spikes.size() >= 8);

    auto isis = computeISIs(spikes);

    double early_min = *std::min_element(isis.begin(), isis.begin() + 3);
    double late_mean = (isis[isis.size()-3] + isis[isis.size()-2] + isis[isis.size()-1]) / 3.0;

    INFO("ISI early min: " << early_min << " ms, late mean: " << late_mean << " ms");
    REQUIRE(late_mean > early_min * 2.0);
}

// ─── 9. Consistencia de setNeuronType ────────────────────────────────────────
// Cambiar el tipo en caliente debe producir el mismo comportamiento
// que construir directamente con ese tipo.

TEST_CASE("setNeuronType: comportamiento equivalente al constructor") {
    // RS → FS en caliente
    IzhikevichNeuron n1(NeuronType::FAST_SPIKING, 10.0);
    IzhikevichNeuron n2(NeuronType::REGULAR_SPIKING, 10.0);
    n2.setNeuronType(NeuronType::FAST_SPIKING);
    n2.setIOffset(10.0);

    // Ambas neuronas deben producir la misma secuencia de spikes
    std::vector<bool> spikes1, spikes2;
    for (int i = 0; i < 200; i++) {
        n1.step(1.0);
        n2.step(1.0);
        spikes1.push_back(n1.hasSpiked());
        spikes2.push_back(n2.hasSpiked());
    }
    REQUIRE(spikes1 == spikes2);
}
