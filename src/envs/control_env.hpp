#ifndef ENVS_CONTROL_ENV_HPP
#define ENVS_CONTROL_ENV_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace envs {

struct StepResult {
    double reward;
    bool terminated;
};

// Interfaz mínima de un entorno de control. Las implementaciones envuelven entornos de
// RLtools. Una instancia NO es segura entre hilos: cada hilo crea la suya con makeEnv().
//
// Multiagente: obsDim() y actionDim() son POR AGENTE; observe() y step() trabajan con los
// vectores de todos los agentes concatenados (agente 0 primero). Con un agente es lo usual.
class ControlEnv {
public:
    virtual ~ControlEnv() = default;
    virtual std::string name() const = 0;
    virtual int obsDim() const = 0;
    virtual int actionDim() const = 0;
    virtual int numAgents() const { return 1; }
    // Observación no negativa (imágenes, [0, 1]): el evaluador usa 1 entrada SNN por dimensión en vez de 2
    virtual bool nonNegativeObs() const { return false; }
    virtual int maxSteps() const = 0;

    // Parámetros del episodio (p. ej. posiciones de las banderas en flag) y estado inicial,
    // muestreados con el RNG del entorno resembrado con `seed` (reproducible)
    virtual void reset(uint64_t seed) = 0;
    // Observación normalizada a [-1, 1] (numAgents() × obsDim())
    virtual std::vector<double> observe() = 0;
    // Observación sin normalizar (diagnóstico: env_probe mide con ella las cotas de normalización)
    virtual std::vector<double> rawObserve() = 0;
    // Acción en [-1, 1] por dimensión (numAgents() × actionDim())
    virtual StepResult step(const std::vector<double>& action) = 0;
};

struct EnvOptions {
    // Escala del torque máximo respecto del entorno por defecto de RLtools (1 = sin cambio).
    // Acrobot de RLtools usa ±5; con 0,2 queda en ±1, como Acrobot-v1 de gymnasium.
    // Solo afecta a pendulum* y acrobot*.
    double torque_scale = 1.0;
    // Entornos con escenas (l2f-visual): usar las escenas de test en vez de las de entrenamiento.
    // El resto de los entornos lo ignora.
    bool test_split = false;
};

// Nombres aceptados por makeEnv (docs/entornos_rltools.md)
std::vector<std::string> envNames();

std::unique_ptr<ControlEnv> makeEnv(const std::string& name, const EnvOptions& opt = EnvOptions());

}  // namespace envs

#endif
