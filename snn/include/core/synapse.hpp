#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

// Sinapsis de conductancia: cuando la neurona presináptica dispara, la
// postsináptica recibe `weight` en g_exc (excitatoria) o en g_inh (inhibitoria).
// El peso es una magnitud (>= 0); el signo lo define el tipo de sinapsis.
class Synapse {
private:
    int pre_neuron_id;
    int post_neuron_id;
    bool g_type;    // true = excitatoria, false = inhibitoria
    bool enabled;   // false = deshabilitada (no participa en la simulación)
    double weight;  // magnitud de la conductancia transmitida por spike

public:
    Synapse(int pre_id, int post_id, bool excitatory, double weight = 1.0)
        : pre_neuron_id(pre_id),
          post_neuron_id(post_id),
          g_type(excitatory),
          enabled(true),
          weight(weight) {}

    // Getters
    int getPreNeuronId() const { return pre_neuron_id; }
    int getPostNeuronId() const { return post_neuron_id; }
    bool isExcitatory() const { return g_type; }
    double getWeight() const { return weight; }

    void setPreNeuronId(int id) { pre_neuron_id = id; }
    void setPostNeuronId(int id) { post_neuron_id = id; }
    void setExcitatory(bool excitatory) { g_type = excitatory; }
    void setWeight(double w) { weight = w; }

    bool isEnabled() const { return enabled; }
    void enable()          { enabled = true; }
    void disable()         { enabled = false; }
};

#endif
