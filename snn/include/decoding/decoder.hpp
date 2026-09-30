#ifndef RL_DECODER_HPP
#define RL_DECODER_HPP

#include <vector>

// Decodificador para convertir spikes de salida en acciones
class RLDecoder {
public:
    enum class DecodingType {
        SPIKE_COUNT,       // Contar spikes — neurona con más spikes gana
        FIRST_SPIKE,       // Neurona que disparó primero gana / latencia → valor continuo
        RATE,              // Frecuencia promedio → valor continuo
        WINNER_TAKES_ALL,  // Tasa normalizada por ventana (argmax por neurona)
        VOTING,            // Ventana dividida en sub-ventanas; winner por sub-ventana acumula votos
        GROUP_SPIKE_COUNT  // Suma spikes de grupos de neuronas por acción, gana el grupo con más
    };

private:
    DecodingType type;
    double window_duration;  // Ventana de observación en ms
    int vote_windows;        // Número de sub-ventanas para VOTING
    int neurons_per_group;   // Neuronas por grupo para GROUP_SPIKE_COUNT

public:
    RLDecoder(DecodingType type = DecodingType::WINNER_TAKES_ALL,
              double window_duration = 50.0,
              int vote_windows = 5,
              int neurons_per_group = 20);

    // Decodificar a acción discreta (e.g., arriba, abajo, izquierda, derecha)
    int decodeDiscreteAction(const std::vector<std::vector<double>>& output_spikes) const;

    // Decodificar a acción discreta agrupando neuronas (GROUP_SPIKE_COUNT)
    // neurons_per_group neuronas por acción; gana el grupo con más spikes totales
    int decodeGroupedAction(
        const std::vector<std::vector<double>>& output_spikes,
        int neurons_per_group) const;

    // Decodificar a acción continua (e.g., torque, velocidad)
    double decodeContinuousAction(const std::vector<double>& spike_times) const;

    // Decodificar acción continua normalizada a [-1, 1].
    // max_spikes: número de spikes que corresponde a la acción máxima (+1).
    double decodeContinuousAction(const std::vector<double>& spike_times, double max_spikes) const;

    // Decodificar múltiples outputs continuos
    std::vector<double> decodeContinuousActions(
        const std::vector<std::vector<double>>& output_spikes) const;

    void setType(DecodingType t) { type = t; }
    void setWindowDuration(double duration) { window_duration = duration; }
    void setVoteWindows(int n) { vote_windows = n; }
    void setNeuronsPerGroup(int n) { neurons_per_group = n; }
    int getNeuronsPerGroup() const { return neurons_per_group; }
};

#endif