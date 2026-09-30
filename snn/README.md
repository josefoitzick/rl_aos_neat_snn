# snn — simulador de redes de espigas (Izhikevich) en C++

Basado en **snn-simulator** de *dilgeto* (https://github.com/dilgeto/snn-simulator, commit `0633a1c`,
2026-09-23), incorporado a este repo con autorización de su autor. Licencia original: GPL v3 (`LICENSE`).

Desde su incorporación se modifica como código propio de la tesis: se agregarán pesos por sinapsis,
sembrado reproducible y los fixes listados en [`../docs/snn-simulator.md`](../docs/snn-simulator.md),
donde también está el análisis completo del modelo, la API y el estado del código.

Bibliotecas CMake: `snn_core` (neurona y red), `snn_encoding` (encoders), `snn_engine` (simulador y decoder).
