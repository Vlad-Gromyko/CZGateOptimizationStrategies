import rydopt as ro
#import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rydopt.types import HamiltonianFunction
import copy
import time
import os
os.environ["JAX_PLATFORMS"] = "cuda"

# 2. Узнать платформу первого устройства (CPU, GPU, TPU)
print("Платформа по умолчанию:", jax.devices()[0].platform)

# 3. Более подробная информация о локальных устройствах (особенно полезно для multi-GPU)
print("Локальные устройства:", jax.local_devices())


# %%
class CZGateThreePhotonLevine:

    def __init__(self, Omega2: float, Omega3: float, Vnn: float, DecayP: float, DecayS: float, DecayR: float):
        self._Omega2 = Omega2
        self._Omega3 = Omega3
        self._Vnn = Vnn
        self._DecayP = DecayP
        self._DecayS = DecayS
        self._DecayR = DecayR

    def initial_basis_states(self) -> tuple[jnp.ndarray, ...]:
        return jnp.array([1, 0, 0, 0], dtype=complex), jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)

    def hamiltonian_functions_for_basis_states(self) -> tuple[HamiltonianFunction, ...]:
        def hamiltonian1(Delta: float, Xi: float, Omega: float) -> jnp.ndarray:
            """single-atom excitation with states 01,0p,0s,0r"""

            DecayP = self._DecayP
            DecayS = self._DecayS
            DecayR = self._DecayR
            Omega2 = self._Omega2
            Omega3 = self._Omega3

            Omega1 = Omega2 * (Omega * jnp.exp(-1j * Xi)) / Omega3
            Omega1C = Omega2 * (Omega * jnp.exp(1j * Xi)) / Omega3
            return jnp.array(
                [
                    [0, Omega1 * 0.5, 0, 0],
                    [Omega1C * 0.5, -0.5 * 1j * DecayP, Omega2 * 0.5, 0],
                    [0, Omega2 * 0.5, -0.5 * 1j * DecayS, Omega3 * 0.5],
                    [0, 0, Omega3 * 0.5, Delta - 0.5 * 1j * DecayR]

                ]
            )

        def hamiltonian2(Delta: float, Xi: float, Omega: float) -> jnp.ndarray:
            """
            11 - 1p+p1 - 1s+s1, 1r+r1,
            11 - 1p+p1 - pp - ps+sp - ss - sr+rs - rr
            11 - 1p+p1 - 1r+r1 - pr+rp - sr+rs - rr
            two-atom excitation with states 11,1p+p1,1s+s1, 1r+r1, pp, ps+sp, pr+rp, ss, sr+rs, rr
            """
            DecayP = self._DecayP
            DecayS = self._DecayS
            DecayR = self._DecayR
            Omega2 = self._Omega2
            Omega3 = self._Omega3
            Vnn = self._Vnn

            Omega1 = Omega2 * (Omega * jnp.exp(-1j * Xi)) / Omega3
            Omega1C = Omega2 * (Omega * jnp.exp(1j * Xi)) / Omega3

            return jnp.array(
                [
                    [0, Omega1 * 0.5 * jnp.sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0],  # 11
                    [Omega1C * 0.5 * jnp.sqrt(2), -0.5 * 1j * DecayP, Omega2 * 0.5, 0, Omega1 * 0.5 * jnp.sqrt(2), 0, 0,
                     0, 0, 0],  # 1p+p1
                    [0, Omega2 * 0.5, -0.5 * 1j * DecayS, Omega3 * 0.5, 0, 0.5 * Omega1, 0, 0, 0, 0],  # 1s+s1
                    [0, 0, Omega3 * 0.5, Delta - 0.5 * 1j * DecayR, 0, 0, 0.5 * Omega1, 0, 0, 0],  # 1r+r1
                    [0, Omega1C * 0.5 * jnp.sqrt(2), 0, 0, -1j * DecayP, Omega2 * 0.5 * jnp.sqrt(2), 0, 0, 0, 0],  # pp
                    [0, 0, 0.5 * Omega1C, 0, Omega2 * 0.5 * jnp.sqrt(2), -0.5 * 1j * (DecayP + DecayS), Omega3 * 0.5,
                     Omega2 * 0.5 * jnp.sqrt(2), 0, 0],  # ps+sp
                    [0, 0, 0, 0.5 * Omega1C, 0, Omega3 * 0.5, Delta - 0.5 * 1j * (DecayP + DecayR), 0, Omega2 * 0.5, 0],
                    # pr+rp
                    [0, 0, 0, 0, 0, Omega2 * 0.5 * jnp.sqrt(2), 0, -1j * DecayS, Omega3 * 0.5 * jnp.sqrt(2), 0],  # ss
                    [0, 0, 0, 0, 0, 0, Omega2 * 0.5, Omega3 * 0.5 * jnp.sqrt(2), Delta - 1j * 0.5 * (DecayS + DecayR),
                     Omega3 * 0.5 * jnp.sqrt(2)],  # sr+rs
                    [0, 0, 0, 0, 0, 0, 0, 0, Omega3 * 0.5 * jnp.sqrt(2), 2 * Delta + Vnn - 1j * DecayR]  # rr
                ]
            )

        return hamiltonian1, hamiltonian2

    def process_fidelity(
            self, final_basis_states: tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray:
        # Obtained diagonal gate matrix
        obtained_gate = jnp.array(
            [
                1,
                final_basis_states[0][0],
                final_basis_states[0][0],
                final_basis_states[1][0],
            ]
        )

        # Targeted diagonal gate matrix
        p = jnp.angle(obtained_gate[1])
        t = jnp.pi

        targeted_gate = jnp.stack(
            [
                1,
                jnp.exp(1j * p),
                jnp.exp(1j * p),
                jnp.exp(1j * (2 * p + t)),
            ]
        )
        return jnp.abs(jnp.vdot(targeted_gate, obtained_gate)) ** 2 / len(targeted_gate) ** 2


# %%
def split(params):
    # Штука шоб делить
    vector = []
    sizes = []
    for item in params:
        if isinstance(item, (int, float)):
            vector.append(item)
            sizes.append(1)
        elif isinstance(item, (list, jnp.ndarray)):
            vector.extend(item)
            sizes.append(len(item))
    return jnp.array(vector), sizes


def assemble(vector, sizes):
    # Штука шоб собирать
    params = []
    pos = 0
    for size in sizes:
        params.append(list(vector[pos:pos + size]))
        pos += size
    return tuple(params)


# %%
params = (15.560089695727132,
          [-0.66202403],
          [0.51990003, -0.2527502, 0.8859972, -1.17347235, -0.08802893,
           -0.327713, -1.00445164, 0.10583959, 1.20175343, 0.62878523,
           0.65908966, -0.25679209, -0.56651597, 1.10736928, -0.32141791,
           0.14352779, -0.2186016],
          [1])
# params_new = assemble(vector_val, structure_val)
vector_val, structure_val = split(params)

params_new = assemble(vector_val, structure_val)

params_new
# при применении assemble тип меняется, наверное, в этом проблема

# %%
lifetime80 = 260.3716142904322
lifetime5p = 26e-3
lifetime7s = 88e-3

Omega2 = 2 * jnp.pi * 5000
Omega3 = jnp.sqrt(Omega2 * 2 * 1 * jnp.pi)
gate = CZGateThreePhotonLevine(Omega2, Omega3, 10000, 0 / lifetime5p / 10, 0 / lifetime7s / 10, 0 / lifetime80 / 10)

vector_val, structure_val = split(params)


def loss(vector, structure):
    start = time.time()
    lifetime80 = 260.3716142904322
    lifetime5p = 26e-3
    lifetime7s = 88e-3

    Omega2 = 2 * jnp.pi * 5000
    Omega3 = jnp.sqrt(Omega2 * 2 * 1 * jnp.pi)
    gate = CZGateThreePhotonLevine(Omega2, Omega3, 10000, 0 / lifetime5p / 10, 0 / lifetime7s / 10, 0 / lifetime80 / 10)
    params = assemble(vector, structure)
    params_shift = copy.deepcopy(params)

    # Изменяем Omega (параметр Раби)
    params_shift[3][0] = params[3][0] * 1.005

    # Извлекаем значения из списков
    duration = params[0][0]                  # длительность (число)
    duration_shift = params_shift[0][0]

    # Параметры детьюнинга (константа) – Xi, массив из одного элемента
    detuning_params = jnp.array(params[1])
    detuning_params_shift = jnp.array(params_shift[1])

    # Параметры фазы (lin_sin_cos_crab) – коэффициенты, массив из 17 элементов
    phase_params = jnp.array(params[2])
    phase_params_shift = jnp.array(params_shift[2])

    # Параметры Раби (константа) – Omega, массив из одного элемента
    rabi_params = jnp.array(params[3])
    rabi_params_shift = jnp.array(params_shift[3])

    # Кортеж из 4 элементов
    params_jax = (duration, detuning_params, phase_params, rabi_params)
    params_shift_jax = (duration_shift, detuning_params_shift, phase_params_shift, rabi_params_shift)

    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const,
        phase_ansatz=ro.pulses.lin_sin_cos_crab,
        rabi_ansatz=ro.pulses.const
    )

    time_evolved_basis_states = ro.simulation.evolve(gate, pulse_ansatz, params_jax)
    time_evolved_basis_states_shift = ro.simulation.evolve(gate, pulse_ansatz, params_shift_jax)
    print('\n')
    print('TIME',time.time() - start )
    print('\n')

    return ((1 - gate.process_fidelity(time_evolved_basis_states)) + \
           (1 - gate.process_fidelity(time_evolved_basis_states_shift))).item()


# %%
params = (15.560089695727132,
          [-0.66202403],
          [0.51990003, -0.2527502, 0.8859972, -1.17347235, -0.08802893,
           -0.327713, -1.00445164, 0.10583959, 1.20175343, 0.62878523,
           0.65908966, -0.25679209, -0.56651597, 1.10736928, -0.32141791,
           0.14352779, -0.2186016],
          [1])
vector_val, structure_val = split(params)
if __name__ == '__main__':
    print(loss(vector_val, structure_val))