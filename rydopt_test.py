
import rydopt as ro
import numpy as np

import time

if __name__ == '__main__':

    start = time.time()

    gate = ro.gates.TwoQubitGate(phi=None, theta=np.pi, Vnn=float("inf"), decay=0.0)
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.const, phase_ansatz=ro.pulses.sin_crab
    )

    gate_time = time.time() - start
    print('gate time: ' ,gate_time)

    initial_params = (7.0, [0.0], [0.0, 0.0], [])

    opt_result = ro.optimization.optimize(gate, pulse_ansatz, initial_params, tol=1e-10)
    optimized_params = opt_result.params


    print('optimization time: ' ,time.time() - gate_time)
    ro.characterization.plot_pulse(pulse_ansatz, optimized_params)