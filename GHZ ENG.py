"""
Program for a small quantum algorithm to prepare three initial qubits in the Greenberger–Horne–Zeilinger (GHZ) state.
Two different Hadamard matrices are used — an ideal one and a physical one.
"""

import numpy as np
import sys
# Matrices of quantum gates
I = np.eye(2)
H_ideal = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

"""
Quantum circuit:
0: ───H───@───────M───
          │       │
1: ───────X───@───M───
              │   │
2: ───────────X───M───

H - Hadamard gate
@--X - CNOT
M - Measuring device (observer)
"""
def tensor(*args):
    """tensor product of an arbitrary number of matrices"""
    r = args[0]
    for m in args[1:]:
        r = np.kron(r, m)
    return r

def ghz_circuit(H):
    """algorithm for obtaining the GHZ state"""
    step1 = tensor(H, I, I)           # Hadamard at q0
    step2 = tensor(CNOT, I)           # CNOT(q0, q1)
    step3 = tensor(I, CNOT)           # CNOT(q1, q2)
    final_matrix = step3 @ step2 @ step1    # ordinary matrix multiplication
    print("\nThe final matrix of the algorithm:")
    print(f"{np.array2string(final_matrix, precision=4)}")
    #print(final_matrix)
    return final_matrix

def run(H, state):
    """passes the state of the input quanta through the chain to obtain the GHZ state"""
    psi = np.zeros(8)
    psi[int(state, 2)] = 1
    ret = ghz_circuit(H) @ psi
    print("\nResult:")
    print(f"{np.array2string(ret, precision=4)}")
    #print(ret)
    return ret

def read_data(filename):
    """parsing the data from the physical gate"""
    data = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            data.append(int(line))
    return data

def build_physical_H(data):
    """building math model of the matrix based on the measurements"""
    counts = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}} # counting passes
    for i in range(len(data)):
        if i % 4 < 2:
            counts[0][i%2] = data[i]
        else:   counts[1][i%2] = data[i]

    # Transition probability (output conditioned on input)
    P = np.zeros((2, 2))
    err = np.zeros((2, 2))
    for j in range(2):
        n = counts[j][0] + counts[j][1]
        for i in range(2):
            P[i, j] = counts[j][i] / n # probability of such an outcome
            err[i, j] = np.sqrt(P[i,j] * (1 - P[i,j]) / n) # the standard error of such an outcome

    # The transition probability is the square of the amplitude (H_ij)**2 = |P(i|j)|;
    # we take the signs from the ideal matrix (deviations are small)
    H = np.sqrt(P) * np.sign(H_ideal)
    # This is error propagation. If H = √P, then by the formula δH = δP / (2√P).
    # We take the error of the probability and recalculate it into the error of the matrix element.
    err_H = np.where(P > 0, err / (2 * np.sqrt(P)), 0)
    return H, err_H


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "Measurements.txt"
    print('File is not chosen — using "Measurements.txt"\n')

data = read_data(filename)
H_phys, H_err = build_physical_H(data)

print("Ideal Hadamard gate:")
print(f"{np.array2string(H_ideal, precision=4)}")
print("Physical Hadamard gate:")
print(f"{np.array2string(H_phys, precision=4)}")
print("Errors in determining the elements of Hadamard gate matrix:")
print(f"{np.array2string(H_err, precision=4)}")

state = input("\nThe initial state of the qubits without spaces").strip()

print("\n\nWorking with ideal Hadamard gate:")
psi_ideal = run(H_ideal, state)
print("\n\nWorking with physical Hadamard gate:")
psi_phys = run(H_phys, state)

F = abs(np.dot(psi_ideal, psi_phys))**2
print(f"\n\nDeviation: {(1-F)*100:.4f}%")
