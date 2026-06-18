import numpy as np
from galois import GF2
import tableau as ta
import networkx as nx

from ortools.sat.python import cp_model

gs = nx.cycle_graph(5)
n = 4
k = 1
g = 1

xlogical = ["X0*Z1"]
zlogical = ["Z0*Z3"]
stabi = ["Z0*X1*Z2", "Z1*X2*Z3", "X0*Z1*Z2*X3"]

previous_meas = ["Z2", "X3"]
previous_meas = ["X1"]
xlogi = [ta.paulistring2tableau(ele, 4) for ele in xlogical][0]
zlogi = [ta.paulistring2tableau(ele, 4) for ele in zlogical][0]
stabi = [ta.paulistring2tableau(ele, 4) for ele in stabi]
previous_meas = [ta.paulistring2tableau(ele, 4) for ele in previous_meas]

# print(x)
# previous_meas = GF2(previous_meas)
lost_qubits = ta.paulistring2tableau("Y2", 4)

T = GF2(stabi)
# T = T.row_space()


n, m = T.shape
print(n,m)



# Model
model = cp_model.CpModel()

# binary variable B_x
bx = [model.NewBoolVar(f"bx_{i}") for i in range(n)]
bxt = [sum(bx[i] * T[i, j] for i in range(n)) for j in range(m)]

mod2_terms = []
for j in range(m):
    raw = xlogi[j] + bxt[j]

    max_raw = 1 + sum(int(T[i, j]) for i in range(n))
    k = model.NewIntVar(0, max_raw // 2, f"k_{j}")

    mod2_expr = raw - 2 * k

    model.Add(mod2_expr >= 0)
    model.Add(mod2_expr <= 1)

    mod2_terms.append(mod2_expr)


num_qubits = m // 2
support = []
stab_x = mod2_terms[:num_qubits]
stab_z = mod2_terms[num_qubits:]
for q in range(num_qubits):
    sq = model.NewBoolVar(f"support_{q}")
    model.Add(sq >= stab_x[q])
    model.Add(sq >= stab_z[q])
    model.Add(sq <= stab_x[q] + stab_z[q])
    support.append(sq)


# lost qubits constraints
raw_L = sum(int(lost_qubits[j]) * mod2_terms[j] for j in range(m))
max_raw_L = sum(int(lost_qubits[j]) for j in range(m))
r = model.NewIntVar(0, max_raw_L // 2, "logical_constraint_k")
model.Add(raw_L == 2 * r)


# measurement constraints
for i, meas in enumerate(previous_meas):
    meas_x = meas[:num_qubits]
    meas_z = meas[num_qubits:]

    for q in range(num_qubits):
        raw = int(meas_z[q]) * stab_x[q] + int(meas_x[q]) * stab_z[q]

        k_comm = model.NewIntVar(0, 1, f"comm_{r}_{q}")
        model.Add(raw == 2 * k_comm)

# constraint for g-SPF

zlogi_x = zlogi[:num_qubits]
zlogi_z = zlogi[num_qubits:]
anti_terms = []
for q in range(num_qubits):
    anit_comm = int(zlogi_z[q]) * stab_x[q] + int(zlogi_x[q]) * stab_z[q]


    anti_j = model.NewIntVar(0, 1, f"anti_{j}")
    k_anti = model.NewIntVar(0, 1, f"k_anti_{j}")


    model.Add(anti_j == anit_comm - 2 * k_anti)
    anti_terms.append(anti_j)

model.Add(sum(anti_terms) <= g)

# objective function
model.Minimize(sum(support))


solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8

status = solver.Solve(model)

print("Status:", solver.StatusName(status))

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print("Minimum support:", solver.ObjectiveValue())
    print("mod2 vector:", [solver.Value(v) for v in mod2_terms])
    print("support:", [solver.Value(v) for v in support])
    a = [solver.Value(v) for v in mod2_terms]
    print(ta.tableau2paulistring(a))
    bx_sol = np.array([solver.Value(bx[i]) for i in range(n)], dtype=int)
    print(bx_sol)
    print(ta.tableau2paulistring(zlogi))