import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

n = 10
m = 3

s = [10, 5, 5]  
q = [100, 80, 40, 120, 110, 50, 90, 200, 170, 150]
t = [1, 1, 1, 2, 3, 1, 1, 1, 3, 3]
d = [30, 10, 20, 20, 30, 10, 30, 40, 50, 40]

p = [[0]*n for _ in range(m)]
for i in range(m):
    for j in range(n):
        p[i][j] = q[j] / s[i]

M = sum(max(p[i][j] for i in range(m)) for j in range(n))

model = gp.Model("Scheduling")

X = model.addVars(n, m, vtype=GRB.BINARY, name="X")
W = model.addVars(n, n, m, vtype=GRB.BINARY, name="W")
C = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="C")
T = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="T")

model.setObjective(gp.quicksum(t[j] * T[j] for j in range(n)), GRB.MINIMIZE)

for j in range(n):
    model.addConstr(gp.quicksum(X[j,k] for k in range(m)) == 1,
                    name=f"AssignEachJobOnce_{j}")

for j in range(n):
    model.addConstr(T[j] >= C[j] - d[j], name=f"TjDeadline_{j}")

for j in range(n):
    model.addConstr(
        C[j] >= gp.quicksum(X[j,k] * p[k][j] for k in range(m)),
        name=f"CompletionAtLeastProcTime_{j}"
    )

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        for k in range(m):
            model.addConstr(
                C[i] + p[k][j] - C[j] <= M*(1 - W[i,j,k]),
                name=f"NoOverlap_k{k}_i{i}_j{j}"
            )

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        for k in range(m):
            model.addConstr(
                W[i,j,k] + W[j,i,k] + 1 >= X[i,k] + X[j,k],
                name=f"WsumLower_k{k}_i{i}_j{j}"
            )

model.setParam('OutputFlag', 1)
model.optimize()

print(f"Optimal objective value (Total Weighted Tardiness) = {model.ObjVal:.2f}")

for j in range(n):
    assigned_machine = None
    for k in range(m):
        if X[j, k].X > 0.5:
            assigned_machine = k
            break
    print(f"Job {j} -> Machine {assigned_machine}, "
          f"Completion Time C[{j}] = {C[j].X:.2f}, "
          f"Tardiness T[{j}] = {T[j].X:.2f}")

print("\n--- Schedule (order) on each machine ---")
for k in range(m):
    jobs_on_k = [j for j in range(n) if X[j, k].X > 0.5]
    jobs_on_k.sort(key=lambda j: C[j].X)
    
    print(f"Machine {k + 1}")
    for j in jobs_on_k:
        print(f"   Job {j + 1}: Completion={C[j].X:.2f}, Tardiness={T[j].X:.2f}")

machine_schedules = [[] for _ in range(m)]
max_finish_time = 0.0

for j in range(n):
    assigned_machine = None
    for k in range(m):
        if X[j, k].X > 0.5:
            assigned_machine = k
            break
    if assigned_machine is None:
        continue

    finish_time = C[j].X
    start_time = finish_time - p[assigned_machine][j]
    max_finish_time = max(max_finish_time, finish_time)

    machine_schedules[assigned_machine].append((start_time, finish_time, j))

fig, ax = plt.subplots()

for k in range(m):
    machine_schedules[k].sort(key=lambda x: x[0])
    for (start, finish, job_id) in machine_schedules[k]:
        width = finish - start
        ax.add_patch(plt.Rectangle(
            (start, k),
            width,
            0.8,
            fill=True,
            edgecolor='black',
            linewidth=1         
        ))
        ax.text(
            start + width/2,
            k + 0.4,
            f"J{job_id+1}",
            ha='center', va='center'
        )

ax.set_xlim(0, max_finish_time * 1.05)
ax.set_ylim(-0.1, m + 0.5)
ax.set_yticks([k + 0.4 for k in range(m)])
ax.set_yticklabels([f"Machine {k+1}" for k in range(m)])

ax.set_xlabel("Time")
ax.set_ylabel("Machine")
plt.title("Gantt Chart for Scheduling Result")
plt.show()
