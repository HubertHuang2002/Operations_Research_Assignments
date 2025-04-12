import gurobipy as gp
from gurobipy import GRB

HC = 800 
HN = 1200 
HE = 200 
D  = [8,4,11,15,14,18]

shifts = range(1,7)

pairCost = {}

set_8HN        = {(3,5), (3,6), (4,6)}
set_8HC        = {(3,4), (4,5), (5,6)}
set_8HN_4HE    = {(1,3), (1,4), (1,5), (2,4), (2,5), (2,6)}
set_8HC_4HE    = {(1,6), (2,3)}
set_8HC_8HE    = {(1,2)}

for i in shifts:
    for j in shifts:
        if i < j:
            if (i,j) in set_8HN:
                cost_ij = 8 * HN
            elif (i,j) in set_8HC:
                cost_ij = 8 * HC
            elif (i,j) in set_8HN_4HE:
                cost_ij = 8 * HN + 4 * HE
            elif (i,j) in set_8HC_4HE:
                cost_ij = 8 * HC + 4 * HE
            elif (i,j) in set_8HC_8HE:
                cost_ij = 8 * HC + 8 * HE
            pairCost[(i,j)] = cost_ij

m = gp.Model("Police_Scheduling")

X_ij = m.addVars(pairCost.keys(), vtype=GRB.CONTINUOUS, lb=0.0 ,name="X_ij")

m.setObjective(gp.quicksum(pairCost[(i,j)] * X_ij[(i,j)]
                           for (i,j) in pairCost),
               GRB.MINIMIZE)

for i in shifts:
    m.addConstr(
        gp.quicksum(X_ij[(i,j)] for j in shifts if j>i and (i,j) in X_ij)
      + gp.quicksum(X_ij[(k,i)] for k in shifts if k<i and (k,i) in X_ij)
      >= D[i-1],
      name=f"Coverage_{i}"
    )

m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"Optimal objective value = {m.objVal}")
    print("Solution (X_{i,j}):")
    for (i,j) in pairCost:
        val = X_ij[(i,j)].x
        if abs(val) > 1e-6:
            print(f"  X[{i},{j}] = {val:.0f}")
else:
    print("No optimal solution found. Status code =", m.status)