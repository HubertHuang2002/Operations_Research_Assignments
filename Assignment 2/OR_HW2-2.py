import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_production_planning_heuristic(excel_file):
    df_basic = pd.read_excel(
        excel_file, sheet_name=0,
        usecols=[0,1],
        nrows=3,
        header=None,
    )
    T = int(df_basic.iloc[0,1])  
    H = float(df_basic.iloc[1,1])
    S = float(df_basic.iloc[2,1])

    df_params = pd.read_excel(
        excel_file, sheet_name=0,
        skiprows=5,
        nrows=T,
        header=None, 
    )

    df_params.columns = ["t", "C", "P", "K", "D"]
    df_params = df_params.apply(pd.to_numeric, errors="coerce")
    c = df_params["C"].tolist()  
    p = df_params["P"].tolist()  
    k = df_params["K"].tolist() 
    d = df_params["D"].tolist()  

    modelLR = gp.Model("ProductionPlanning_LR")

    xLR = modelLR.addVars(T, vtype=GRB.CONTINUOUS, name="xLR")
    wLR = modelLR.addVars(T, vtype=GRB.CONTINUOUS, name="wLR")
    yLR = modelLR.addVars(T+1, vtype=GRB.CONTINUOUS, name="yLR")
    zLR = modelLR.addVars(T, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="zLR")

    M = [10000]*T

    modelLR.addConstr(yLR[0] == 0, "initial_inventory_LR")
    for t in range(T):
        modelLR.addConstr(yLR[t] + xLR[t] - wLR[t] == yLR[t+1], f"balance_LR_{t+1}")
        modelLR.addConstr(xLR[t] <= k[t], f"cap_LR_{t+1}")
        modelLR.addConstr(wLR[t] <= d[t], f"dmd_LR_{t+1}")
        modelLR.addConstr(xLR[t] <= M[t]*zLR[t], f"setup_LR_{t+1}")

    revenueLR = gp.quicksum(p[t]*wLR[t] for t in range(T))
    productionLR = gp.quicksum(c[t]*xLR[t] for t in range(T))
    holdingLR = H * gp.quicksum(yLR[t] for t in range(T))
    setupLR = S * gp.quicksum(zLR[t] for t in range(T))

    modelLR.setObjective(revenueLR - productionLR - holdingLR - setupLR, GRB.MAXIMIZE)
    modelLR.optimize()

    print("\n--- [Step 1] Linear Relaxation Solution (z in [0,1]) ---")
    if modelLR.status == GRB.OPTIMAL:
        print(f"LR Objective = {modelLR.objVal:.2f}")
        zLR_values = []
        for t in range(T):
            zLR_values.append(zLR[t].X)
            print(f"Period {t+1}: zLR={zLR[t].X:.2f}")
    else:
        print("No optimal solution found in LR step.")
        return

    zIP = []
    for t in range(T):
        if zLR[t].X >= 0.5:
            zIP.append(1.0)
        else:
            zIP.append(0.0)

    print("\n--- [Step 2] Converting z^LR to z^IP ---")
    print("z^IP =", zIP)

    modelHeuristic = gp.Model("ProductionPlanning_Heuristic")

    xH = modelHeuristic.addVars(T, vtype=GRB.CONTINUOUS, name="xH")
    wH = modelHeuristic.addVars(T, vtype=GRB.CONTINUOUS, name="wH")
    yH = modelHeuristic.addVars(T+1, vtype=GRB.CONTINUOUS, name="yH")

    modelHeuristic.addConstr(yH[0] == 0, "initial_inventory_H")

    for t in range(T):
        modelHeuristic.addConstr(yH[t] + xH[t] - wH[t] == yH[t+1], f"balance_H_{t+1}")
        modelHeuristic.addConstr(xH[t] <= k[t], f"cap_H_{t+1}")
        modelHeuristic.addConstr(wH[t] <= d[t], f"dmd_H_{t+1}")
        modelHeuristic.addConstr(xH[t] <= M[t]*zIP[t], f"setup_H_{t+1}")

    revenueH = gp.quicksum(p[t]*wH[t] for t in range(T))
    prodCostH = gp.quicksum(c[t]*xH[t] for t in range(T))
    holdCostH = H * gp.quicksum(yH[t] for t in range(1,T+1))
    setupCostH = S * gp.quicksum(zIP[t] for t in range(T))
    modelHeuristic.setObjective(revenueH - prodCostH - holdCostH - setupCostH, GRB.MAXIMIZE)

    modelHeuristic.optimize()

    print("\n--- [Step 3] Heuristic Final Solution ---")
    if modelHeuristic.status == GRB.OPTIMAL:
        objHeuristic = modelHeuristic.objVal
        print(f"Heuristic Objective = {objHeuristic:.2f}")
        for t in range(T):
            print(f"Period {t+1}: x={xH[t].X:.1f}, w={wH[t].X:.1f}, y={yH[t+1].X:.1f}, z^IP={int(zIP[t])}")
    else:
        print("No optimal solution found in the heuristic phase.")
    return

if __name__ == "__main__":
    excel_filename = "OR113-2_hw02_data.xlsx"
    solve_production_planning_heuristic(excel_filename)
