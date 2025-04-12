import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_production_planning(excel_file):
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

    model = gp.Model("ProductionPlanning")

    x = model.addVars(T, vtype=GRB.CONTINUOUS, name="x")
    w = model.addVars(T, vtype=GRB.CONTINUOUS, name="w")
    y = model.addVars(T+1, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(T, vtype=GRB.BINARY, name="z")
    print(y)

    model.addConstr(y[0] == 0, "initial_inventory")

    for t in range(T):
        model.addConstr(y[t] + x[t] - w[t] == y[t+1])
        model.addConstr(x[t] <= k[t], f"cap_{t+1}")
        model.addConstr(w[t] <= d[t], f"dmd_{t+1}")
        model.addConstr(x[t] <= k[t]*z[t], f"setup_{t+1}")

    revenue = gp.quicksum(p[t]*w[t] for t in range(T))
    production_cost = gp.quicksum(c[t]*x[t] for t in range(T))
    holding_cost = H * gp.quicksum(y[t] for t in range(T))
    setup_cost = S * gp.quicksum(z[t] for t in range(T))

    model.setObjective(revenue - production_cost - holding_cost - setup_cost, GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective = {model.objVal:.2f}")
        for t in range(T):
            print(f"Period {t+1}: x={x[t].X:.1f}, w={w[t].X:.1f}, y={y[t+1].X:.1f}, z={z[t].X:.0f}")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    excel_filename = "OR113-2_hw02_data.xlsx"
    solve_production_planning(excel_filename)
