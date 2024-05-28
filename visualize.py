# %%
import os
import json
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from typing import List, Tuple
from utils import Timer, scatter, boxplot, plot_df, describe, plot_network, scatter_categorical
from network import Network
from assignment import UserEquilibrium
from alns_pareto import ALNS

# %%
# output directories
case_name = "macmini_acc1_minw1_addmall" ##this is the main result that has been reported in the paper; best_solution_idx=140
# case_name = "R1_pseudo_holiday_1" 
case_name = "R1_macmini_new" 
# case_name = "imacpro_acc5_rea60"
# case_name = "macmini_acc1_reac60_minw1"
# case_name = "macbookair_acc1_reac50_minw1"
# case_name = "macbookair_acc1_reac50_minw1_addmall"
out_dir = os.path.join("ALNS_" + case_name)
config_dir = os.path.join(out_dir, 'config/')
result_dir = os.path.join(out_dir, 'result/')

# %%
with open(config_dir + "config.json") as f:
    param_ = json.load(f)
    
# %%
net = Network()
model = UserEquilibrium(net, "base", fldr_path=config_dir, accuracy=param_["accuracy"])
alns = ALNS(model, maxChanges=param_["maxChanges"], maxSearch=param_["maxSearch"],
            maxZones=param_["maxZones"], nWarmup=param_["nWarmup"], nReturn=param_["nReturn"], 
            min_weight=param_["min_weight"], max_del_weight=param_["max_del_weight"],
            react_factor=param_["react_factor"],
            init_temp=param_["init_temp"])    
alns.timer = Timer()

# %%
replaces = {
    "expand_zone": "EX", "shrink_zone": "SH", "create_new_zone": "NW", "delete_zone": "DL",
    "random": "r", "max_size": "max", "min_size": "min",
    "min_cap": "mincap", "max_walk": "maxwalk", "max_mixed_traffic": "mix",
    "max_cap": "maxcap", "min_walk": "minwalk", "": ""
}
operator_names = []
for op_idx in alns.op_idxs:
    operator = alns.operators[op_idx]
    for opzone_idx in alns.opzone_idxs[op_idx]:
        op_zone = alns.zone_operators[op_idx][opzone_idx]
        for opst_idx in alns.opst_idxs[op_idx]:
            op_st = alns.st_operators[op_idx][opst_idx]
            operator_names.append(f"{replaces[operator]}_{replaces[op_zone]}_{replaces[op_st]}")

# %%
operator_names = []
operator_names += ["EX$_{" + str(i) + "}$" for i in range(1,13)]
operator_names += ["SH$_{" + str(i) + "}$" for i in range(1,10)]
operator_names += ["NW$_{" + str(i) + "}$" for i in range(1,5)]
operator_names += ["DL$_{" + str(i) + "}$" for i in range(1,4)]

# %%
df_success = pd.read_csv(result_dir + "success.csv")
df_frontier = pd.read_csv(result_dir + "frontier.csv")
df_obj = pd.read_csv(result_dir + "objval.csv")
df_y = pd.read_csv(result_dir + "y.csv")
df_temp = pd.read_csv(result_dir + "temparature.csv")
df_temp.columns = alns.obj_keys
df_time = pd.read_csv(result_dir + "times.csv")
df_p = pd.read_csv(result_dir + "p.csv")
# major operator categories
df_op_probs = df_p.copy()
df_op_probs.columns = np.arange(len(df_op_probs.columns))
df_op_probs["EX"] = df_op_probs[alns.p_op_idxs[0]].sum(axis=1)
df_op_probs["SH"] = df_op_probs[alns.p_op_idxs[1]].sum(axis=1)
df_op_probs["NW"] = df_op_probs[alns.p_op_idxs[2]].sum(axis=1)
df_op_probs["DL"] = df_op_probs[alns.p_op_idxs[3]].sum(axis=1)
df_op_probs = df_op_probs[["EX", "SH", "NW", "DL"]].T
# detailed operator categories
df_p.columns = operator_names
df_p = df_p.T

# objective values
# frontiers
frontiers = df_frontier["0"].values
z_frontiers = df_obj.loc[frontiers].values
z_frontiers[:,1] = -z_frontiers[:,1]
# take nonzero
df_obj = df_obj.iloc[1:]
df_obj = df_obj[df_obj.sum(axis=1) > 0]
z = np.array(df_obj)
z[:,1] = -z[:,1] # because pedestrian comfort should be maximized

# decision variables
y = np.array(df_y)

# metrics for operators
df_success.columns = ["n_trials", "n_accepted", "n_frontiers"]
df_success["success_rate"] = df_success["n_accepted"] / df_success["n_trials"]
df_success["frontier_rate"] = df_success["n_frontiers"] / df_success["n_trials"]
df_success.index = operator_names

print(f"no. frontiers = {len(df_frontier)}")

# %%
df_success.to_csv(result_dir + "success_added.csv", index=True)

# %%
### Table
# best for each of z0, z1 and z2
z0_idx = df_obj.index[np.argmin(z[:,0])]
z1_idx = df_obj.index[np.argmax(z[:,1])]
z2_idx = df_obj.index[np.argmin(z[:,2])]

### Plot
bests = df_obj.loc[[z0_idx, z1_idx, z2_idx]]
print(bests)

# %%
## pareto: z0 vs z1
scatter(x = z_frontiers[:,0], 
        y = z_frontiers[:,1], 
        alpha = z_frontiers[:,2],
        focus = z_frontiers[0,:2],
        fcolor = "red",
        cmap = "binary",
        c_rev = "",
        xlabel = "Vehiclar travel time",
        ylabel = "Pedestrian comfort",
        flabel = "Current network",
        figsize = (8,6),
        grid = False,
        colorbar = True,
        size=30,
        save=True,
        file_path=result_dir+"pareto_01_curr.pdf"
        )

# %%
## pareto: z0 vs z2
scatter(x = z_frontiers[:,0], 
        y = z_frontiers[:,2], 
        alpha = z_frontiers[:,1],
        focus = z_frontiers[0,[0,2]],
        fcolor = "red",
        cmap = "binary",
        c_rev = "",
        xlabel = "Vehiclar travel time",
        ylabel = "Areas of pedestrianized streets",
        flabel = "Current network",
        figsize = (8,6),
        grid = False,
        colorbar = True,
        size=30,
        save=True,
        file_path=result_dir+"pareto_02_curr.pdf"
        )

# %%
## pareto: z1 vs z2
scatter(x = z_frontiers[:,2], 
        y = z_frontiers[:,1], 
        alpha = z_frontiers[:,0],
        focus = z_frontiers[0,[2,1]],
        fcolor = "red",
        cmap = "binary",
        c_rev = "",
        xlabel = "Areas of pedestrianized streets",
        ylabel = "Pedestrian comfort",
        flabel = "Current network",
        figsize = (8,6),
        grid = False,
        colorbar = True,
        size=30,
        save=True,
        file_path=result_dir+"pareto_12_curr.pdf"
        )

# %%
### Box plot of operator selection probabilities
boxplot(df_op_probs.T, rotation=0, figsize=(6,4))
plot_df(df_op_probs.T, ["EX", "SH", "NW", "DL"], figsize=(8,4),
        xlabel="Iteration", ylabel="Operator selection probability",
        save=True, 
        file_path=result_dir+"op_prob_major.pdf")

# %%
### Box plot of details
M = alns.maxChanges
boxplot(df_p.T[:int(M/2)], rotation=90, figsize=(8,4),
        xlabel="Operator", ylabel="Selection probability",
        ylim=(0, 0.12),
        save=True, 
        file_path=result_dir+"op_prob_firsthalf.pdf")
boxplot(df_p.T[int(M/2):], rotation=90, figsize=(8,4),
        xlabel="Operator", ylabel="Selection probability",
        ylim=(0, 0.12),
        save=True, 
        file_path=result_dir+"op_prob_secondhalf.pdf")

# %%
### Plot temperature
plot_df(df_temp, columns=alns.obj_keys, ylog=True, xlabel="Iteration", ylabel="Temperature")

# %%
### Plot temperature
df_time["CPUtime"] = np.cumsum(df_time["0"].values)
plot_df(df_time, columns=["CPUtime"], ylog=True, xlabel="Iteration", ylabel="CPU time [s]")

# %%
def get_metric(y, vmax_p=0.95):
    alns.n = 0
    alns.m = 0
    alns.reset_design(y=y)

    keys_ = ["traffic_car", "ped_traffic_car", "congestion", "ped_comfort_car", "travel_time_car"]
    metrics = {key_: alns.__dict__[key_] for key_ in keys_}
    metric_cand = {key_: np.zeros(len(alns.candidates), dtype=np.float64) for key_ in keys_}
    for key_ in keys_:
        for cand_id, ilinks in alns.candidates.items():
            metric_cand[key_][cand_id] = metrics[key_][ilinks].sum()
        metric_cand[key_][np.where(metric_cand[key_] > 0.9e+8)[0]] = 0.
        metric_cand[key_] = np.clip(metric_cand[key_], None, np.quantile(metric_cand[key_], vmax_p))
    
    return metric_cand

# %%
# number of times a street is used in pareto frontier solutions
n_used_frontier = df_y.sum(axis=0).values
print(describe(n_used_frontier))

plot_network(df_y.loc[0], n_used_frontier,
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greens",
             vmin_p=0, plot_node=False, fcolor="green",
             save=False, file_path=result_dir+"network_pareto.pdf") #demand=net.demand, 

# %%
# raw network
y_orig = df_y.loc[0].values
print("Current network:")
print(f"obj val = {z_frontiers[0]}")
plot_network(y_orig, np.ones_like(y_orig),
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="binary",
             vmin_p=0, plot_node=False,
             demand=net.demand.query("origin == 374"),
             parking=net.parking_links,
             y_width=1, v_width=1, f_width=1) #

# %%
vmax_p = 1.
metric_cand_original = get_metric(y_orig, vmax_p)
vmax_traffic = metric_cand_original["traffic_car"].max()
vmax_ped_traffic = metric_cand_original["ped_traffic_car"].max()
vmax_ped_comfort = metric_cand_original["ped_comfort_car"].max()
print(f"The maximum value to visualize based on {vmax_p} quantile. \n \
      Car traffic: {vmax_traffic}; \n \
      Ped traffic: {vmax_ped_traffic}; \n \
      Ped comfort: {vmax_ped_comfort}. \
      ")

# %%
plot_network(y_orig, metric_cand_original["traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Blues",
             vmax_p=vmax_traffic)


plot_network(y_orig, metric_cand_original["ped_traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Reds",
             vmax_p=vmax_ped_traffic)

plot_network(y_orig, metric_cand_original["ped_comfort_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greens",
             vmax_p=vmax_ped_comfort)


# %%
## maximum number
idx_ = df_y[df_y.sum(axis=1) == df_y.sum(axis=1).max()].index.values[0]
y_max = df_y.loc[idx_].values
print(f"max no. streets: {y_max.sum()}")
print(f"obj val = {z_frontiers[idx_]}")
metric_cand_max = get_metric(y_max)

# %%
plot_network(y_max, metric_cand_max["traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Blues",
             vmax_p=vmax_traffic, fcolor="red",
             save=True, file_path=result_dir+"designMax_car_red.pdf")

plot_network(y_max, metric_cand_max["ped_traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Reds",
             vmax_p=vmax_ped_traffic, fcolor="red",
             save=True, file_path=result_dir+"designMax_ped_red.pdf")

plot_network(y_max, metric_cand_max["ped_comfort_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greens",
             vmax_p=vmax_ped_comfort, fcolor="green")


# %%
i_min = np.where(z_frontiers[:,0] == z_frontiers[:,0].min())[0][0]
if z_frontiers[i_min,2] < alns.eps:
    print("same as current network")
else:
    y_mintt = df_y.iloc[i_min].values
    metric_cand_mintt = get_metric(y_mintt)
    print(f"obj val = {z_frontiers[idx_]}")

    plot_network(y_mintt, metric_cand_mintt["traffic_car"], 
                st_nodes=alns.cand_nodes, node_data=net.car_nodes,
                figsize=(7,6), size=4, cmap="Blues",
                vmax_p=vmax_traffic)

    plot_network(y_mintt, metric_cand_mintt["ped_traffic_car"], 
                st_nodes=alns.cand_nodes, node_data=net.car_nodes,
                figsize=(7,6), size=4, cmap="Reds",
                vmax_p=vmax_ped_traffic)
    
    plot_network(y_max, metric_cand_mintt["ped_comfort_car"], 
                st_nodes=alns.cand_nodes, node_data=net.car_nodes,
                figsize=(7,6), size=4, cmap="Greens",
                vmax_p=vmax_ped_comfort)

# %%
## other designs
# i_acctt = np.where(z_frontiers[:,0] < 0.855e+7)[0] # for best
# i_ped5 = np.where(z_frontiers[:,1] > 8.65e+4)[0] # for best
i_acctt = np.where(z_frontiers[:,0] < 0.645e+7)[0] # for best
i_ped5 = np.where(z_frontiers[:,1] > 21.9e+4)[0] # for best
idxs = list(set(i_acctt).intersection(set(i_ped5)))
i = np.random.choice(idxs) # 97
y_random = df_y.iloc[i].values
print(f"randomly picked i = {i}")
print(f"obj val = {z_frontiers[i]}")
metric_cand_random = get_metric(y_random)

# %%
plot_network(y_random, metric_cand_random["traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Blues",
             vmax_p=vmax_traffic)

plot_network(y_random, metric_cand_random["ped_traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Reds",
             vmax_p=vmax_ped_traffic)

plot_network(y_random, metric_cand_random["ped_comfort_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greens",
             vmax_p=vmax_ped_comfort)

# %%
## 185
## 140: Best!!!
i = 140
# i = 153 # R1_pseudo_holiday
y_choice = df_y.iloc[i].values
print(f"Choice i = {i}")
print(f"obj val = {z_frontiers[i]}")
metric_cand_random = get_metric(y_choice)

# %%
plot_network(y_choice, metric_cand_random["traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Blues",
             vmax_p=vmax_traffic, fcolor="red",
             save=True, file_path=result_dir+"designB_car_red.pdf")

plot_network(y_choice, metric_cand_random["ped_traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Reds",
             vmax_p=vmax_ped_traffic, fcolor="red",
             save=True, file_path=result_dir+"designB_ped_red.pdf")

plot_network(y_choice, metric_cand_random["ped_comfort_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greens",
             vmax_p=vmax_ped_comfort)



# %%
### OD-specific travel time analysis
# link travel time calculation
alns.reset_design(y_orig)
ueRes_orig = alns.model.dfRes
alns.reset_design(y_max)
ueRes_max = alns.model.dfRes
alns.reset_design(y_choice)
ueRes_choice = alns.model.dfRes
# t_orig = ueRes_orig.UE_time_min.values
# t_max = ueRes_max.UE_time_min.values
# t_choice = ueRes_choice.UE_time_min.values

# shortest path calculation on network x
nodes = alns.model.net.node_data.node_id.values
links_orig = ueRes_orig[["from_", "to_", "UE_time_min"]].values
links_max = ueRes_max[["from_", "to_", "UE_time_min"]].values
links_choice = ueRes_choice[["from_", "to_", "UE_time_min"]].values
G_orig = nx.DiGraph()
G_max = nx.DiGraph()
G_choice = nx.DiGraph()
G_orig.add_nodes_from(nodes)
G_max.add_nodes_from(nodes)
G_choice.add_nodes_from(nodes)
G_orig.add_weighted_edges_from(links_orig)
G_max.add_weighted_edges_from(links_max)
G_choice.add_weighted_edges_from(links_choice)
od_table = pd.read_csv("data/od_newidx.csv")
ods = od_table[['origin', 'destination']].values
sp_times = []
for o, d in ods:
    sp_orig = nx.dijkstra_path_length(G_orig, o, d)
    sp_max = nx.dijkstra_path_length(G_max, o, d)
    sp_choice = nx.dijkstra_path_length(G_choice, o, d)
    sp_times.append({"o":o, "d":d, "original": sp_orig, "max": sp_max, "choice": sp_choice})
SPdf = pd.DataFrame(sp_times)
SPdf.to_csv(result_dir+"UE_time_od.csv")
od_ue_time = pd.concat([od_table, SPdf], axis=1)
od_ue_time["increase_max"] = 100 * (od_ue_time["max"]/od_ue_time["original"])
od_ue_time["increase_choice"] = 100 * (od_ue_time["choice"]/od_ue_time["original"])
# remove walk to walk OD pairs
od_ue_time = od_ue_time.drop(index=[0,1,2,3,4,5,6])
# remove OD pairs that were not considered in the paper
od_ue_time = od_ue_time.drop(index=[40,41,42,43])
# fill NaN as aa
od_ue_time = od_ue_time.fillna("aa")
clr = {1: "blue", 0: "red"}
od_ue_time["through_traffic"] = (od_ue_time["demand_type"] == "aa") * 1 + (od_ue_time["demand_type"] == "aw") * 0
od_ue_time["through_traffic_color"] = od_ue_time["through_traffic"].map(clr)
scatter_categorical(x=od_ue_time["original"], 
                    y=od_ue_time["choice"], color=od_ue_time["through_traffic_color"], 
                    xlim=(4.5, 26.5), ylim=(4.5, 26.5), figsize=(6,6), save=True, file_path=result_dir+"plot_od_tt_choice.eps")
scatter_categorical(x=od_ue_time["original"], 
                    y=od_ue_time["max"], color=od_ue_time["through_traffic_color"], 
                    xlim=(4.5, 26.5), ylim=(4.5, 26.5), figsize=(6,6), save=True, file_path=result_dir+"plot_od_tt_max.eps")
od_ue_time["increase_choice_perc"] = od_ue_time["increase_choice"]-100
od_ue_time["increase_max_perc"] = od_ue_time["increase_max"]-100
od_ue_time["increase_max_perc"].describe().to_csv("od_tt_max_incr_describe.csv")
od_ue_time["increase_choice_perc"].describe().to_csv("od_tt_choice_incr_describe.csv")


# %%
def plot_net_under_budget(y, budget):
    plot_network(y, metric_cand_max["traffic_car"], 
             st_nodes=alns.cand_nodes, node_data=net.car_nodes,
             figsize=(7,6), size=4, cmap="Greys",
             vmax_p=5e+15, fcolor="red",
             save=True, file_path=result_dir+f"budget/net_{budget}.pdf")


# %%
obj_budget = []
area_consts = [1000]
area_consts += [5000 * (i+1) for i in range(9)]
for area in area_consts:
    i_smaller = np.where(z_frontiers[:,2] < area)[0]
    max_comf = np.max(z_frontiers[i_smaller,1])
    max_comf_idx = np.argmax(z_frontiers[i_smaller,1])
    # y_bdt = df_y.iloc[i_smaller[max_comf_idx]].values
    # plot_net_under_budget(y_bdt, area)
    obj_budget.append({"area":area, "max_comfort": max_comf})
    # i_larger = np.where(z_frontiers[:,2] > area)[0]
    # min_tt = np.min(z_frontiers[i_larger,0])
    # obj_budget.append({"area":area, "max_comfort": max_comf, "min_traveltime": min_tt})
    
# %%
