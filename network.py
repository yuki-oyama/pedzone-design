# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
def str_to_list(hoge):
    return eval(hoge)

def is_crossnode(node_id):
    cross_nodes = [2, 3, 4, 87, 93, 94, 195, 95, 96, 196, 97, 98, 99, 7, 8, 9, 10, 101, 12, 14, 15, 16, 18, 19, 22, 105, 197, 198, 199, 106, 202, 203, 108, 204, 205, 208, 110, 210, 212, 214, 29, 216, 111, 219, 221, 113, 222, 223, 224, 225, 226, 227, 228, 34, 35, 36, 229, 115, 116, 117, 230, 39, 41, 234, 118, 235, 43, 119, 120, 233, 122, 45, 46, 123, 124, 48, 49, 236, 237, 53, 127, 128, 129, 54, 130, 135, 136, 137, 138, 139, 140, 242, 244, 141, 245, 55, 143, 56, 247, 60, 144, 61, 148, 149, 153, 154, 156, 157, 255, 259, 261, 262, 264, 67, 68, 161, 162, 69, 163, 265, 72, 269, 73, 270, 272, 273, 169, 74, 75, 76, 171, 173, 174, 177, 178, 181, 183, 185, 190, 192, 193, 194, 84]
    _extend = [n + 273 for n in cross_nodes] # 273: number of nodes (and maximum index of car node)
    cross_nodes.extend(_extend)
    return (node_id in cross_nodes) * 1

# %%
class Network:

    def __init__(self) -> None:
        self.inf = 1e+8
        self.eps = 1e-8

        ## node data
        self.car_nodes = pd.read_csv("data/node_reduced.csv")
        self.ped_nodes = self.car_nodes.copy()
        self.car_nodes = self.car_nodes.assign(z = 0)
        self.ped_nodes = self.ped_nodes.assign(z = 1000)
        self.ped_nodes["node_id"] += len(self.car_nodes)
        self.node_data = pd.concat([self.car_nodes, self.ped_nodes])
        self.node_data["cross_node"] = self.node_data["node_id"].apply(is_crossnode)
        
        ## link data
        # driving links
        self.car_links = pd.read_csv("data/link_newidx.csv").drop(columns=["start", "goal"])
        self.car_links["capacity"] *= 12
        self.street = self.car_links["street"].unique()        
        
        # walking links
        oneway_links = self.car_links.query("n_direction == 1").rename(columns={"from_": "to_", "to_": "from_"})
        self.ped_links = pd.concat([self.car_links, oneway_links]) # assume pedestrians can walk bidirectionally even if car drives one way
        self.ped_links = self.ped_links.assign(corresp_car_link = self.ped_links.index) # car-ped correspondence for mixed traffic
        self.ped_links.index = np.arange(len(self.ped_links)) + len(self.car_links)
        self.ped_links["from_"] += len(self.car_nodes)
        self.ped_links["to_"] += len(self.car_nodes)
        self.ped_links["capacity"] = self.inf
        add_link = pd.Series(
            data = [76+len(self.car_nodes), 53+len(self.car_nodes), 800, self.inf, "_", 2, 1, None, 15, 0, -1],
            index = ["from_", "to_", "length", "capacity", "street", "n_direction", "hodou", "walk_correp", "width", "cross_node", "corresp_car_link"],
            name = self.ped_links.index.max() + 1
            )
        self.ped_links = pd.concat([self.ped_links, add_link.to_frame().T])
        
        # parking links
        parking = pd.read_csv("data/parking_newidx.csv").drop(columns=["i", "j"])
        parking.index += len(self.car_links) + len(self.ped_links)
        parking = parking.drop(columns=["name", "first_hour", "first_30min", "fee_per_15min", "fee_per_30min", "fee_per_hour", "condition"])
        parking["capacity"] *= 4
        coin = pd.read_csv("data/coin_newidx.csv").drop(columns=["i", "j"])
        coin.index += len(self.car_links) + len(self.ped_links) + len(parking)
        coin = coin.drop(columns=["first_20min", "first_40min", "first_30min", "first_hour", "max_fee"])
        coin["capacity"] *= 4
        self.parking_links = pd.concat([parking, coin])
        
        # different new attributes
        # mode, alpha, beta, and others for correspondence
        self.car_links = self.car_links.assign(mode = "car", alpha = 0.96, beta = 1.2, 
                                               fee = 0) # beta = 1.2 or 2.82?
        self.ped_links = self.ped_links.assign(mode = "walk", alpha = 0, beta = 1 + self.eps, 
                                               fee = 0)
        self.parking_links = self.parking_links.assign(mode = "parking", alpha = 2.62, beta = 5.0, 
                                                       length = 0, street = "_", n_direction = 0, hodou = 0, walk_corresp = None, cross_node = 0)
        # free flow time
        self.car_links["free_flow_time"] = self.car_links["length"]/(40000/60)
        self.ped_links["free_flow_time"] = self.ped_links["length"]/(4000/60)
        self.parking_links = self.parking_links.assign(free_flow_time = 1.0)
        
        # merge links
        self.link_data = pd.concat([self.car_links, self.ped_links, self.parking_links])
        self.link_data = self.link_data.astype({
            "length": "float64",
            "capacity": "float64",
            "fee": "float64",
        })

        
        ## demand (od) data
        self.demand = pd.read_csv("data/od_newidx.csv").drop(columns=["o", "d"])
        # self.demand = pd.read_csv("data/od_newidx_pseudo_holiday.csv").drop(columns=["o", "d"])
        
        # links that cannot be pedestrianized due to origin/destination flows
        self.non_candidates = []
        for onode in self.demand.origin:
            if onode < len(self.car_nodes):
                olinks = self.car_links.query(f"from_ == {onode}").index.values
                # for olink in olinks:
                # only when a single link connects from the origin
                if olinks.shape[0] == 0:
                    olink = olinks[0]
                    if self.link_cand[olink] not in self.non_candidates:
                        self.non_candidates.append(self.link_cand[olink])
        for dnode in self.demand.destination:
            if dnode < len(self.car_nodes):
                dlinks = self.car_links.query(f"to_ == {dnode}").index.values
                # for dlink in dlinks:
                # only when a single link connects to the destination
                if dlinks.shape[0] == 0:
                    dlink = dlinks[0]
                    if dlink not in self.non_candidates:
                        self.non_candidates.append(self.link_cand[dlink])

        ## prepare for Aequilibrium
        self.prepare_AE_graph()
    
    def prepare_AE_graph(self) -> None:
        
        # set demand
        origins = self.demand['origin'].unique()
        dests = self.demand['destination'].unique()
        self.zones = np.unique(np.append(origins, dests))
        self.nZone = self.zones.max()

        # set network
        self.link_data = self.link_data.rename(columns={
            "init_node": "a_node", 
            "term_node": "b_node",
            "from_": "a_node", 
            "to_": "b_node",
            "b": "alpha",
            "power": "beta",
            })
        self.link_data["link_id"] = self.link_data.index + 1
        self.link_data = self.link_data.astype({
            "link_id": "int64", 
            "a_node": "int64", 
            "b_node": "int64",
            "beta": "float64",
            "free_flow_time": "float64",
            })
        self.link_data = self.link_data.assign(direction = 1)
        self.link_data["free_flow_time"] = self.link_data["free_flow_time"].replace(0, self.eps)
        self.link_data["beta"] = self.link_data["beta"].replace(0, 1 + self.eps) # beta should not be smaller than one

# %%
