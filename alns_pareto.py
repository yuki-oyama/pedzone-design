# %%
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from network import Network
from assignment import UserEquilibrium
from utils import Timer

# %%
class ALNS:

    def __init__(self, 
                model: UserEquilibrium,
                maxChanges: int = 100,
                maxSearch: int = 50,
                # maxAccept: int = 100,
                maxZones: int = 10,
                nWarmup: int = 300,
                nReturn: int = 500,
                obj: list = ["travel_time_of_car", "ped_comfort", "capital_cost"],
                init_temp: float = 1e+9,
                temp_factor: float = 0.9,
                p_0: float = 0.99,
                p_f: float = 0.01,
                scores: list[float] = [10, 2],
                react_factor: float = 0.6,
                min_weight: float = 0.3,
                max_del_weight: float = 1.2,
                ) -> None:
        
        self.eps = 1e-8
        self.inf = 1e+8
        
        # objective values
        self.obj_keys = obj
        # self.obj = {key_: [] for key_ in self.obj_keys}
        self.obj_vals = np.zeros((1 + nWarmup + maxChanges * maxSearch + 1, len(obj)), dtype=np.float64) # saving all history of the objectives

        # solution
        self.frontiers = [] # list of frontier ids

        # temparatures
        self.T = np.ones(len(obj)) * init_temp
        self.temps = np.zeros((1 + maxChanges, len(obj)), dtype = np.float64)
        self.temps[0] = self.T
        self.temp_factor = temp_factor
        self.p_0 = p_0
        self.p_f = p_f

        # parameters
        self.maxChanges = maxChanges # maximum number of temparature updates
        self.maxSearch = maxSearch # maximum number of neighborhood searches
        # self.maxAccept = maxAccept # maximum number of accepted neighborhood during iteration
        self.maxZones = maxZones
        self.zone_ids = set([z for z in range(maxZones)])
        self.nWarmup = nWarmup
        self.nReturn = nReturn
        self.next_return = self.nWarmup + self.nReturn
        self.scores = scores
        self.gamma = react_factor
        self.min_weight = min_weight
        self.max_del_weight = max_del_weight

        # save history
        self.n = 0 # number of overall iterations
        self.m = 0 # number of solutions accepted
        self.history = {}
        
        # network and UE model
        self.model = model
        self.net = model.net
        self.fft = self.net.link_data["free_flow_time"].values.copy()
        self.car_cap = self.net.car_links["capacity"].values.copy()
        self.car_nodes = np.unique(np.concatenate(self.net.car_links[["from_", "to_"]].values))
        self.link_nodes = np.vstack(self.net.car_links[["from_", "to_"]].values)

        # decision candidates, foward & backward stars
        self.candidates = {} # key: cand number, value: list of corresponding link_ids
        self.link_cand = {} # key: link id, value: corresponding cand number
        self.link_adj = np.zeros((len(self.net.car_links), len(self.net.car_links)), dtype=np.int16)
        fstars = {a: [] for a in self.net.car_links.index}
        bstars = {a: [] for a in self.net.car_links.index}
        senders = self.net.car_links.from_.values
        receivers = self.net.car_links.to_.values
        n_cand, pair_link = 0, None
        for k in self.net.car_links.index:
            ki, kj = senders[k], receivers[k]
            is_unique = True
            for a in self.net.car_links.index:
                ai, aj = senders[a], receivers[a]
                if kj == ai:
                    fstars[k].append(a)
                    bstars[a].append(k)
                    self.link_adj[k,a] = 1
                    # k is the pair link of a which is already registered
                    if k in fstars[a]:
                        is_unique = False
                        pair_link = a
            # allocate candidate number for decision variables
            if is_unique:
                self.candidates[n_cand] = [k]
                self.link_cand[k] = n_cand
                n_cand += 1
            else:
                assert pair_link is not None, "If link is not unique, a pair link must be found."
                self.candidates[self.link_cand[pair_link]].append(k)
                self.link_cand[k] = self.link_cand[pair_link]
        
        # candidate adjacency (for neighborhood design)
        self.cand_adj = {cand: [] for cand in self.candidates.keys()}
        self.cand_nodes = np.zeros((len(self.candidates),2), dtype=np.int16)
        self.cand_areas = np.zeros(len(self.candidates), dtype=np.float64)
        for i in self.candidates.keys():
            ilinks = self.candidates[i]
            self.cand_nodes[i] = self.link_nodes[ilinks[0]]
            self.cand_areas[i] = self.net.car_links["length"].values[i] * self.net.car_links["width"].values[i]
            # search adjacent streets of i
            for j in self.candidates.keys():
                jlinks = self.candidates[j]
                for ilink in ilinks:
                    for jlink in jlinks:
                        new_connection = (self.link_adj[ilink, jlink] == 1) & (i != j) & (j not in self.cand_adj[i])
                        if new_connection:
                            self.cand_adj[i].append(j)
                            break

        # define neighborhoods
        # main categories
        self.operators = ["expand_zone", "shrink_zone", "create_new_zone", "delete_zone"]
        self.op_idxs = np.arange(len(self.operators))
        # zone choice
        self.zone_operators = [
            ["random", "max_size", "min_size"],
            ["random", "max_size", "min_size"],
            [""], # for get_operator_idx()
            ["random", "max_size", "min_size"]
        ]
        self.opzone_idxs = [np.arange(len(l)) for l in self.zone_operators]
        # link choice
        self.st_operators = [
            ["random", "min_cap", "max_walk", "max_mixed_traffic"], # pedestrianize
            ["random", "max_cap", "min_walk"], # de-pedestrianize
            ["random", "min_cap", "max_walk", "max_mixed_traffic"], # create_new
            [""] # for get_operator_idx()
        ] # different link choice mechanisms for each operator
        self.opst_idxs = [np.arange(len(l)) for l in self.st_operators]
        
        # total number of neighborhoods
        self.n_neighbor = 0
        for i in self.op_idxs:
            self.n_neighbor += len(self.zone_operators[i]) * len(self.st_operators[i])
        
        # neighbor weight and probability
        self.w = np.ones(self.n_neighbor, dtype=np.float64)
        self.p = self.w / self.w.sum()
        self.p_op_idxs = [] # list of indexes for each operator
        self.p_zone_idxs = {} # key: op_idx, val: dict = {key: opzone_idx, val: list of indexes for each zone operator}
        for o in range(len(self.operators)):
            operator_idxs = []
            self.p_zone_idxs[o] = []
            for z in range(len(self.zone_operators[o])):
                zone_idxs = []
                for s in range(len(self.st_operators[o])):
                    idx_ = self.get_operator_idx(o, z, s)
                    operator_idxs.append(idx_)
                    zone_idxs.append(idx_)
                self.p_zone_idxs[o].append(zone_idxs)
            self.p_op_idxs.append(operator_idxs)
        
        # decompose probabilities
        self.decompose_prob()

        # acceptance-rejection records
        self.n_trials = np.zeros(self.n_neighbor, dtype=np.int16)
        self.op_scores = np.zeros(self.n_neighbor, dtype=np.float32)

        # history record
        # operator selection probability
        self.p_hist = np.zeros((self.maxChanges+1, self.n_neighbor), dtype=np.float64)
        self.p_hist[0] = self.p
        # success rate history overall: n_trials, n_acceptance, n_frontiers
        self.success = np.zeros((self.n_neighbor, 3), dtype=np.int32)
        # accepted / return_to_archive timings
        self.accept_idxs = [] # after warming up: used to update temparature
        self.return_idxs = []
        # changes in the number of frontier solutions
        self.n_frontiers = []
        # store indexes of operators used every iteration
        self.operator_used = [""]
    
    def get_operator_idx(self, op_idx, opzone_idx, opst_idx):
        idx_ = 0
        for i in range(op_idx):
            idx_ += len(self.zone_operators[i]) * len(self.st_operators[i])
        return idx_ + opzone_idx * len(self.st_operators[op_idx]) + opst_idx
    
    def initialize_design(self, n_streets: int = 0) -> None:
        # randomly choose designed streets
        designed_streets = list(np.random.choice(np.arange(len(alns.candidates)), n_streets, replace=False))
        # allocate zones (not always different zones for different streets)
        zone_id = 0
        add_streets = {}
        while True:
            k = designed_streets.pop(0)
            add_streets[k] = zone_id
            for a in designed_streets:
                if a in self.cand_adj[k]:
                    add_streets[a] = zone_id
                    designed_streets.remove(a)
            if len(designed_streets) == 0:
                break
            zone_id += 1
        
        # initialize decision variables
        self.y = np.zeros(len(self.candidates.keys()), dtype=np.int16)
        self.y_zone = -1 * np.ones(len(self.candidates.keys()), dtype=np.int16)
        self.y_node = {n: [] for n in self.car_nodes} # key: node id, value: list of street ids (multiple times of same id allowed)
        self.isolated = np.ones(len(self.candidates), dtype=np.int16)
        
        # update design and perform assignment
        self.update_design(add_streets = add_streets)
    
    def reset_design(self, y: np.ndarray = None) -> None:
        if y is None:
            # initialize decision variables to zero
            y = np.zeros(len(self.candidates.keys()), dtype=np.int16)
            y_zone = -1 * np.ones(len(self.candidates.keys()), dtype=np.int16)
            y_node = {n: [] for n in self.car_nodes} # key: node id, value: list of street ids (multiple times of same id allowed)

            # isolated candidate list (for new zone operator)
            isolated = np.ones(len(self.candidates), dtype=np.int16)
        else:
            # load design (from a solution)
            designed_streets = y.nonzero()[0]
            y_zone = -1 * np.ones(len(self.candidates.keys()), dtype=np.int16)
            y_zone[designed_streets] = np.arange(len(designed_streets))
            y_node = {n: [] for n in self.car_nodes} # key: node id, value: list of street ids (multiple times of same id allowed)
            isolated = np.ones(len(self.candidates), dtype=np.int16)
            
            for k in designed_streets:
                for a in designed_streets:
                    if k >= a:
                        continue
                    if a in self.cand_adj[k]:
                        y_zone[a] = y_zone[k]
            for k in designed_streets:
                nodes = self.cand_nodes[k]
                isolated[k] = 0
                for node in nodes:
                    y_node[node].append(y_zone[k])
                for a in self.candidates.keys():
                    isolated[a] = 0

        # perform assignment
        self.perform_assignment(design=[y, y_zone, y_node, isolated])
        
        # check currently generated zones and their sizes
        self.check_zones()
    
    def update_design(self, add_streets: dict = {}, remove_streets: dict = {}) -> None:
        # print("Update:", add_streets, remove_streets)
        y = self.y.copy()
        y_zone = self.y_zone.copy()
        y_node = deepcopy(self.y_node)
        isolated = self.isolated.copy()

        # update decision variables
        for cand_id, zone_id in add_streets.items():
            y[cand_id] = 1
            y_zone[cand_id] = zone_id
            for node_id in self.cand_nodes[cand_id]:
                y_node[node_id].append(zone_id)
            # connected candidates from newly pedestrianized street
            for k in self.cand_adj[cand_id]:
                isolated[k] = 0 # not isolated anymore
        for cand_id, zone_id in remove_streets.items():
            y[cand_id] = 0
            y_zone[cand_id] = -1
            for node_id in self.cand_nodes[cand_id]:
                y_node[node_id].remove(zone_id)
            # connected candidates from removed street
            for k in self.cand_adj[cand_id]:
                if y[k] == 1:
                    continue
                # if k is isolated or not               
                iso_ = 1
                for a in self.cand_adj[k]:
                    iso_ -= y[a] # if a is pedestrian street then y = 1 -> not isolated
                    if iso_ == 0:
                        break
                isolated[k] = iso_

        # perform assignment
        new_design = [y, y_zone, y_node, isolated]
        self.perform_assignment(new_design)
        
        # check currently generated zones and their sizes
        self.check_zones()
    
    def perform_assignment(self, design) -> None:
        y, y_zone, y_node, isolated = design
        
        # update counter and record time
        self.n += 1
        self.timer.stop()
        self.timer.start()

        # update graph: assume set the costs of pedestrianized links to inf
        designed_links = []
        for street in y.nonzero()[0]:
            designed_links.extend(self.candidates[street])
        new_fft = self.fft.copy()
        new_fft[designed_links] = self.inf
        self.model.update_graph(new_cost = new_fft)        
        
        # evaluate objectives
        self.model.assignment()
        obj, metrics = self.model.evaluate_objective()
        # capital cost calculation
        if "capital_cost" in self.obj_keys:
            cost = self.cand_areas[y.nonzero()[0]].sum()
            obj["capital_cost"] = cost
        z_new = np.array([obj[key_] for key_ in self.obj_keys], dtype=np.float64)
        self.obj_vals[self.n] = z_new

        # update only when accepted
        accept = True
        is_frontier = True
        if self.m > 0:
            # z_old = self.obj_vals[self.n - 1]
            p_accept, score, is_frontier = self.accept_prob(self.curr_z, z_new)
            accept = (np.random.random() <= p_accept)
            self.operator_used.append(self.try_idx)
        
        if accept:
            # print("new solution accepted")
            if self.m > 0:
                self.n_trials[self.try_idx] += 1
                self.op_scores[self.try_idx] += score
                self.success[self.try_idx][:2] += 1
                self.success[self.try_idx][2] += (is_frontier * 1)
            if self.n >= self.nWarmup:
                self.accept_idxs.append(self.n)
            # update solutions
            self.m += 1
            self.y = y
            self.y_zone = y_zone
            self.y_node = y_node
            self.isolated = isolated
            self.curr_z = z_new
            # update metrics for next neighborhood search
            self.__dict__.update(**metrics)
            # archive frontier
            if is_frontier:
                # record how number of pareto frontiers changed
                self.n_frontiers.append(len(self.frontiers)+1)
                print(f"New frontier found: {self.n}, {self.m}, {len(self.frontiers)+1}, {z_new}")
                self.frontiers.append(self.n)
                self.history[self.n] = {
                    "y": y,
                    "y_zone": y_zone,
                    "y_node": y_node,
                    "isolated": isolated,
                    "obj": z_new,
                    }
            # save obective value
            # if self.n >= self.nWarmup:
            #     for i, obj_key in enumerate(self.obj.keys()):
            #         self.obj[obj_key].append(z_new[i])
        else:
            # print("new solution rejected")
            self.n_trials[self.try_idx] += 1
            self.success[self.try_idx][0] += 1
    
    def check_zones(self) -> None:
        # check currently generated zones and their sizes
        curr_zones, zone_sizes = np.unique(self.y_zone, return_counts=True)
        # remove zone_id -1
        self.curr_zones = curr_zones[1:]
        self.zone_sizes = zone_sizes[1:]
        self.n_zones = len(self.curr_zones)
            
    def find_neighbor(self) -> None:
        """Find a neighbor based on the current solution
        """
        # till a neighbor is found
        while True:
            # select operator
            avail_op = np.array([(self.n_zones > 0)*1, (self.n_zones > 0)*1, (self.n_zones < self.maxZones)*1, (self.n_zones > 0)*1])
            p = avail_op * self.op_probs
            p = p / p.sum()
            op_idx = np.random.choice(self.op_idxs, p = p)
            operator = self.operators[op_idx]

            # select zone operator
            opzone_idx = np.random.choice(
                    self.opzone_idxs[op_idx], p = self.opzone_probs[op_idx] / np.sum(self.opzone_probs[op_idx]))
            zone_operator = self.zone_operators[op_idx][opzone_idx]

            # select street operator
            opst_idx = np.random.choice(
                    self.opst_idxs[op_idx], p = self.opst_probs[op_idx][opzone_idx] / np.sum(self.opst_probs[op_idx][opzone_idx]))
            st_operator = self.st_operators[op_idx][opst_idx]

            # neighborhood idx
            self.try_idx = self.get_operator_idx(op_idx, opzone_idx, opst_idx)

            # print("Search for", operator, zone_operator, st_operator)

            # select street and do operation
            # Operator 1: expanding a zone
            if operator == "expand_zone":
                # select zone
                zone_id = self.choose_zone(zone_operator)

                # prepare possible street list to expand
                new_cands = []
                curr_streets = np.where(self.y_zone == zone_id)[0]
                for k in curr_streets:
                    for a in self.cand_adj[k]:
                        # only for street not yet pedestrianized
                        if self.y[a] == 1:
                            continue
                        a_nodes = self.cand_nodes[a]
                        possible = True
                        for a_node in a_nodes:
                            connections = self.y_node[a_node]
                            # connection to other zones is not allowed
                            if not (np.unique(connections) == zone_id).all():
                                possible = False
                        if possible:
                            new_cands.append(a)
                
                if len(new_cands) > 0:
                    # choose street
                    add_st = self.choose_street(new_cands, st_operator)
                    self.update_design(add_streets = {add_st: zone_id})
                    break
                else:
                    print("Candidate list is empty.")
        
            # Operator 2: shrinking a zone
            elif operator == "shrink_zone":
                # select zone
                zone_id = self.choose_zone(zone_operator)

                # prepare street list to remove
                curr_streets = np.where(self.y_zone == zone_id)[0]
                st_ranks = np.zeros_like(curr_streets)
                for i, a in enumerate(curr_streets):
                    a_nodes = self.cand_nodes[a]
                    for a_node in a_nodes:
                        connections = self.y_node[a_node]
                        st_ranks[i] += len(connections)
                # candidates are only streets with minimum ranks (can be multiple)
                new_cands = curr_streets[np.where(st_ranks == st_ranks.min())[0]]
                
                # choose street
                remove_st = self.choose_street(new_cands, st_operator)
                self.update_design(remove_streets = {remove_st: zone_id})
                break

            # Operator 3: creating a new zone
            elif operator == "create_new_zone":
                # candidates: all isolated streets
                new_cands = np.nonzero(self.isolated)[0]
                # new zone id: pick up from available zone id set
                zone_id = np.random.choice(list(self.zone_ids.difference(set(self.curr_zones))))
                # choose street
                add_st = self.choose_street(new_cands, st_operator)
                self.update_design(add_streets = {add_st: zone_id})
                break
            
            # Operator 4: deleting an existing zone
            elif operator == "delete_zone":
                # select zone
                zone_id = self.choose_zone(zone_operator)

                # remove all streets in the zone
                curr_streets = np.where(self.y_zone == zone_id)[0]
                remove_streets = {st_id: zone_id for st_id in curr_streets}
                self.update_design(remove_streets = remove_streets)
                break
    
    def choose_zone(self, zone_operator: str) -> int:
        """
        Choose a zone to operate

        Args:
            zone_operator (str): "random", "max_size", or "min_size"
        
        Returns:
            chosen zone id (int)
        """
        if zone_operator == "random":
            return np.random.choice(self.curr_zones)
        elif zone_operator == "max_size":
            return self.curr_zones[np.argmax(self.zone_sizes)]
        elif zone_operator == "min_size":
            return self.curr_zones[np.argmin(self.zone_sizes)]
    
    def choose_street(self, new_cands, st_operator: str) -> int:
        """
        Choose a street to pedestrianize or de-pedestrianize

        Args:
            new_cands (list or np.ndarray): candidate ids
            st_operator (str): street operator
        
        Returns:
            chosen street id (int)
        """

        if st_operator == "random":
            cand_id = np.random.choice(new_cands)
        elif st_operator == "max_cap":
            cand_id = self.choose_street_by_metric(new_cands, self.car_cap, "max")
        elif st_operator == "min_cap":
            cand_id = self.choose_street_by_metric(new_cands, self.car_cap, "min")
        elif st_operator == "min_congestion":
            cand_id = self.choose_street_by_metric(new_cands, self.congestion, "min")
        elif st_operator == "max_walk":
            cand_id = self.choose_street_by_metric(new_cands, self.ped_traffic_car, "max")
        elif st_operator == "min_walk":
            cand_id = self.choose_street_by_metric(new_cands, self.ped_traffic_car, "min")
        elif st_operator == "max_mixed_traffic":
            cand_id = self.choose_street_by_metric(new_cands, self.mixed_traffic_car, "max")
        
        return cand_id
    
    def choose_street_by_metric(self, cands, metric: np.ndarray, syntax: str) -> int:
        """
        Choose a street to pedestrianize or de-pedestrianize

        Args:
            cands (list or np.ndarray): candidate ids
            metric (np.ndarray): vector of |A_car| to choose a street (e.g., pedestrian traffic)
            syntax (str): "max" or "min"
        
        Returns:
            chosen street id (int)
        """

        metric_cand = np.zeros(len(cands), dtype=np.float64)
        for i, cand_id in enumerate(cands):
            ilinks = self.candidates[cand_id]
            # update capacity of candidates
            metric_cand[i] += metric[ilinks].sum()
        
        if syntax == "max":
            return cands[np.argmax(metric_cand)]
        elif syntax == "min":
            return cands[np.argmin(metric_cand)]
    
    def accept_prob(self, z_old: np.ndarray, z_new: np.ndarray):
        # else: acceptance probability
        p = 1.
        for zo, zn, Ti in zip(z_old, z_new, self.T):
            if zn < zo:
                p *= 1.
            else:
                p *= np.exp(-(zn - zo) / Ti)
        
        # do not accept a too bad solution as a frontier
        if p > 1e-2:
            # for a new pareto frontier solution
            if self.is_frontier(z_new):
                return 1., self.scores[0], True
        
        return p, self.scores[1], False
    
    def is_frontier(self, z_new: np.ndarray) -> bool:
        for f_n in self.frontiers:
            z_pareto = self.obj_vals[f_n]
            dominated = (z_pareto <= z_new + self.eps).all() # added eps to remove the same value
            if dominated:
                return False
        # not dominated by any frontier solutions
        self.remove_old_frontiers(z_new)
        return True

    def remove_old_frontiers(self, z_new: np.ndarray) -> None:
        frontiers = []
        for f_n in self.frontiers:
            z_pareto = self.obj_vals[f_n]
            dominated = (z_new <= z_pareto + self.eps).all()
            if not dominated:
                frontiers.append(f_n)
        self.frontiers = frontiers    
    
    def update_prob(self):
        # update weight
        new_w = self.op_scores / np.clip(self.n_trials, 1, None)
        reaction = self.gamma * (self.n_trials > 0) # if no trials, keep previous weight (i.e. reaction = 0)
        self.w = (1 - reaction) * self.w + reaction * new_w
        self.w = np.clip(self.w, self.min_weight, None)
        self.w[self.p_op_idxs[-1]] = np.clip(self.w[self.p_op_idxs[-1]], self.min_weight, self.max_del_weight)
        # print(f"scores: {self.op_scores}, weights: {self.w}")
        # update p
        self.p = self.w / self.w.sum()
        self.decompose_prob()
        # reset scores
        self.op_scores *= 0
        self.n_trials *= 0
    
    def decompose_prob(self):
        # decompose probabilities
        self.op_probs = [self.p[idxs_].sum() for idxs_ in self.p_op_idxs]
        self.opst_probs = [
            [self.p[self.p_zone_idxs[o][z]] for z in range(len(self.zone_operators[o]))]
            for o in range(len(self.operators))
        ]
        self.opzone_probs = [
            [self.p[self.p_zone_idxs[o][z]].sum() for z in range(len(self.zone_operators[o]))]
            for o in range(len(self.operators))
        ]
        print(f"New operator probability: {self.op_probs}")
        print(f"Detail: {self.p}")
    
    def update_temparature(self, n_change: int) -> None:
        # sigma = np.array([np.std(z) for z in self.obj.values()], dtype=np.float64)
        sigma = np.std(self.obj_vals[self.accept_idxs], axis=0)
        self.T = -sigma / np.log(self.p_0 + (self.p_f - self.p_0)*(n_change/self.maxChanges))
        self.temps[n_change+1] = self.T
        print(f"New temparature: {self.T}")
    
    def return_to_archive(self) -> None:
        print("Return-to-archive strategy")
        # record when the strategy is used
        self.return_idxs.append(self.n)
        
        # reset design to a frontier
        f_n = np.random.choice(self.frontiers)
        design = self.history[f_n]
        self.y = design["y"].copy()
        self.y_zone = design["y_zone"].copy()
        self.y_node = deepcopy(design["y_node"])
        self.isolated = design["isolated"].copy()
        self.curr_z = self.obj_vals[f_n]

        # update zone infomation
        self.check_zones()

        # next return
        self.nReturn = max(self.nReturn * 0.9, self.maxSearch)
        self.next_return += int(self.nReturn)

    def process_iteration(self, n_change: int) -> None:        
        # update_temparature
        self.update_temparature(n_change)

        # update probability
        self.update_prob()
        # save
        self.p_hist[n_change+1] = self.p

    def run(self):
        print("Start ALNS")
        self.timer = Timer()

        # Initialization: decision variables and perform assignment
        self.reset_design()
        # self.initialize_design(n_streets=5)
        print(f"Initial solution. No. ped streets: {self.y.sum()}, curr_zones: {self.curr_zones}, zone_size: {self.zone_sizes}")

        print(f"Warming up...")
        for _ in range(self.nWarmup):
            self.find_neighbor()

        # loop start
        for n_change in range(self.maxChanges):
            print(f"Iteration {n_change} started...")
            
            # update accuracy of assignment
            # if n_change > 0.75 * self.maxChanges:
            #     self.model.accuracy = 0.01
            # elif n_change > 0.5 * self.maxChanges:
            #     self.model.accuracy = 0.025
            
            for _ in range(self.maxSearch):
                if self.n == self.next_return:
                    # return to archive (one of frontier)
                    self.return_to_archive()
                # else:
                
                # search neighbor solution and update if it's accepted
                self.find_neighbor()
                
                # print
                if self.n % 10 == 0:
                    print(f"n: {self.n}, m: {self.m}, no. ped streets: {self.y.sum()}, curr_zones: {self.curr_zones}, zone_size: {self.zone_sizes}")
            # while True:
            #     # search neighbor solution and update if it's accepted
            #     self.find_neighbor()
            #     # print
            #     if self.n % 10 == 0:
            #         print(f"n: {self.n}, m: {self.m}, no. ped streets: {self.y.sum()}, curr_zones: {self.curr_zones}, zone_size: {self.zone_sizes}")
            #     # check termination of the iteration
            #     if self.n > self.maxSearch or self.m > self.maxAccept:
            #         break
            runtime = np.sum(self.timer.times[-self.maxSearch:])
            print(f"Iteration {n_change} finished with {runtime:.2f}s")
            self.process_iteration(n_change)

        self.timer.stop()
        runtime = np.sum(self.timer.times)
        print(f"Algorithm terminated, took {runtime:.2f}s in total.")

# %%
if __name__ == '__main__':
    # seed
    np.random.seed(123)

    # output directories
    case_name = "_test2"
    out_dir = os.path.join("ALNS" + case_name)
    result_dir = os.path.join(out_dir, 'result/')
    calculate_dir = os.path.join(out_dir, 'calculate')
    for dir in {out_dir, result_dir, calculate_dir}:
        os.makedirs(dir, exist_ok = True)
    
    # network data
    net = Network()

    # define model
    model = UserEquilibrium(net, file_name='base', fldr_path=calculate_dir,
                            accuracy=0.05, maxIter=100,
                            )                                                                                                                                                  

    # define algorithm
    alns = ALNS(model, maxChanges=100, maxSearch=50, nWarmup=100, nReturn=1000, maxZones=5)
    
    # run
    alns.run()

    # %%
    # results
    print(alns.frontiers)
    df_p = pd.DataFrame(alns.p_hist)
    df_success = pd.DataFrame(alns.success)
    df_frontier = pd.DataFrame(alns.frontiers)
    df_obj = pd.DataFrame(alns.obj_vals)

    y = [alns.history[f_n]["y"] for f_n in alns.frontiers]
    df_y = pd.DataFrame(y)

    df_t = pd.DataFrame(alns.timer.times)
    df_nF = pd.DataFrame(alns.n_frontiers)
    df_R = pd.DataFrame(alns.return_idxs)
    df_op = pd.DataFrame(alns.operator_used)

    df_success.to_csv(result_dir + "success.csv", index=False)
    df_frontier.to_csv(result_dir + "frontier.csv", index=False)
    df_obj.to_csv(result_dir + "objval.csv", index=False)
    df_y.to_csv(result_dir + "y.csv", index=False)
    df_p.to_csv(result_dir + "p.csv", index=False)
    df_t.to_csv(result_dir + "times.csv", index=False)
    df_nF.to_csv(result_dir + "n_frontier.csv", index=False)
    df_R.to_csv(result_dir + "returns.csv", index=False)
    df_op.to_csv(result_dir + "operator_used.csv", index=False)

    

# %%
