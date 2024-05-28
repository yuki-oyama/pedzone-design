# %%
import os
import pandas as pd
import numpy as np
from tempfile import gettempdir

from network import Network
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment
from aequilibrae.paths.traffic_class import TrafficClass

from utils import Timer
timer = Timer()
hoge = timer.stop()

# %%
class UserEquilibrium:
    
    def __init__(self, net: Network, file_name: str, fldr_path: str,
                 algorithm: str = "bfw",
                 accuracy: float = 0.05, 
                 maxIter: int = 150,
                 congestion_threshold: float = 3.0,
                 ) -> None:
        
        # parameters
        self.inf = 1e+8
        self.eps = 1e-8
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.maxIter = maxIter
        self.zeta = congestion_threshold

        # initialize model
        self.net = net
        self.iZone = np.arange(net.nZone) + 1
        self.link_data = net.link_data.copy()

        # define demand matrix
        self.aemfile = os.path.join(fldr_path, f"demand_{file_name}.aem")
        mtx = np.zeros(shape=(net.nZone, net.nZone))
        mtx[net.demand["origin"].values - 1, net.demand["destination"].values - 1] = net.demand["flow"].values
        aem = AequilibraeMatrix()
        kwargs = {'file_name': self.aemfile,
                'zones': net.nZone,
                'matrix_names': ['matrix'],
                "memory_only": False}  # We'll save it to disk so we can use it later
        aem.create_empty(**kwargs)
        aem.matrix['matrix'][:,:] = mtx[:,:]
        aem.index[:] = self.iZone[:]

        # define graph
        self.define_graph()

    def define_graph(self) -> None:
        g = Graph()
        g.cost = self.link_data["free_flow_time"].values
        g.capacity = self.link_data["capacity"].values
        g.free_flow_time = self.link_data["free_flow_time"].values
        g.network = self.link_data
        g.network_ok = True
        g.status = "OK"
        g.prepare_graph(self.iZone)
        g.set_graph("free_flow_time")
        g.cost = np.array(g.cost, copy=True)
        g.set_skimming(["free_flow_time"])
        g.set_blocked_centroid_flows(False)
        g.network["id"] = g.network["link_id"]
        self.graph = g

    def update_graph(self, new_capacity: np.ndarray = None, new_cost: np.ndarray = None) -> None:
        if new_capacity is not None:
            self.link_data["capacity"] = new_capacity
        if new_cost is not None:
            self.link_data["free_flow_time"] = new_cost
        self.define_graph()
    
    def recover_graph(self) -> None:
        self.link_data = self.net.link_data.copy()
        self.define_graph()

    def assignment(self) -> None:
        # read demand matrix
        aem = AequilibraeMatrix()
        aem.load(self.aemfile)
        aem.computational_view(["matrix"])

        # define assignment
        assigclass = TrafficClass("car", self.graph, aem)
        assig = TrafficAssignment()
        assig.set_classes([assigclass]) # start setting UE assignment
        assig.set_vdf("BPR") # link cost performance function: choose from ['bpr', 'bpr2', 'conical', 'inrets']
        assig.set_vdf_parameters({"alpha": "alpha", "beta": "beta"}) # BPR parameters
        assig.set_capacity_field("capacity") # capacity setting
        assig.set_time_field("free_flow_time") # free flow time setting
        assig.set_algorithm(self.algorithm) # solution algorithm, options: 'all-or-nothing', 'msa', 'frank-wolfe', 'fw', 'cfw', 'bfw'
        assig.max_iter = self.maxIter
        assig.rgap_target = self.accuracy
        
        # run assignment
        # timer.start()
        assig.execute() 
        # runtime = timer.stop()

        # results
        result = assig.results()
        self.report = assig.report()
        # nIter = self.report["iteration"].max()
        # print(f"Elapsed time: {runtime} seconds")
        # print(f"UE w. {self.algorithm} converged at {nIter} iterations")
        dfRes = pd.merge(self.link_data, result, on="link_id")
        dfRes = dfRes.drop(columns=["matrix_ab", "matrix_ba", "Congested_Time_AB", "Congested_Time_BA", "Delay_factor_AB", "Delay_factor_BA", "VOC_AB", "VOC_BA", "PCE_AB", "PCE_BA"])
        dfRes = dfRes.rename(columns={
            "a_node": "from_", "b_node": "to_", 
            "matrix_tot": "UE_flow",
            "Congested_Time_Max": "UE_time_min",
            "VOC_max": "Value_of_Time",
            "PCE_tot": "Passenger_Car_Equivalent"})
        self.dfRes = dfRes
    
    def save_csv(self, file_name: str, file_path: str) -> None:
        if file_name == 'base':
            self.dfRes.to_csv(file_path)
        else:
            # res_path = os.path.dirname(file_path)
            result_path = os.path.join(file_path, f'UEresult_{file_name}.csv')
            self.dfRes.to_csv(result_path)
    
    def evaluate_objective(self) -> None:
        carRes = self.dfRes.query("mode == 'car'")[["UE_flow", "UE_time_min", "length", "capacity"]]
        pedRes = self.dfRes.query("mode == 'walk'")[["UE_flow", "UE_time_min"]]
        parkRes = self.dfRes.query("mode == 'parking'")[["UE_flow", "UE_time_min"]]
        x_car, t_car = carRes["UE_flow"].values, carRes["UE_time_min"].values
        len_car, cap_car = carRes["length"].values, carRes["capacity"].values
        x_walk, t_walk = pedRes["UE_flow"].values, pedRes["UE_time_min"].values
        x_park, t_park = parkRes["UE_flow"].values, parkRes["UE_time_min"].values

        # objectives
        tt_car = x_car.dot(t_car)  #* 39.6
        tt_walk = x_walk.dot(t_walk) #* 25.64
        tt_park = x_park.dot(t_park)
        co2_car = x_car.dot(len_car) * 126.3 * 1e-3 * 3.6

        # compute mixed traffic
        congestion = x_car / cap_car
        pair_idx = self.net.ped_links["corresp_car_link"].values[:-1].astype(np.int64) # remove クレアモール
        mixed_traffic = x_walk[:-1] * t_walk[:-1] * congestion[pair_idx] * 25.64 # |Aw| x 1
        total_mixed_traffic = mixed_traffic.sum()
        comfort = self.zeta - congestion[pair_idx]
        ped_comfort = x_walk[:-1] * t_walk[:-1] * np.clip(comfort, 0, self.zeta) # |Aw| x 1
        total_ped_comfort = ped_comfort.sum()

        # store pedestrian and mixed traffic by car link idx for neighborhood design
        ped_traffic_car = x_walk[:x_car.shape[0]]
        mixed_traffic_car = mixed_traffic[:x_car.shape[0]]
        ped_comfort_car = ped_comfort[:x_car.shape[0]]
        onedir_idxs = self.net.ped_links["corresp_car_link"].values[x_car.shape[0]:-1].astype(np.int64) # remove クレアモール
        ped_traffic_car[onedir_idxs] += x_walk[x_car.shape[0]:-1] # remove クレアモール
        mixed_traffic_car[onedir_idxs] += mixed_traffic[x_car.shape[0]:] # クレアモール is already removed above
        ped_comfort_car[onedir_idxs] += ped_comfort[x_car.shape[0]:] # クレアモール is already removed above
        
        self.obj = {
            "travel_time_of_car": tt_car,
            "travel_time_of_walking": tt_walk,
            "search_time_of_parking": tt_park,
            "CO2_emission_by_car": co2_car,
            "mixed_traffic": total_mixed_traffic,
            "ped_comfort": -total_ped_comfort, # for maximize
        }
        self.metrics = {
            "congestion": congestion,
            "ped_traffic_car": ped_traffic_car,
            "mixed_traffic_car": mixed_traffic_car,
            "travel_time_car": t_car,
            "traffic_car": x_car,
            "ped_comfort_car": ped_comfort_car
        }
        return self.obj, self.metrics
        


# %%
# Test for UE assignment
if __name__ == '__main__':
    # output directories
    case_name = "kawagoe"
    out_dir = os.path.join("UE" + case_name)
    demand_dir = os.path.join(out_dir, 'demand')
    network_dir = os.path.join(out_dir, 'network')
    result_dir = os.path.join(out_dir, 'result')
    calculate_dir = os.path.join(out_dir, 'calculate')
    for dir in {out_dir, network_dir, demand_dir, result_dir, calculate_dir}:
        os.makedirs(dir, exist_ok = True)
    
    # network data
    netdata = Network()
    model = UserEquilibrium(netdata, file_name='base', fldr_path=calculate_dir,
                            accuracy=0.05, maxIter=100,
                            )

    # run assignment
    model.assignment()

    # save results
    model.save_csv(case_name, result_dir)

    # evaluate objective
    model.evaluate_objective()
    print(model.obj)

# %%
