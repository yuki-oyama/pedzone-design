import os
import numpy as np
import pandas as pd
from network import Network
from assignment import UserEquilibrium
from alns_pareto import ALNS
import json
import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Parameters
model_arg = add_argument_group('ALNS')
model_arg.add_argument('--seed', type=int, default=124, help='seed number')
model_arg.add_argument('--case_name', type=str, default="test", help='name of test case')
model_arg.add_argument('--accuracy', type=float, default=0.01, help='accuarcy for UE')
model_arg.add_argument('--init_temp', type=float, default=1e+8, help='initial temparature')
model_arg.add_argument('--maxChanges', type=int, default=200, help='number of temparature changes')
model_arg.add_argument('--maxSearch', type=int, default=50, help='number of iterations at a temparature')
model_arg.add_argument('--maxZones', type=int, default=5, help='maximum number of zones')
model_arg.add_argument('--nWarmup', type=int, default=300, help='number of iterations for warming up')
model_arg.add_argument('--nReturn', type=int, default=1000, help='initial number of iterations to return to archive')
model_arg.add_argument('--react_factor', type=float, default=0.5, help='reaction factor')
model_arg.add_argument('--min_weight', type=float, default=0.1, help='minimum weight value')
model_arg.add_argument('--max_del_weight', type=float, default=3.0, help='maximum weight value for zone-delete operation')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


if __name__ == '__main__':
    config, _ = get_config()
    
    # seed
    np.random.seed(config.seed)

    # output directories
    case_name = config.case_name
    out_dir = os.path.join("ALNS_" + case_name)
    result_dir = os.path.join(out_dir, 'result/')
    config_dir = os.path.join(out_dir, 'config/')
    for dir in {out_dir, result_dir, config_dir}:
        os.makedirs(dir, exist_ok = True)
    
    # network data
    net = Network()

    # define model
    model = UserEquilibrium(net, file_name='base', fldr_path=config_dir,
                            accuracy=config.accuracy, maxIter=100,
                            )                                                                                                                                                  

    # define algorithm
    alns = ALNS(model, init_temp=config.init_temp,
                    maxChanges=config.maxChanges, maxSearch=config.maxSearch, 
                    nWarmup=config.nWarmup, nReturn=config.nReturn, maxZones=config.maxZones,
                    react_factor=config.react_factor, 
                    min_weight=config.min_weight, max_del_weight=config.max_del_weight)
    
    # run
    alns.run()

    # results
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
    df_temp = pd.DataFrame(alns.temps)

    df_success.to_csv(result_dir + "success.csv", index=False)
    df_frontier.to_csv(result_dir + "frontier.csv", index=False)
    df_obj.to_csv(result_dir + "objval.csv", index=False)
    df_y.to_csv(result_dir + "y.csv", index=False)
    df_p.to_csv(result_dir + "p.csv", index=False)
    df_t.to_csv(result_dir + "times.csv", index=False)
    df_nF.to_csv(result_dir + "n_frontier.csv", index=False)
    df_R.to_csv(result_dir + "returns.csv", index=False)
    df_op.to_csv(result_dir + "operator_used.csv", index=False)
    df_temp.to_csv(result_dir + "temparature.csv", index=False)
    
    
    with open(config_dir + "config.json", mode="w") as f:
        json.dump(config.__dict__, f, indent=4)