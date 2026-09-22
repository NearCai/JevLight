from utils.utils import pipeline_wrapper, merge
from utils import config, error
import time
from multiprocessing import Process
import argparse
import os
from datetime import datetime



# 配置
set_config={
    'rounds':100,
    'dataset':'anon_3_4_jinan_real_2500.json',
    'memo':'compare',
    'process_num':1,
    'project_name':'Two_Stage-compare',
    'label':'DynamicLight-2500',
}
# agent配置
agent_config={
    'SROUND': 80, 
    'SROUND2': 160, 
    'SROUND3': 110, 
    'D_DENSE': 20, 
    'LEARNING_RATE': 0.001, 
    'LEARNING_RATE2': 0.0002, 
    'LEARNING_RATE3': 0.0005,
    'PATIENCE': 10, 
    'BATCH_SIZE': 128, 
    'BATCH_SIZE1': 128, 
    'EPOCHS': 100, 
    'SAMPLE_SIZE1': 4000, 
    'MAX_MEMORY_LEN': 12000, 
    'UPDATE_Q_BAR_FREQ': 5, 
    'UPDATE_Q_BAR_EVERY_C_ROUND': False, 
    'GAMMA': 0.8, 
    'NORMAL_FACTOR': 20, 
    'EPSILON': 0.8, 
    'EPSILON_DECAY': 0.95, 
    'MIN_EPSILON': 0.2, 
    'LOSS_FUNCTION': 
    'mean_squared_error'
}
# 交通仿真环境配置
traffic_config={
    "LABEL":set_config['label'],
    "LIST_MODEL_NEED_TO_UPDATE":['DynamicLight'],
    "MODEL":"DynamicLight",
    "PROJECT_NAME":set_config['project_name'],
    "RUN_COUNTS": 3600,
    "MODEL_NAME": "DynamicLight",
    "TOP_K_ADJACENCY": 5,
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 12,
    "OBS_LENGTH": 167,
    "BINARY_PHASE_EXPANSION": True,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [
        3,
        3,
        3,
        3
    ],
    "INTERVAL": 1,
    "DIC_REWARD_INFO": {
        "queue_length": -0.25
    },
    "PHASE": {
        "1": [0,1,1,0,1,1,0,0,1,0,0,1],
        "2": [0,0,1,0,0,1,0,1,1,0,1,1],
        "3": [1,0,1,1,0,1,0,0,1,0,0,1],
        "4": [0,0,1,0,0,1,1,0,1,1,0,1]
    },
    "list_lane_order": [
        "WL",
        "WT",
        "EL",
        "ET",
        "NL",
        "NT",
        "SL",
        "ST"
    ],
    "PHASE_LIST": [
        "WT_ET",
        "NT_ST",
        "WL_EL",
        "NL_SL"
    ],
    "NUM_ROUNDS": set_config['rounds'],
    "NUM_LANE": 12,
    "NUM_GENERATORS": 1,
    "NUM_AGENTS": 1,
    "NUM_ROW": 3,
    "NUM_COL": 4,
    "TRAFFIC_FILE": set_config['dataset'],
    "ROADNET_FILE": "roadnet_3_4.json",


    "LIST_STATE_FEATURE": [
        "phase_total",
        "lane_queue_vehicle_in",
        "lane_run_in_part",
        "num_in_deg"
    ],
    "PHASE_MAP": [
        [1,4],
        [7,10],
        [0,3],
        [6,9]
    ],
    "FORGET_ROUND": 1,
    "MIN_ACTION_TIME": 15,
    "MIN_ACTION_TIME1": 5,
    "MEASURE_TIME": 10,
    
    # dynamiclight config
    "ACTION_DURATION": {
        0: 10,
        1: 15,
        2: 20,
        3: 25,
        4: 30,
        5: 35,
        6: 40
    },
    "MAX_LANE": 16,
    "PHASETOTAL": {
        1: [0,1,1,0,1,1,0,0,1,0,0,1],
        2: [0,0,1,0,0,1,0,1,1,0,1,1],
        3: [1,0,1,1,0,1,0,0,1,0,0,1],
        4: [0,0,1,0,0,1,1,0,1,1,0,1]
    },
    "LIST_STATE_FEATURE_1": [
        "phase_total",
        "lane_queue_vehicle_in",
        "lane_run_in_part",
        "num_in_deg"
    ],
    "LIST_STATE_FEATURE_2": [
        "num_in_deg"
    ],
    "TRAFFIC_SEPARATE": set_config['dataset'],
    "RUN_COUNTS2": 3600,
}

# 目录配置
dic_config={
    'PATH_TO_MODEL': os.path.join("model", set_config['memo'], set_config['dataset'] + "_" +
                                      time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
    'PATH_TO_WORK_DIRECTORY': os.path.join("records", set_config['memo'], set_config['dataset'] + "_"
                                               + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))+"xiugaiban",
    'PATH_TO_DATA': 'data/Jinan/3_4', 
    'PATH_TO_PRETRAIN_MODEL': 'model/default', 
    'PATH_TO_ERROR': os.path.join("errors", set_config['memo'])
}



# 开始时间
print('开始时间:',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

process_list=[]
if set_config['process_num']>1:
    ppl = Process(target=pipeline_wrapper,
                    args=(agent_config,
                        traffic_config,
                        dic_config,
                        'Jinan-3_4',
                        set_config['dataset'].split(".")[0]))
    process_list.append(ppl)

    for i in range(0, len(process_list), set_config['process_num']):
        i_max = min(len(process_list), i + set_config['process_num'])
        for j in range(i, i_max):
            print(j)
            print("start_traffic")
            process_list[j].start()
            print("after_traffic")
        for k in range(i, i_max):
            print("traffic to join", k)
            process_list[k].join()
            print("traffic finish join", k)
else:
    pipeline_wrapper(dic_agent_conf=agent_config,
                         dic_traffic_env_conf=traffic_config,
                         dic_path=dic_config,
                         roadnet='Jinan-3_4',
                         trafficflow=set_config['dataset'].split(".")[0])

# 结束时间
print('结束时间:',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))