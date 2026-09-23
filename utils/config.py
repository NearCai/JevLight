from models.random_agent import RandomAgent
from models.fixedtime_agent import FixedtimeAgent
from models.maxpressure_agent import MaxPressureAgent
from models.mplight_agent import MPLightAgent
from models.colight_agent import CoLightAgent
from models.presslight_one import PressLightAgentOne
from models.chatgpt import (ChatGPTTLCS_Wait_Time_Forecast, ChatGPTTLCS_Commonsense)
from models.dynamiclight import DynamicLightAgent
from models.jev import JevAgent
from models.laya_agent import LayaAgent


DIC_AGENTS = {
    "Random": RandomAgent,
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,
    "PressLight": PressLightAgentOne,
    "MPLight": MPLightAgent,
    "Colight": CoLightAgent,
    "ChatGPTTLCSWaitTimeForecast": ChatGPTTLCS_Wait_Time_Forecast,
    "ChatGPTTLCSCommonsense": ChatGPTTLCS_Commonsense,
    "DynamicLight": DynamicLightAgent,
    "Jev": JevAgent,
    "Laya": LayaAgent,
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Random", "Fixedtime", "MaxPressure", "PressLight", "MPLight", "Colight",
                   "DynamicLight", "Jev"],
    "LIST_MODEL_NEED_TO_UPDATE": ["PressLight", "MPLight", "Colight", "DynamicLight"],

    "NUM_LANE": 12,
    # 'WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'/ 'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT'
    "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
                  # [0, 1, 15, 16, 17, 18, 19, 20], [3, 4, 12, 13, 14, 21, 22, 23], [9, 10, 18, 19, 20, 12, 13, 14], [6, 7, 21, 22, 23, 15, 16, 17]],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 10,
    "MEASURE_TIME": 30,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_CHATGPT_AGENT_CONF = {
    "GPT_VERSION": "gpt-4",
    "LOG_DIR": "../GPT_logs"
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}
