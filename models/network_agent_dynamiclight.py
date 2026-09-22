from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.optimizers import Adam
import os
from .agent import Agent
import traceback
import copy


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path)
        # ===== check num actions == num phases ============
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])
        self.num_action_dur = len(dic_traffic_env_conf["ACTION_DURATION"])
        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.num_intersections = dic_traffic_env_conf["NUM_INTERSECTIONS"]

        self.cyclicInd = [[0] * self.num_phases for _ in range(self.num_intersections)]
        self.cyclicInd2 = [0] * self.num_intersections

        self.Xs, self.Ys, self.Xs_one, self.Y_one = None, None, None, None

        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        self.max_lane = dic_traffic_env_conf["MAX_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]
        len_feat1, len_feat2 = self.cal_input_len()
        self.num_feat1 = int(len_feat1/self.max_lane)
        self.num_feat2 = int(len_feat2/self.max_lane)

        if cnt_round == 0:
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.q_network = self.build_network()
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))

                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] *
                                    self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

        # decay the epsilon
        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
    
    def cal_input_len(self):
        N1, N2 = 0, 0
        used_feature1 = self.dic_traffic_env_conf["LIST_STATE_FEATURE_1"]
        used_feature2 = self.dic_traffic_env_conf["LIST_STATE_FEATURE_2"]
        for feat_name1 in used_feature1:
            if "num_in_deg" in feat_name1:
                N1 += self.max_lane*4
            elif "phase_total" in feat_name1:
                N1 += 0
            else:
                N1 += self.max_lane
        for feat_name2 in used_feature2:
            if "num_in_deg" in feat_name2:
                N2 += self.max_lane*4
            elif "phase_total" in feat_name2:
                N2 += 0
            else:
                N2 += self.max_lane
        return N1, N2
    
    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path,  "%s.h5" % file_name))
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path,  "%s.h5" % file_name))
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    @staticmethod
    def build_memory():
        return []

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        try:
            network = copy.deepcopy(network_copy)
        except (TypeError, ValueError, AttributeError):
            network = self.build_network()
            network.set_weights(network_copy.get_weights())
        network.compile(optimizer=Adam(learning_rate=self.dic_agent_conf["LEARNING_RATE"]),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        return network

    def train_network(self):
        pass
