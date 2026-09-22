from copy import deepcopy
import copy
import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
from multiprocessing import Process
from .my_utils import get_state_detail, load_json, calculate_road_length
from functools import reduce

location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {"N": "North", "S": "South", "E": "East", "W": "West"}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

class Intersection:
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, lanes_length_dict):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.lane_length = lanes_length_dict
        self.obs_length = dic_traffic_env_conf["OBS_LENGTH"]

        self.list_approachs = ["W", "E", "N", "S"]
        # corresponding exiting lane for entering lanes
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approachs}
        self.list_phases = dic_traffic_env_conf["PHASE"]

        # generate all lanes
        self.list_entering_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + "_" + str(i) for i in
                                         range(lane_number)]
        self.list_exiting_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + "_" + str(i) for i in
                                        range(lane_number)]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]

        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        # =========== signal info set ================
        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode="a", header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time, path_to_log):
        action=int(action)
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                # self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment
                self.next_phase_to_set_index = action + 1
            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state["get_lane_vehicles"][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step_in)
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step_in) - set(self.list_lane_vehicle_previous_step_in))
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(set(self.list_lane_vehicle_current_step_in) -
                                         set(self.list_lane_vehicle_previous_step_in))
        else:
            list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step_in) -
                                         set(self.list_lane_vehicle_current_step_in))
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle_entring()
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream()
        dic_feature["delta_lane_num_vehicle"] = [dic_feature["lane_num_vehicle"][i] -
                                                 dic_feature["lane_num_vehicle_downstream"][i]
                                                 for i in range(12)]
        dic_feature["lane_num_waiting_vehicle_in"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_waiting_vehicle_out"] = self._get_lane_queue_length(self.list_exiting_lanes)

        dic_feature["traffic_movement_pressure_queue"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["traffic_movement_pressure_queue_efficient"] = self._get_traffic_movement_pressure_efficient(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["traffic_movement_pressure_num"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_vehicle"], dic_feature["lane_num_vehicle_downstream"])

        tmp_part_n, tmp_part_q, tmp_efficient_part, enter_running_part, lepq = self._get_part_traffic_movement_features()

        dic_feature["lane_enter_running_part"] = list(enter_running_part)

        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        dic_feature["num_in_seg_attend"] = self._orgnize_several_segments_attend(dic_feature["lane_num_waiting_vehicle_in"],
                                                                                 dic_feature["lane_num_waiting_vehicle_out"])
        self.dic_feature = dic_feature

    def _orgnize_several_segments_attend(self, queue_in, queue_out):
        part1, part2, part3 = self._get_several_segments_attend(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                                vehicle_distance=self.dic_vehicle_distance_current_step,
                                                                vehicle_speed=self.dic_vehicle_speed_current_step,
                                                                lane_length=self.lane_length,
                                                                list_lanes=self.list_lanes)
        run_in_part1 = [float(len(part1[lane])) for lane in self.list_entering_lanes]
        run_in_part2 = [float(len(part2[lane])) for lane in self.list_entering_lanes]
        run_in_part3 = [float(len(part3[lane])) for lane in self.list_entering_lanes]

        run_out_part1 = [float(len(part1[lane])) for lane in self.list_exiting_lanes]
        run_out_part2 = [float(len(part2[lane]))for lane in self.list_exiting_lanes]
        run_out_part3 = [float(len(part3[lane])) for lane in self.list_exiting_lanes]

        total_in, total_out = [], []
        for i in range(12):
            total_in.extend([run_in_part1[i], run_in_part2[i], run_in_part3[i], queue_in[i]])
            total_out.extend([run_out_part1[i], run_out_part2[i], run_out_part3[i], queue_out[i]])
        return total_in + total_out

    def _get_several_segments_attend(self, lane_vehicles, vehicle_distance, vehicle_speed,
                                           lane_length, list_lanes):
        obs_length = 100
        part1, part2, part3 = {}, {}, {}
        for lane in list_lanes:
            part1[lane], part2[lane], part3[lane] = [], [], []
            for vehicle in lane_vehicles[lane]:
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                    continue
                if vehicle_speed[vehicle] > 0.1:
                    temp_v_distance = vehicle_distance[vehicle]
                    if temp_v_distance > lane_length[lane] - obs_length:
                        part1[lane].append(vehicle)
                    elif lane_length[lane] - 2 * obs_length < temp_v_distance <= lane_length[lane] - obs_length:
                        part2[lane].append(vehicle)
                    elif lane_length[lane] - 3 * obs_length < temp_v_distance <= lane_length[lane] - 2 * obs_length:
                        part3[lane].append(vehicle)
        return part1, part2, part3

    @staticmethod
    def _get_traffic_movement_pressure_general(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] for j in range(12)]
        return t_m_p

    @staticmethod
    def _get_traffic_movement_pressure_efficient(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]]/3 for j in range(12)]
        return t_m_p

    def _get_part_traffic_movement_features(self):
        """
        return: part_traffic_movement_pressure_num:     both the end and the beginning of the lane
                part_patrric_movement_pressure_queue:   all at the end of the road
                part_entering_running_vehicles:         part obs of the running vehicles
        """
        f_p_num, l_p_num, l_p_q = self._get_part_observations(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                              vehicle_distance=self.dic_vehicle_distance_current_step,
                                                              vehicle_speed=self.dic_vehicle_speed_current_step,
                                                              lane_length=self.lane_length,
                                                              obs_length=self.obs_length,
                                                              list_lanes=self.list_lanes)
        """calculate traffic_movement_pressure with part queue"""
        list_entering_part_queue = [len(l_p_q[lane]) for lane in self.list_entering_lanes]
        list_exiting_part_queue = [len(l_p_q[lane]) for lane in self.list_exiting_lanes]
        tmp_queue_efficient_part = self._get_traffic_movement_pressure_efficient(list_entering_part_queue,
                                                                                 list_exiting_part_queue)
        tmp_queue_part = self._get_traffic_movement_pressure_general(list_entering_part_queue,
                                                                     list_exiting_part_queue)

        """calculate traffic_movement_pressure with part num vehicle"""
        # entering
        list_entering_num_f = [len(f_p_num[lane]) for lane in self.list_entering_lanes]
        list_entering_num_l = [len(l_p_num[lane]) for lane in self.list_entering_lanes]
        entering_num = np.array(list_entering_num_f) + np.array(list_entering_num_l)
        # exiting
        list_exiting_num_f = [len(f_p_num[lane]) for lane in self.list_exiting_lanes]
        list_exiting_num_l = [len(l_p_num[lane]) for lane in self.list_exiting_lanes]
        exiting_num = np.array(list_exiting_num_f) + np.array(list_exiting_num_l)
        traffic_movement_pressure_nums = self._get_traffic_movement_pressure_general(entering_num, exiting_num)
        # part of entering running vehicles, all at the end of the road
        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)

        return traffic_movement_pressure_nums, tmp_queue_part, tmp_queue_efficient_part, part_entering_running, list_entering_part_queue

    @staticmethod
    def _get_part_observations(lane_vehicles, vehicle_distance, vehicle_speed,
                               lane_length, obs_length, list_lanes):
        """
            Input: lane_vehicles :      Dict{lane_id    :   [vehicle_ids]}
                   vehicle_distance:    Dict{vehicle_id :   float(dist)}
                   vehicle_speed:       Dict{vehicle_id :   float(speed)}
                   lane_length  :       Dict{lane_id    :   float(length)}
                   obs_length   :       The part observation length
                   list_lanes   :       List[lane_ids at the intersection]
        :return:
                    part_vehicles:      Dict{ lane_id, [vehicle_ids]}
        """
        # get vehicle_ids and speeds
        first_part_num_vehicle = {}
        first_part_queue_vehicle = {}  # useless, at the begin of lane, there is no waiting vechiles
        last_part_num_vehicle = {}
        last_part_queue_vehicle = {}

        for lane in list_lanes:
            first_part_num_vehicle[lane] = []
            first_part_queue_vehicle[lane] = []
            last_part_num_vehicle[lane] = []
            last_part_queue_vehicle[lane] = []
            last_part_obs_length = lane_length[lane] - obs_length
            for vehicle in lane_vehicles[lane]:
                """ get the first part of obs
                    That is vehicle_distance <= obs_length 
                """
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        first_part_queue_vehicle[lane].append(vehicle)

                """ get the last part of obs
                    That is  lane_length-obs_length <= vehicle_distance <= lane_length 
                """
                if temp_v_distance >= last_part_obs_length:
                    last_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle_entring(self):
        """
        vehicle number for each lane
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_entering_lanes]

    def _get_lane_num_vehicle_downstream(self):
        """
        vehicle number for each lane, exiting
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_exiting_lanes]

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for
                     state_feature_name in list_state_features}
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        # dic_reward["sum_lane_queue_length"] = None
        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"]))
        dic_reward["queue_length"] = np.absolute(np.sum(self.dic_feature["lane_num_waiting_vehicle_in"]))
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                # print(dic_reward_info[r])
                reward += dic_reward_info[r] * dic_reward[r]
                # print(reward, dic_reward_info[r], dic_reward[r])
        return reward

    def _orgnize_several_segments2(self):
        part1, part2, part3, part4 = self._get_several_segments(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                                vehicle_distance=self.dic_vehicle_distance_current_step,
                                                                vehicle_speed=self.dic_vehicle_speed_current_step,
                                                                lane_length=self.lane_length,
                                                                list_lanes=self.list_lanes)
        num_in_part1 = [len(part1[lane]) for lane in self.list_entering_lanes]
        num_in_part2 = [len(part2[lane]) for lane in self.list_entering_lanes]
        num_in_part3 = [len(part3[lane]) for lane in self.list_entering_lanes]
        num_in_part4 = [len(part4[lane]) for lane in self.list_entering_lanes]

        total_in = []
        for i in range(len(self.list_entering_lanes)):
            total_in.extend([num_in_part1[i], num_in_part2[i], num_in_part3[i], num_in_part4[i]])

        return total_in

    def _get_several_segments(self, lane_vehicles, vehicle_distance, vehicle_speed,
                              lane_length, list_lanes):
        # get four segments [100, 200, 300, 400] for segment
        obs_length = 100
        part1, part2, part3, part4 = {}, {}, {}, {}
        for lane in list_lanes:
            part1[lane], part2[lane], part3[lane], part4[lane] = [], [], [], []
            for vehicle in lane_vehicles[lane]:
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                    continue
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance > lane_length[lane] - obs_length:
                    part1[lane].append(vehicle)
                    #  running vehicles
                    # if vehicle_speed[vehicle] > 0.1:
                    #     part1[lane].append(vehicle)
                elif lane_length[lane] - 2 * obs_length < temp_v_distance <= lane_length[lane] - obs_length:
                    part2[lane].append(vehicle)
                elif lane_length[lane] - 3 * obs_length < temp_v_distance <= lane_length[lane] - 2 * obs_length:
                    part3[lane].append(vehicle)
                elif lane_length[lane] - 4 * obs_length < temp_v_distance <= lane_length[lane] - 3 * obs_length:
                    part4[lane].append(vehicle)
        return part1, part2, part3, part4
    


class CityFlowEnv:

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.intersection_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None
        self.waiting_vehicle_list = {}


        # The evaluator checks for a new decision on a fixed tick.  Jev uses
        # five seconds; legacy DynamicLight falls back to MIN_ACTION_TIME1.
        self.decision_interval = int(self.dic_traffic_env_conf.get(
            "DECISION_INTERVAL",
            self.dic_traffic_env_conf.get("MIN_ACTION_TIME1", self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 5)),
        ))
        if self.decision_interval < self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("DECISION_INTERVAL should be at least YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()


        ##################
        self.duration=[]   # 每个路口还需要执行的秒数
        self.actions=[]   # 每个路口当前采取的signal
        # ================

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")
        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": int(np.random.randint(0, 100)),
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": True,  # if "GPT" in self.dic_traffic_env_conf["MODEL_NAME"] or "llm" in self.dic_traffic_env_conf["MODEL_NAME"] else False,
            "roadnetLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL_NAME']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-roadnetLogFile.json",
            "replayLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL_NAME']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-replayLogFile.txt"
        }
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # get lane length
        _, self.lane_length = self.get_lane_length()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                               self.path_to_log,
                                               self.lane_length)
                                  for i in range(self.dic_traffic_env_conf["NUM_COL"])
                                  for j in range(self.dic_traffic_env_conf["NUM_ROW"])]
        self.list_inter_log = [[] for _ in range(self.dic_traffic_env_conf["NUM_COL"] *
                                                 self.dic_traffic_env_conf["NUM_ROW"])]
        
        
        ###############################################################################
        self.actions=['1']*len(self.list_intersection)
        self.duration=[0]*len(self.list_intersection)
        # ==============================================================================



        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_COL"]):
            for j in range(self.dic_traffic_env_conf["NUM_ROW"]):
                self.id_to_index["intersection_{0}_{1}".format(i+1, j+1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }
        self.list_memory = [[[], [], [], []] for _ in range(len(self.list_intersection))]
        # create roadnet dict
        if self.intersection_dict is None:
            self.create_intersection_dict()

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        state, done = self.get_state()

        return state
    
    def create_intersection_dict(self):
        roadnet = load_json(f'./{self.dic_path["PATH_TO_DATA"]}/{self.dic_traffic_env_conf["ROADNET_FILE"]}')

        intersections_raw = roadnet["intersections"]
        roads_raw = roadnet["roads"]

        agent_intersections = {}

        # init agent intersections
        for i, inter in enumerate(intersections_raw):
            inter_id = inter["id"]
            intersection = None
            for env_inter in self.list_intersection:
                if env_inter.inter_name == inter_id:
                    intersection = env_inter
                    break

            if len(inter['roadLinks']) > 0:
                # collect yellow allowed road links
                yellow_time = None
                phases = inter['trafficLight']['lightphases']
                all_sets = []
                yellow_phase_idx = None
                for p_i, p in enumerate(phases):
                    all_sets.append(set(p['availableRoadLinks']))
                    if p["time"] < 30:
                        yellow_phase_idx = p_i
                        yellow_time = p["time"]
                yellow_allowed_links = reduce(lambda x, y: x & y, all_sets)

                # init intersection
                agent_intersections[inter_id] = {"phases": {"Y": {"time": yellow_time, "idx": yellow_phase_idx}},
                                                 "roads": {}}

                # init roads
                roads = {}
                for r in inter["roads"]:
                    roads[r] = {"location": None, "type": "incoming", "go_straight": None, "turn_left": None,
                                "turn_right": None, "length": None, "max_speed": None,
                                "lanes": {"go_straight": [], "turn_left": [], "turn_right": []}}

                # collect road length speed info & init road location
                road_links = inter["roadLinks"]
                for r in roads_raw:
                    r_id = r["id"]
                    if r_id in roads:
                        roads[r_id]["length"] = calculate_road_length(r["points"])
                        roads[r_id]["max_speed"] = r["lanes"][0]["maxSpeed"]
                        for env_road_location in intersection.dic_entering_approach_to_edge:
                            if intersection.dic_entering_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break
                        for env_road_location in intersection.dic_exiting_approach_to_edge:
                            if intersection.dic_exiting_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break

                # collect signal phase info
                for p_idx, p in enumerate(phases):
                    other_allowed_links = set(p['availableRoadLinks']) - yellow_allowed_links
                    if len(other_allowed_links) > 0:
                        allowed_directions = []
                        for l_idx in other_allowed_links:
                            link = road_links[l_idx]
                            location = roads[link["startRoad"]]["location"]
                            direction = link["type"]
                            allowed_directions.append(f"{location_dict[location]}{direction_dict[direction]}")
                        allowed_directions = sorted(allowed_directions)
                        allowed_directions = f"{allowed_directions[0]}{allowed_directions[1]}"
                        agent_intersections[inter_id]["phases"][allowed_directions] = {"time": p["time"], "idx": p_idx}

                # collect location type direction info
                for r_link in road_links:
                    start = r_link['startRoad']
                    end = r_link['endRoad']
                    lane_links = r_link['laneLinks']

                    for r in roads:
                        if r != start:
                            continue
                        # collect type
                        roads[r]["type"] = "outgoing"

                        # collect directions
                        if r_link["type"] == "go_straight":
                            roads[r]["go_straight"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["go_straight"]:
                                    roads[r]["lanes"]["go_straight"].append(lane_id)

                        elif r_link["type"] == "turn_left":
                            roads[r]["turn_left"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_left"]:
                                    roads[r]["lanes"]["turn_left"].append(lane_id)

                        elif r_link["type"] == "turn_right":
                            roads[r]["turn_right"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_right"]:
                                    roads[r]["lanes"]["turn_right"].append(lane_id)

                agent_intersections[inter_id]["roads"] = roads

        self.intersection_dict = agent_intersections

    def step(self, action, duration=None):
        '''
        action:[-1,-1,-1,0,-1,3]
        无需修改的路口设置为-1
        duration同理
        '''
        # 处理action与duration 记录状态
        ###############################################################
        # 填充剩余action
        before_action_state,_=self.get_state()
        for i in range(len(action)):
            if action[i]==-1:
                action[i]=self.actions[i]
            else:
                self.list_memory[i][0] = action[i]
                self.list_memory[i][2] = before_action_state[i]

        # 填充duration
        if duration is None:
            duration=[self.decision_interval]*len(action)
        else:
            for i in range(len(duration)):
                if duration[i]!=-1:
                    self.list_memory[i][1] = duration[i]
                    duration[i]=self.dic_traffic_env_conf['ACTION_DURATION'][duration[i]]
                    if (
                        self.dic_traffic_env_conf.get("DURATION_EXCLUDES_YELLOW")
                        and int(action[i]) + 1 != self.list_intersection[i].current_phase_index
                    ):
                        duration[i] += self.dic_traffic_env_conf["YELLOW_TIME"]
                else:
                    duration[i]=self.duration[i]

        # 源代码
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.decision_interval - 1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())
        # ===============================================================


        step_start_time = time.time()
        
        # 执行action
        average_reward_action_list = [0]*len(action)
        for i in range(self.decision_interval):

            # adnvance colight需要的日志信息
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]
            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()
            before_action_feature = self.get_feature()



            if i == 0:
                print("time: {0}".format(instant_time))

            # 执行1秒
            self._inner_step(action)


            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            
            tmp_reward=self.get_reward()
            for inter_id in range(len(self.list_intersection)):
                self.list_memory[inter_id][3].append(tmp_reward[inter_id])
            
            # 记录advanced colight日志
            if self.dic_traffic_env_conf['MODEL_NAME']=='AdvancedColight':
                self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

           
        print("Step time: ", time.time() - step_start_time)
        # 更新信息 记录日志
        ########################################################################
        # 更新action和duration
        self.actions=action
        self.duration=duration
       
        # 更新时间
        for i in range(len(action)):
            self.duration[i]-=self.decision_interval
       
        # get reward
        list_need_action=[]
        for i in range(len(self.duration)):
            if self.duration[i]==0:
                list_need_action.append(i)
        self.list_need_action=list_need_action
        reward = self.get_reward_dy(list_need_action)

        # get state
        next_state, done = self.get_state(list_need_action=list_need_action)
        if self.dic_traffic_env_conf['MODEL_NAME']=='DynamicLight':
            self.log_dy(after_action_state=next_state,
                    final_reward=reward,
                    )
        # ======================================================================

        next_state,_=self.get_state()
        return next_state, average_reward_action_list, done, average_reward_action_list

    # 
    def get_reward_dy(self,list_need_action):
        list_reward = [self.list_intersection[idx].get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for
                       idx in list_need_action]
        return list_reward

    def get_duration(self):
        return self.duration
    

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

            # update queuing vehicle info
            vehicle_ids = self.eng.get_vehicles(include_waiting=False)
            for v_id in vehicle_ids:
                v_info = self.eng.get_vehicle_info(v_id)
                speed = float(v_info["speed"])
                if speed < 0.1:
                    if v_id not in self.waiting_vehicle_list:
                        self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                        self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                        self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                    else:
                        if self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                            self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                            self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                            self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                        else:
                            self.waiting_vehicle_list[v_id]["time"] += self.dic_traffic_env_conf["INTERVAL"]
                else:
                    if v_id in self.waiting_vehicle_list:
                        self.waiting_vehicle_list.pop(v_id)

                if v_id in self.waiting_vehicle_list and self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                    self.waiting_vehicle_list.pop(v_id)

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self,list_need_action=[0,1,2,3,4,5,6,7,8,9,10,11]):
        if self.dic_traffic_env_conf['MODEL_NAME']=='DynamicLight':
            dic=['lane_num_waiting_vehicle_in','lane_enter_running_part']

            list_state = [self.list_intersection[idx].get_state(dic)
                        for idx in list_need_action]
            state=[]
            for i in range(len(list_need_action)):
                map={
                    1: [0,1,1,0,1,1,0,0,1,0,0,1,   0,0,0,0],
                    2: [0,0,1,0,0,1,0,1,1,0,1,1,   0,0,0,0],
                    3: [1,0,1,1,0,1,0,0,1,0,0,1,   0,0,0,0],
                    4: [0,0,1,0,0,1,1,0,1,1,0,1,   0,0,0,0]
                }
                tem={
                    'phase_total':map[int(self.actions[list_need_action[i]])+1],
                    'lane_run_in_part':list_state[i]['lane_enter_running_part'],
                    'lane_queue_vehicle_in':list_state[i]['lane_num_waiting_vehicle_in'],
                    'num_in_deg':self.list_intersection[list_need_action[i]]._orgnize_several_segments2(),
                }
                tem=copy.deepcopy(tem)
                tem['lane_run_in_part']+=[0,0,0,0]
                tem['lane_queue_vehicle_in']+= [0,0,0,0]
                tem['num_in_deg']+=[0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0]
                state.append(tem)
            return state,False
            
        else :
            list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
            return list_state , False
    

    def get_reward(self,state=None):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):
        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})
    

    def log_dy(self, after_action_state, final_reward):
        # [cur-state, phase-action, duration-action, next-state, final-reward, average_reward]
        for i, inter_ind in enumerate(self.list_need_action):
            # action1 = self.list_memory[inter_ind][0]
            average_reward = np.mean(self.list_memory[inter_ind][3])
            self.list_memory[inter_ind][3] = []
            self.list_inter_log[inter_ind].append([self.select_phase_feature(self.list_memory[inter_ind][2]),
                                                   self.list_memory[inter_ind][0],  # phase action
                                                   self.list_memory[inter_ind][1],  # duration action
                                                   self.select_phase_feature(after_action_state[i]),
                                                   final_reward[i],
                                                   average_reward
                                                   ])
    def select_phase_feature(self, state):
        """
        state: [None, 12]
        return: [None, 2]
        """
        # feat = state[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1]]
        return state
    
    def batch_log_dy(self, start, stop):
        """
        only log inter_{}.pkl
        """
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            # path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            # dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            # df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            # df.to_csv(path_to_log_file, na_rep="nan")
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()


    
    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
            
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log_dy,args=(start,stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for inter in net["intersections"]:
                if not inter["virtual"]:
                    traffic_light_node_dict[inter["id"]] = {"location": {"x": float(inter["point"]["x"]),
                                                                         "y": float(inter["point"]["y"])},
                                                            "total_inter_num": None, "adjacency_row": None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net["roads"]:
                if road["id"] not in edge_id_dict.keys():
                    edge_id_dict[road["id"]] = {}
                edge_id_dict[road["id"]]["from"] = road["startIntersection"]
                edge_id_dict[road["id"]]["to"] = road["endIntersection"]

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]["location"]

                row = np.array([0]*total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]["location"]
                    dist = self._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]["adjacency_row"] = [inter_id_to_index[i]]+adjacency_row_unsorted
                traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
                traffic_light_node_dict[i]["neighbor_ENWS"] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    if edge_id_dict[road_id]["to"] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                    else:
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(edge_id_dict[road_id]["to"])

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for i in range(3):
                lane_id = road['id'] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict
