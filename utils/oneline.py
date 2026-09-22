from .config import DIC_AGENTS
from .my_utils import merge, get_state, get_state_detail, eight_phase_list, dump_json
from copy import deepcopy
from .cityflow_env import CityFlowEnv
from .pipeline import path_check, copy_cityflow_file, copy_conf_file
import os
import time
import numpy as np
import wandb
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor

class OneLine:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.initialize()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()

        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            if "ChatGPT" in agent_name:
                agent = DIC_AGENTS[agent_name.split("-")[0]](
                    GPT_version=self.dic_agent_conf["GPT_VERSION"],
                    intersection=self.env.intersection_dict[self.env.list_intersection[i].inter_name],
                    inter_name=self.env.list_intersection[i].inter_name,
                    phase_num=len(self.env.list_intersection[i].list_phases),
                    log_dir=self.dic_agent_conf["LOG_DIR"],
                    dataset=f"{self.roadnet}-{self.trafficflow}"
                )
            elif agent_name == "Jev":
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    intersection=self.env.intersection_dict[self.env.list_intersection[i].inter_name],
                    inter_name=self.env.list_intersection[i].inter_name,
                    phase_num=len(self.env.list_intersection[i].list_phases),
                )
            elif "open_llm" in agent_name:
                agent = DIC_AGENTS[agent_name.split("-")[0]](
                    ex_api=self.dic_agent_conf["WITH_EXTERNAL_API"],
                    model=agent_name.split("-")[1],
                    intersection=self.env.intersection_dict[self.env.list_intersection[i].inter_name],
                    inter_name=self.env.list_intersection[i].inter_name,
                    phase_num=len(self.env.list_intersection[i].list_phases),
                    log_dir=self.dic_agent_conf["LOG_DIR"],
                    dataset=f"{self.roadnet}-{self.trafficflow}"
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=0,
                    intersection_id=str(i)
                )
            self.agents.append(agent)

    def train(self, round):
        print("================ start train ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        file_name_memory = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "memories.txt")
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        step_num = 0
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"round_{round}",
            config=all_config,
        )

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]
        is_jev = self.dic_traffic_env_conf["MODEL_NAME"] == "Jev"
        is_shared_network = self.dic_traffic_env_conf["MODEL_NAME"] in {
            "PressLight", "MPLight", "Colight", "EfficientPressLight",
            "EfficientMPLight", "EfficientColight", "DynamicLight",
        }
        while not done and current_time < total_run_cnt:
            action_list = [-1] * len(state) if is_jev else []
            duration_list = (
                [-1] * len(state)
                if is_jev or self.dic_traffic_env_conf["MODEL_NAME"] == "DynamicLight"
                else None
            )
            threads = []
            jev_due = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming, "approaching_speed": mean_speed})

                one_state = state[i]
                count = step_num
                if is_jev:
                    # CityFlow checks the remaining duration every 5 seconds;
                    # only intersections at a decision node call Jev.
                    if self.env.duration[i] <= 0:
                        jev_due.append((i, one_state, count))
                    else:
                        state_action_log[i][-1]["duration"] = None

                elif is_shared_network:
                    # These agents consume the complete intersection batch and
                    # return one action per intersection; call agent 0 once
                    # below instead of passing a single state dict.
                    continue
                elif "ChatGPT" in self.dic_traffic_env_conf["MODEL_NAME"] or "open_llm" in self.dic_traffic_env_conf["MODEL_NAME"]:
                    thread = threading.Thread(target=self.agents[i].choose_action, args=(self.env,))
                    threads.append(thread)
                else:
                    action = self.agents[i].choose_action(count, one_state)
                    action_list.append(action)

            # Jev requests are independent across intersections. Sending them
            # concurrently avoids turning a one-hour simulation into a many-
            # hour serial HTTP benchmark while preserving per-agent ordering
            # (phase request, then duration request).
            if is_jev and jev_due:
                max_workers = int(self.dic_traffic_env_conf.get("JEV_MAX_CONCURRENCY", len(jev_due)))
                max_workers = max(1, min(max_workers, len(jev_due)))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        i: pool.submit(self.agents[i].choose_action, self.env, one_state, due_count)
                        for i, one_state, due_count in jev_due
                    }
                    for i, _, _ in jev_due:
                        action, duration = futures[i].result()
                        action_list[i] = action
                        duration_list[i] = duration
                        state_action_log[i][-1]["duration"] = self.agents[i].duration_options[duration]
                        state_action_log[i][-1]["decision_source"] = (
                            "jev" if self.agents[i].last_response is not None else "fallback"
                        )
                        if self.agents[i].last_error:
                            state_action_log[i][-1]["decision_error"] = self.agents[i].last_error
                        if self.agents[i].last_guardrail:
                            state_action_log[i][-1]["guardrail"] = self.agents[i].last_guardrail
                        if self.agents[i].last_exchange:
                            state_action_log[i][-1]["jev_exchange"] = self.agents[i].last_exchange

            if is_shared_network:
                if self.dic_traffic_env_conf["MODEL_NAME"] == "DynamicLight":
                    shared_actions, shared_durations = self.agents[0].choose_action(state)
                    action_list.extend(np.asarray(shared_actions, dtype=int).reshape(-1).tolist())
                    duration_list[:] = np.asarray(shared_durations, dtype=int).reshape(-1).tolist()
                else:
                    shared_actions = self.agents[0].choose_action(step_num, state)
                    action_list.extend(np.asarray(shared_actions, dtype=int).reshape(-1).tolist())

            # multi-thread
            if "ChatGPT" in self.dic_traffic_env_conf["MODEL_NAME"]:
                for thread in threads:
                    thread.start()

                for thread in tqdm(threads):
                    thread.join()

                for i in range(len(state)):
                    action = self.agents[i].temp_action_logger
                    action_list.append(action)
                    state_action_log[i][-1]["decision_source"] = (
                        "chatgpt" if self.agents[i].fallback_count == 0 else "fallback"
                    )
                    if self.agents[i].errors:
                        state_action_log[i][-1]["decision_error"] = self.agents[i].errors[-1]["error"]

            # multi-thread
            if "open_llm" in self.dic_traffic_env_conf["MODEL_NAME"]:
                started_thread_id = []
                thread_num = self.dic_traffic_env_conf["LLM_API_THREAD_NUM"] if not self.dic_agent_conf["WITH_EXTERNAL_API"] else 2
                for i, thread in enumerate(tqdm(threads)):
                    thread.start()
                    started_thread_id.append(i)

                    if (i + 1) % thread_num == 0:
                        for t_id in started_thread_id:
                            threads[t_id].join()
                        started_thread_id = []

                for i in range(len(state)):
                    action = self.agents[i].temp_action_logger
                    action_list.append(action)

            if is_jev:
                next_state, reward, done, _ = self.env.step(action_list, duration_list)
            else:
                next_state, reward, done, _ = self.env.step(action_list)

            # log action
            for i in range(len(state)):
                if is_jev:
                    state_action_log[i][-1]["action"] = (
                        f"Phase-{action_list[i] + 1}" if action_list[i] >= 0 else None
                    )
                else:
                    state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            f_memory = open(file_name_memory, "a")
            # output to std out and file
            current_phases = [
                state[i].get("cur_phase", [self.env.actions[i]])[0]
                for i in range(len(state))
            ]
            memory_str = 'time = {0}\taction = {1}\tcurrent_phase = {2}\treward = {3}'.\
                format(current_time, str(action_list), str(current_phases),
                       str(reward),)
            f_memory.write(memory_str + "\n")
            f_memory.close()
            current_time = self.env.get_current_time()  # in seconds

            state = next_state
            step_num += 1

            # calculate logger results
            total_reward += sum(reward)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        logger.log(results)
        print(results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        wandb.finish()

        print("Training time: ", time.time()-start_time)

        self.env.batch_log_2()

        return results
