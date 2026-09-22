from .config import DIC_AGENTS
from .cityflow_env import CityFlowEnv
import time
import os
import copy
import numpy as np
import pickle
from copy import deepcopy
from utils.my_utils import get_state_detail, getPrompt, state2text, action2code, four_phase_list
import re
from tqdm import tqdm
import json

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                with open('t.json','w') as f:
                    json.dump(dic_traffic_env_conf,f,ensure_ascii=False,indent=4)
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=dic_path
        )

    
    def generate(self, logger,is_=True,scale=0.2):
        # test=[]
        # for i in range(10000):
        #     test.extend([0,1,2,3])


        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []

        for step_num in tqdm(range(int(self.dic_traffic_env_conf["RUN_COUNTS"] / self.dic_traffic_env_conf['MIN_ACTION_TIME1']))):
            if done:
                break

            action_list = []
            step_start_time = time.time()
            list_need=[]
            ####################################################################
            # 选择策略
            if self.dic_traffic_env_conf['MODEL_NAME']=="DynamicLight":
                # 先获取哪些路口需要action
                for i in range(len(self.env.duration)):
                    if self.env.duration[i]<=0:
                        list_need.append(i)
                if len(list_need)==0:
                    action_list=[-1]*12
                    duration_list=[-1]*12
                else:

                    '''
                    {
                        'phase_total': [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], 
                        'lane_queue_vehicle_in': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        'lane_run_in_part': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        'num_in_deg': [0]*64
                    }
                    '''
                    '''
                    {   
                        'cur_phase': [1], 
                        'traffic_movement_pressure_queue_efficient': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                        'lane_enter_running_part': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        'adjacency_matrix': [0, 3, 1, 6, 4]
                    }
                    '''

                    one_state=[]
                    # 只需要需要修改action的路口的state
                    for i in list_need:
                        one_state.append(state[i])


                    # 获取 action 与 duration
                    action,duration = self.agents[0].choose_action(one_state)
                    
                    
                    # 转变action和duration的格式,将无需改变的填充为-1
                    flag=0
                    action_list =[-1]*len(self.env.duration)
                    duration_list=[-1]*len(action_list)

                    for i in range(len(self.env.duration)):
                        if i in list_need:
                            action_list[i]=action[flag]
                            duration_list[i]=duration[flag]
                            flag+=1
            elif self.dic_traffic_env_conf['MODEL_NAME']=="AdvancedColight":
                one_state = state
                action = self.agents[0].choose_action(step_num, one_state)
                action_list = action    
                duration_list=None    
            # debug用
            # if len(list_need)==0:
            #     action_list,duration_list = [-1]*12,[-1]*12
            # else:
            #     # action_list,duration_list = [0]*12,[0]*12
            #                 # debug用
            #     action_list=[-1]*12
            #     duration_list=[-1]*12
            #     for i in list_need:
            #         action_list[i]=test.pop()
            #     for i in list_need:
            #         duration_list[i]=test.pop()
            next_state, reward, done, _ = self.env.step(action_list,duration_list)
            # ============================================================================================= # 
            # 在reward中添加busy score
            if is_:
                current_states=[]
                for i in range(len(next_state)):
                    intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                    roads = deepcopy(intersection["roads"])
                    statistic_state, _, _ = get_state_detail(roads, self.env)
                    current_states.append(statistic_state)
                for i in range(len(current_states)):
                    reward[i]+=-scale*self.get_intersection_busy_score(i,current_states)
                print(f'\033[1;31;40系数为：{scale}\033[0m')
            else:
                pass
            # ============================================================================================= # 

            print("time: {0}, running_time: {1}".format(self.env.get_current_time() -
                                                        self.dic_traffic_env_conf["MIN_ACTION_TIME1"],
                                                        time.time()-step_start_time))
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

            state = next_state
        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time) and not np.isnan(leave_time):
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "training_reward": total_reward,
            "training_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_travel_time": total_travel_time}
        logger.log(results)

        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)

    def get_lane_busy_score(self,state):
        busy_score=state['cells'][0]*0.8+state['cells'][1]*0.6+state['cells'][2]*0.4+state['cells'][3]*0.2+state['queue_len']*1
        return busy_score
    
    def get_intersection_busy_score(self,id,all_state,scale=0.1):
        score=0
        for i in all_state[id]:
            score+=self.get_lane_busy_score(all_state[id][i])
        # 获取周围路口
        near_intersection=self.get_near_intersections(self.id2name(id))
        if scale is not None:
            if 'East' in near_intersection:
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['East'])]['ET'])
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['East'])]['SL'])        
            if 'West' in near_intersection:
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['West'])]['WT'])
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['West'])]['NL'])        
            if 'North' in near_intersection:
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['North'])]['NT'])
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['North'])]['EL'])        
            if 'South' in near_intersection:
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['South'])]['ST'])
                score+=scale*self.get_lane_busy_score(all_state[self.name2id(near_intersection['South'])]['WL'])
        return score
    
    # 获取临近路口
    def get_near_intersections(self,name,shape=(4,3)):
        numbers = re.findall(r'\d', name)
        for i in range(len(numbers)):
            numbers[i]=int(numbers[i])

        result={}
        template='intersection_{}_{}'
        if numbers[0]+1<=shape[0]:
            result['East']=template.format(numbers[0]+1,numbers[1])
        
        if numbers[0]-1>0:
            result['West']=template.format(numbers[0]-1,numbers[1])
        
        if numbers[1]-1>0:
            result['North']=template.format(numbers[0],numbers[1]-1)

        if numbers[1]+1<=shape[1]:
            result['South']=template.format(numbers[0],numbers[1]+1)
        return result

    def id2name(self,tid):
        for id,item in enumerate(self.env.list_intersection):
            if id==tid:
                return item.inter_name

    def name2id(self,name):
        for id,item in enumerate(self.env.list_intersection):
            if item.inter_name==name:
                return id


class Generator_LLMLight:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
            start_time = time.time()
            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=self.cnt_round,
                    intersection_id=str(i)
                )
                self.agents[i] = agent
            print("Create intersection agent time: ", time.time()-start_time)

        self.env = CityFlowEnv(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=dic_path
        )

    def generate(self, logger, tokenizer, llm_model, generation_kwargs):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []

        for step_num in tqdm(range(int(self.dic_traffic_env_conf["RUN_COUNTS"] / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done:
                break

            current_states = []
            action_list = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                current_states.append(statistic_state)

            prompts = []
            alpaca_prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                alpaca_prompts.append({"instruction": prompt[1]['content'], "input": "", "output": ""})

                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = tokenizer(prompts, truncation=True, max_length=2048, padding=True, return_tensors='pt').to('cuda')

            response_ids = llm_model.generate(input_ids=inputs["input_ids"], **generation_kwargs)
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            fail_num = 0
            fail_flags = [False for _ in responses]
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "Phase-1"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    if vehicle_nums[i] != 0:
                        fail_num += 1
                        fail_flags[i] = True

            # action_list = []
            step_start_time = time.time()
            # for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
            #
            #     if self.dic_traffic_env_conf["MODEL_NAME"] in ["EfficientPressLight", "EfficientColight",
            #                                                    "EfficientMPLight", "Attend",
            #                                                    "AdvancedMPLight", "AdvancedColight", "AdvancedDQN"]:
            #         one_state = state
            #         action = self.agents[i].choose_action(step_num, one_state)
            #         action_list = action
            #     else:
            #         one_state = state[i]
            #         action = self.agents[i].choose_action(step_num, one_state)
            #         action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)

            print("time: {0}, running_time: {1}, fail_num".format(self.env.get_current_time() -
                                                                  self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                                  time.time()-step_start_time,
                                                                  fail_num))
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

            state = next_state
            step_num += 1
        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time) and not np.isnan(leave_time):
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "training_reward": total_reward,
            "training_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "training_avg_travel_time": total_travel_time}
        logger.log(results)

        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)

    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums
