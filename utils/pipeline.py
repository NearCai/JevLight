from .generator import Generator
from .construct_sample import ConstructSample,AConstructSample
from .updater import Updater
import json
import shutil
import os
import time
from multiprocessing import Process
import wandb
import copy

os.environ['NCCL_DEBUG'] = 'INFO'


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf, logger,scale=1.0):
    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,

                          )
    print("make generator")
    generator.generate(logger, is_=False, scale=scale)
    print("generator_wrapper end")
    return

def updater_wrapper(cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):
    updater = Updater(
        cnt_round=cnt_round,
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    # 加载样例
    updater.load_sample_for_agents()
    # 更新网络
    updater.update_network_for_agents()
    print("updater_wrapper end")
    return


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.roadnet = roadnet
        self.trafficflow = trafficflow

        self.initialize()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self, round, multi_process=False):
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()


        scale="no_busy_score"
        # wandb init
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        run_label = self.dic_traffic_env_conf.get(
            "LABEL",
            f"{self.dic_traffic_env_conf.get('MODEL_NAME', 'model')}-round_{round}",
        )
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL']}-{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=str(run_label),
            config=all_config,
        )


        for cnt_round in range(self.dic_traffic_env_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()
            process_list = []
            # 生成数据
            print("==============  generator =============")
            generator_start_time = time.time()
            if multi_process:
                print("-------------- use multi-process for generator -------------")
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    p = Process(target=generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, logger)
                                )
                    print("before")
                    p.start()
                    print("end")
                    process_list.append(p)
                print("before join")
                for i in range(len(process_list)):
                    p = process_list[i]
                    print("generator %d to join" % i)
                    p.join()
                    print("generator %d finish join" % i)
                print("end join")
            else:
                for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                    generator_wrapper(cnt_round=cnt_round,
                                      cnt_gen=cnt_gen,
                                      dic_path=self.dic_path,
                                      dic_agent_conf=self.dic_agent_conf,
                                      dic_traffic_env_conf=self.dic_traffic_env_conf,
                                      logger=logger,scale=scale)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time


            # 处理样例,更改样例中的数据格式
            print("==============  make samples =============")
            # make samples and determine which samples are good
            making_samples_start_time = time.time()
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(train_round):
                os.makedirs(train_round)
            if self.dic_traffic_env_conf['MODEL_NAME']=='DynamicLight':
                cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
                cs.prepare_samples_for_system()
            if self.dic_traffic_env_conf['MODEL_NAME']=='AdvancedColight':
                cs = AConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                    dic_traffic_env_conf=self.dic_traffic_env_conf)
                cs.make_reward_for_system()
            making_samples_end_time = time.time()
            making_samples_total_time = making_samples_end_time - making_samples_start_time

            # 更新网络
            print("==============  update network =============")
            update_network_start_time = time.time()
            
            if multi_process:
                p = Process(target=updater_wrapper,
                            args=(cnt_round,
                                    self.dic_agent_conf,
                                    self.dic_traffic_env_conf,
                                    self.dic_path))
                p.start()
                print("update to join")
                p.join()
                print("update finish join")
            else:
                updater_wrapper(cnt_round=cnt_round,
                                dic_agent_conf=self.dic_agent_conf,
                                dic_traffic_env_conf=self.dic_traffic_env_conf,
                                dic_path=self.dic_path)

            update_network_end_time = time.time()
            update_network_total_time = update_network_end_time - update_network_start_time

            # 测试模型
        #     print("==============  test evaluation =============")
        #     test_evaluation_start_time = time.time()
        #     results, state_action_log = model_test.test(self.dic_path["PATH_TO_MODEL"], self.dic_path["PATH_TO_DATA"],
        #                                                 cnt_round,
        #                                                 self.dic_traffic_env_conf["RUN_COUNTS"],
        #                                                 self.dic_traffic_env_conf, logger)
        #     if cnt_round + 1 > self.dic_traffic_env_conf["NUM_ROUNDS"] - 10:
        #         for ele in last_10_results:
        #             last_10_results[ele].append(results[ele])

        #     test_evaluation_end_time = time.time()
        #     test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time

        #     # 总结
        #     print("Generator time: ", generator_total_time)
        #     print("Making samples time:", making_samples_total_time)
        #     print("update_network time:", update_network_total_time)
        #     print("test_evaluation time:", test_evaluation_total_time)

        #     print("round {0} ends, total_time: {1}".format(cnt_round, time.time() - round_start_time))
        #     f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "a")
        #     f_time.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(generator_total_time, making_samples_total_time,
        #                                                     update_network_total_time, test_evaluation_total_time,
        #                                                     time.time() - round_start_time))
        #     f_time.close()

        # last_10_results = {ele: np.mean(last_10_results[ele]) for ele in last_10_results}
        # logger.log(last_10_results)
        # print(last_10_results)
        # f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        # dump_json(state_action_log, f_state_action)
        wandb.finish()

        return 1
