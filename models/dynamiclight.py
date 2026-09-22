"""
DynamicLight under feature fusion method 1
Input shape: [batch, max_lane*4]
Created by Liang Zhang
"""
from tensorflow.keras.layers import Input, Dense, Reshape, Lambda, Subtract, Add, MultiHeadAttention, Activation, Embedding, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent_dynamiclight import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import copy, os


class DynamicLightAgent(NetworkAgent):
    
    def build_network(self):
        ins0 = Input(shape=(self.max_lane, self.num_feat1), name="input_total_features") 
        ins1 = Input(shape=(self.max_lane,), name="input_cur_phase")
        ins2 = Input(shape=(1, self.num_phases))
        uni_dim, dims2 = 4, 16
        phase_emb = Activation('sigmoid')(Embedding(2, uni_dim, input_length=self.max_lane)(ins1))
        feat_list = [Lambda(lambda x, index=i: x[:, :, index:index + 1], name=f"feature_slice_{i}")(ins0)
                     for i in range(self.num_feat1)]
        feat_embs = [Dense(uni_dim, activation='sigmoid')(feat_list[i]) for i in range(self.num_feat1)]
        feat_embs.append(phase_emb)    
        feats = Concatenate(axis=2)(feat_embs)
        feats = Dense(dims2, activation="relu")(feats)
        lane_feats = [Lambda(lambda x, index=i: x[:, index:index + 1, :], name=f"lane_slice_{i}")(feats)
                      for i in range(self.max_lane)]
        phase_feats = []
        MHA1 = MultiHeadAttention(4, 16, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        for i in range(self.num_phases):
            tmp_feat_1 = Concatenate(axis=1)([lane_feats[idx] for idx in self.phase_map[i]])
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_2)
            phase_feats.append(tmp_feat_3)
        phase_feat_all = Concatenate(axis=1)(phase_feats)
        att_encoding = MultiHeadAttention(4, 8, attention_axes=1)(phase_feat_all, phase_feat_all)
        hidden = Dense(20, activation="relu")(att_encoding)
        hidden = Dense(20, activation="relu")(hidden)
        phase_feature_final = Dense(1, activation="linear", name="beformerge")(hidden)
        pscore = Reshape((4,))(phase_feature_final)
        selected_phase_feat = Dot(axes=(2, 1))([ins2, phase_feat_all])
        selected_phase_feat = Reshape((dims2, ))(selected_phase_feat)
        hidden2 = Dense(20, activation="relu")(selected_phase_feat)
        hidden2 = Dense(20, activation="relu")(hidden2)
        dscore = self.dueling_block(hidden2)
        network = Model(inputs=[ins0, ins1, ins2],
                        outputs=[pscore, dscore])
        network.compile(optimizer=Adam(learning_rate=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()
        return network
    
    def freze_layers(self):
        for layer in self.q_network.layers[0:21]:
            layer.trainable = False
        self.q_network.compile(optimizer=Adam(learning_rate=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])

    def unfreze_layers(self):
        for layer in self.q_network.layers[0:21]:
            layer.trainable = True
        self.q_network.compile(optimizer=Adam(learning_rate=self.dic_agent_conf["LEARNING_RATE2"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])

    def resetlr(self):
        self.q_network.compile(optimizer=Adam(learning_rate=self.dic_agent_conf["LEARNING_RATE3"], epsilon=1e-08),
                               loss=self.dic_agent_conf["LOSS_FUNCTION"])
    
    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        val_spl = 0.2
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')
        batch_size = min(self.dic_agent_conf["BATCH_SIZE1"], len(self.Ys[0]))
        self.q_network.fit(self.Xs, self.Ys, batch_size=batch_size, epochs=epochs, shuffle=1,
                           verbose=2, validation_split=val_spl, callbacks=[early_stopping])
        print("=== training two model ===")

    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        a = Dense(self.num_action_dur, activation="linear", name="dueling_advantages")(tmp_a)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantages = Subtract()([a, mean])
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values
    
    # 选择阶段
    def choose_action(self, states):
        dic_state_feature_arrays = {}
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE_1"])
        cur_phase = []
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in used_feature:
                if feature_name == "phase_total":
                    cur_phase.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        used_feature.remove("phase_total")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), self.max_lane, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        
        # phase action 
        tmp_p, _ = self.q_network.predict([state_input, np.array(cur_phase), np.random.rand(len(states),1, 4)])
        if self.cnt_round < self.dic_agent_conf["SROUND"]:
            paction = self.epsilon_choice_one(tmp_p)
        else:
            paction = np.argmax(tmp_p, axis=1)
        # duration action
        phase_idx =  np.array(paction).reshape(len(paction), 1, 1)
        phase_matrix = self.phase_index2matrix(phase_idx)
        _, tmp_d = self.q_network.predict([state_input, np.array(cur_phase), phase_matrix])
        if self.cnt_round < self.dic_agent_conf["SROUND"]:
            daction = [1] * len(states)
        elif self.cnt_round < self.dic_agent_conf["SROUND2"]:
            daction = self.epsilon_choice_two(tmp_d)
        else:
            daction = np.argmax(tmp_d, axis=1)
        return paction, daction

    def phase_index2matrix(self, phase_index):
        # [batch, 1] -> [batch, 1, num_phase]
        lab = to_categorical(phase_index, num_classes=self.num_phases)
        return lab
    
    def epsilon_choice_one(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_phases, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act
    
    def epsilon_choice_two(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_action_dur, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act
    
    # 更新q网络
    def prepare_Xs_Y(self, memory):
        """ used for update phase control model """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))
        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE1"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE_1"][1:]
        _phase1, _phase2 = [], []
        _state = [[] for _ in used_feature]
        _next_state = [[] for _ in used_feature]
        _action1 = []
        _action2 = []
        _action3 = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action1, action2, next_state, reward, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _phase1.append(state["phase_total"])
            _phase2.append(next_state["phase_total"])
            _action1.append([[action1]])
            _action3.append(action1)
            _action2.append(action2)
            _reward.append(reward)
        _state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _state], axis=-1)
        _next_state2 = np.concatenate([np.array(ss).reshape(len(ss), self.max_lane, -1) for ss in _next_state], axis=-1)
        phase_matrix = self.phase_index2matrix(np.array(_action1))
        
        cur_p, cur_d = self.q_network.predict([_state2, np.array(_phase1), phase_matrix])
        next_p, next_d = self.q_network_bar.predict([_next_state2, np.array(_phase2), phase_matrix])
        
        target1 = np.copy(cur_p)
        target2 = np.copy(cur_d)
        if self.cnt_round < self.dic_agent_conf["SROUND"]:
            for i in range(len(sample_slice)):
                target1[i, _action3[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_p[i, :])
        elif self.cnt_round < self.dic_agent_conf["SROUND2"]:
            for i in range(len(sample_slice)):
                target2[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_d[i, :])
        else:
            for i in range(len(sample_slice)):
                target1[i, _action3[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] *np.max(next_p[i, :])
            for i in range(len(sample_slice)):
                target2[i, _action2[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * np.max(next_d[i, :])
            
        self.Xs = [_state2, np.array(_phase1), phase_matrix]
        self.Ys = [target1, target2]

    # 保存权重文件
    def save_network(self, file_name):
        if self.cnt_round == self.dic_agent_conf["SROUND"]-1:
            self.freze_layers()
        if self.cnt_round == self.dic_agent_conf["SROUND2"]-1:
            self.unfreze_layers()
        if self.cnt_round == self.dic_agent_conf["SROUND3"]-1:
            self.resetlr()
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
