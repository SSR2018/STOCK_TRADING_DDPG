import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import gym
from gym import spaces
import pymysql
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
class StockTradingEnv(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self,num=0):
        super(StockTradingEnv,self).__init__()
        self.action_space = spaces.Box(low = 0,high=2,shape =(1,),dtype=np.float16)
        self.observation_space = spaces.Box(low = -2,high = 2,shape = (5,19),dtype = np.float16)
        self._db = pymysql.connect('localhost','root','123456','stock')
        self.cursor = self._db.cursor()
        self.stock_list,self.trade_date = self.Stock_List_and_Trade_Date()
        self.ts_code = self.stock_list[num]
    def get_data(self,ts_code = '',start_date = '',limit_num = 5):
        self.cursor.execute('Select * from {} where trade_date_ > {} limit {}'\
                          .format(ts_code,start_date,limit_num))
        data = self.cursor.fetchall()
        data = pd.DataFrame(data)
        data.columns = ['trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', \
                  'pct_chg', 'vol', 'amount', 'ma5', 'ma_v_5', 'ma10', 'ma_v_10', \
                  'volume_ratio', 'pe', 'pb', 'ps', 'ps_ttm']
        return data
    def Stock_List_and_Trade_Date(self):
        self.cursor.execute('SHOW TABLES')
        stock_list =  [i[0] for i in self.cursor.fetchall()]
        self.cursor.execute('select trade_date_ from {}'.format(stock_list[0]))
        trade_date = [i[0] for i in  self.cursor.fetchall()]
        return stock_list,trade_date
    def _next_observation(self,standard = True):
        obs = self.get_data(ts_code=self.ts_code, start_date=self.trade_date[self.current_step])
        if standard:
            obs.iloc[:, 1:6] = StandardScaler().fit_transform(np.log(obs.iloc[:, 1:6]))
            obs.iloc[:, [10, 12]] = StandardScaler().fit_transform(np.log(obs.iloc[:, [10, 12]]))
            obs.iloc[:, 7:10] = MinMaxScaler().fit_transform(obs.iloc[:, 7:10])
            obs.iloc[:, 11] = MinMaxScaler().fit_transform(obs.iloc[:, 11:12])
            obs.iloc[:, 13:] = MinMaxScaler().fit_transform(obs.iloc[:, 13:])
            obs = obs.values[:, 1:]
            if self.shares_held != 0:
                  return obs, np.array([[1]])
            else:
                  return obs, np.array([[0]])
        else:
              return obs
    def _take_action(self,action):
        self.current_price = self._next_observation(standard=False)['close'][4]
        if action <=1:
            if self.shares_held != 0:
                self.balance += self.shares_held*self.current_price*(1-self.mu)
                self.shares_held =0
            else:
                if self.balance > self.current_price * 100 * (1 + self.mu):  # 最低购买100股
                    self.shares_held = self.balance // (self.current_price * (1 + self.mu))
                    self.shares_held = self.shares_held // 100 * 100  # 100的整数倍
                    self.balance -= self.shares_held * self.current_price * (1 + self.mu)

    def step(self,action):
        self._take_action(action)
      # print( self.trade_date,self.current_step)
        next_3_date_chg = self.get_data(ts_code=self.ts_code,\
                                    start_date=self.trade_date[self.current_step+3],\
                                    )['pct_chg'].values[-3:]
        # print(next_3_date_chg)
        s1,s2 = self._next_observation()
        done = (self.current_step == 2628)
        pos_chg = next_3_date_chg.sum()
        if self.shares_held !=0:
              reward = pos_chg
        else:
              reward = 0
        self.current_step +=1
        return s1,s2,reward,done,{}

    def reset(self,istest = False):
        self.balance = 100000
        self.net_worth = 0
        self.mu =0.001
        self.shares_held=0
        self.current_step = 0 if not istest else 2629
        self.current_price = 0
        return self._next_observation()

    def render(self,mode = 'human',close = False):
        profit = (self.balance + self.shares_held * self.current_price) / 100000
        return profit
class Actor():
    def __init__(self,action_dim,action_bound,learning_rate,batch,tau=0.01):
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.batch = batch
        self.tau = tau
        self.target =self._build_net(trainable=False,scope = 'target')
        self.eval = self._build_net(trainable = True,scope = 'eval')
        self.target.set_weights(self.eval.get_weights())
        self.metric = tf.keras.metrics.MeanSquaredError()
    def _build_net(self,trainable=True,scope = ''):
        init_w = tf.random_normal_initializer(-.1,.1)
        init_b = tf.constant_initializer(.5)
        input1 = keras.Input(batch_shape = (None,5,18))
        input2 = keras.Input(batch_shape = (None,1))
        hidden1 = tf.keras.layers.Dense(32,name = scope+'_hidden1')(input1)
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=16,kernel_initializer=init_w,\
                                            bias_initializer=init_b,return_sequences=False\
                                            ,activation = 'tanh',dropout = .5,name = scope+'_bi_rnn')\
                                            )(hidden1)
#         lstm_bn = tf.keras.layers.BatchNormalization()(lstm)
        hidden2 = tf.keras.layers.Dense(units = 32,kernel_initializer=init_w,bias_initializer=init_b,\
                                        trainable = trainable,activation = 'tanh',name = scope+'_hidden2')(input2)
        compact = tf.keras.layers.concatenate([lstm,hidden2],axis = 1,name =scope+'_compact')
        action = tf.keras.layers.Dense(self.action_dim,kernel_initializer=init_w,bias_initializer=init_b,\
                                       activation = 'softmax',name = scope+'_action')(compact)
        output = tf.multiply(action,self.action_bound,name = scope+'_action_output')
        model = keras.models.Model(inputs = [input1,input2],outputs = [output])
        return model
    def choose_action(self,s1,s2):
        # print(s1.shape,s2.shape)
        # print(self.eval.summary())?
        s1 = s1.reshape(-1,5,18)
        s2 = s2.reshape(-1,1)
        # print(s1.shape)
        # print(self.eval.summary())
        return self.target.predict([s1,s2])[0]

    def learn(self,s1,s2,g):
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            y = self.eval([s1,s2])
            y = tf.cast(y,tf.float64)
            y = tf.multiply(y,-g)
        grads = tape.gradient(y,self.eval.variables)
        grads_and_vars = zip(grads,self.eval.variables)
        opt = tf.keras.optimizers.Adam(lr =self.lr)
        opt.apply_gradients(grads_and_vars)
        target_parmas = [(1-self.tau)*t +self.tau*e for t,e in zip(self.target.get_weights(),self.eval.get_weights())]
        self.target.set_weights(target_parmas)
class Critic():
    def __init__(self,state_dim,action_dim,learnig_rate,gamma,tau,actor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learnig_rate
        self.gamma = gamma
        self.tau = tau
        self.a = actor
        self.q = self._build_net(trainable=True,scope = 'eval')
        self.q_ = self._build_net(trainable =False,scope = 'target')
        self.q_.set_weights(self.q.get_weights())
        self.metric =  tf.keras.metrics.MeanSquaredError()
    def _build_net(self,trainable=True,scope = ''):
        init_w = tf.random_normal_initializer(-.1, .1)
        init_b = tf.constant_initializer(.5)
        Input1 = tf.keras.layers.Input(batch_shape=(None,5,18),name=scope+'_state1')
        Input2 = tf.keras.layers.Input(batch_shape = (None,1),name =scope+'_starte2')
        Input3 = tf.keras.layers.Input(batch_shape=(None,1),name =scope+'_action')
        lstm = tf.keras.layers.LSTM(30,activation = 'tanh',return_sequences=False,kernel_initializer=init_w,bias_initializer=init_b,\
                                    trainable=trainable,dropout = 0.5)(Input1)
        hidden1 = tf.keras.layers.Dense(30,trainable=trainable,kernel_initializer=init_w,bias_initializer=init_b)(Input2)
        hidden2 = tf.keras.layers.Dense(30,trainable=trainable,kernel_initializer=init_w,bias_initializer=init_b)(Input3)
#         lstm_bn = tf.keras.layers.BatchNormalization()(lstm)
        hidden_1_lstm = tf.keras.layers.concatenate([lstm,hidden1],axis = 1)
        hidden_1_lstm = tf.keras.layers.Dense(30,trainable=trainable,kernel_initializer=init_w,bias_initializer=init_b)(hidden_1_lstm)
        # print(hidden_1_lstm.shape,hidden2.shape)
        net = tf.keras.layers.add([hidden_1_lstm,hidden2])
        q = tf.keras.layers.Dense(1,trainable=trainable,name='q')(net)
        q = tf.keras.activations.tanh(q)
        model = tf.keras.models.Model(inputs=[Input1,Input2,Input3],outputs=[q])
        return model
    def learn(self,s1,s2,a,r,s1_,s2_):
        # print(s1.shape,s2.shape,a.shape,r.shape.s1_.shape.s2_.shape)
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            pre = self.q([s1,s2,a])
            a_ = self.a.predict([s1_,s2_])
            y = r+self.gamma*self.q_([s1_,s2_,a_])
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y,pre))
        grads = tape.gradient(loss,self.q.variables)
        grads_and_vars = zip(grads,self.q.variables)
        opt = tf.keras.optimizers.Adam(lr =self.lr)
        opt.apply_gradients(grads_and_vars)
        eval_parmas = self.q.get_weights()
        target_parmas = self.q_.get_weights()
        target_parmas = [(1-self.tau)*t+self.tau*e for t,e in zip(eval_parmas,target_parmas)]
        self.q_.set_weights(target_parmas)
        a = tf.Variable(a)
        with tf.GradientTape() as tape1:
            y = self.q([s1, s2, a])
        grades = tape1.gradient(y, a)
        return grades
class Memory():
    def __init__(self,max_store=500):
        self.s1 = np.empty(shape = (1,5,18))
        self.s2 = np.empty(shape = (1,1))
        self.a = np.empty(shape = (1,1))
        self.r = np.empty(shape=(1,1))
        self.s1_ = np.empty(shape = (1,5,18))
        self.s2_ =  np.empty(shape=(1,1))
        self.pointer = 0
        self.ms = max_store
    def stor_trainsiton(self,s1,s2,a,r,s1_,s2_):
        if self.pointer <= self.ms:
            self.s1 = np.vstack([self.s1,s1.reshape(1,5,18)])
            self.s2 = np.vstack([self.s2,s2.reshape(1,1)])
            self.a = np.vstack([self.a,a.reshape(1,1)])
            self.r = np.vstack([self.r,r.reshape(1,1)])
            self.s1_ = np.vstack([self.s1_,s1_.reshape(1,5,18)])
            self.s2_ = np.vstack([self.s2_,s2_.reshape(1,1)])
        else:
            index = self.pointer %self.ms
            self.s1[index,:] = s1.reshape(1,5,18)
            self.s2[index,:] = s2.reshape(1,1)
            self.a[index,:] = a.reshape(1,1)
            self.r[index,:] =r.reshape(1,1)
            self.s1_[index,:] = s1_.reshape(1,5,18)
            self.s2_[index,:] = s2_.reshape(1,1)
        self.pointer +=1
    def sample(self,n=64):
        assert self.pointer >= self.ms,'Memory has not been fullfilled'
        indices = np.random.choice(self.ms,size = n)
        return self.s1[indices,:].astype('float'),self.s2[indices,:].astype('float'),\
               self.a[indices,:].astype('float'),self.r[indices,:].astype('float'),\
               self.s1_[indices,:].astype('float'),self.s2_[indices,:].astype('float')

if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    learning_rate = 1e-2
    state_dim = (5,19)
    action_dim = 1
    action_bound = 2
    batch = 64
    gamma = 0.9
    tau = 0.01
    actor = Actor(action_dim,action_bound,learning_rate,batch)
    plot_model(actor.target,'./actor.png',show_shapes = True,show_layer_names = True)
    critic = Critic(state_dim,action_dim,learning_rate,gamma,tau,actor.target)
    plot_model(critic.q,'./critic.png',show_shapes = True,show_layer_names = True)
    memory = Memory(max_store=100)
    env = StockTradingEnv()
    max_episodes = 100
    max_ep_steps =  200
    isstart = 1
    result = []
    for i in range(max_episodes ):
        import time
        s1,s2 = env.reset()
        ep_reward = 0
        t1 = time.time()
        var = 1
        for j  in tqdm(range(max_ep_steps)):
            a = actor.choose_action(s1,s2)[0]
            a = round(np.clip(abs(np.random.normal(a,var)),1,2))
            s1_,s2_,r,done,_  = env.step(a)
            r = np.array([[r/100]])
            memory.stor_trainsiton(s1,s2,a,r,s1_,s2_)
            if memory.pointer > memory.ms:
                if isstart ==1 :
                    print('start train')
                    isstart +=1
                var *= 0.995
                b_s1,b_s2,b_a,b_r,b_s1_,b_s2_ = memory.sample(batch)
                # print(b_s1.shape,b_s2.shape,b_a.shape,b_r.shape,b_s1_.shape,b_s2_.shape)
                grades = critic.learn(b_s1,b_s2,b_a,b_r,b_s1_,b_s2_)
                actor.learn(b_s1,b_s2,grades)
                critic.a = actor.target
#                 print(actor.target.weights)
            s1 = s1_
            s2 = s2_
            ep_reward += r[0][0]
            if j == max_ep_steps-1:
                print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %.2f' % var, 'net_worth:%.2f' % env.render())
                result.append(env.render())
        print('running time: ',str(time.time()-t1))
    plt.plot(range(len(result)),result)
    plt.xlabel('次数')
    plt.ylabel('净值')
    plt.show()