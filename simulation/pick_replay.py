import pickle

def load_recorded_data(file_path):
    with open(file_path, 'rb') as f:
        recorded_data = pickle.load(f)
    return recorded_data

def replay_agent(self, recorded_data):
    log.info('开始重放记录的数据')

    # 创建环境实例
    env = Block_Pick_Env(render=self.render)
    env.start()

    # 遍历记录的数据
    for record in recorded_data:
        # 获取上下文
        context = record['context']
        
        # 重置环境，使用记录的上下文
        env.reset(random=False, context=context)

        # 执行每个动作
        for action in record['actions']:
            pred_action = action['pred_action']
            obs, reward, done, info = env.step(pred_action)
            
            # 打印或验证观测值、奖励、结束状态等信息
            print(f'观测值: {obs}, 奖励: {reward}, 结束: {done}, 其他信息: {info}')
            
            if done:
                break

    log.info('重放完成')

    
