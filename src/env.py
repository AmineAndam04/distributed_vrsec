from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
#from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from src.unity_parallel_env import UnityParallelEnv
import uuid
class HyperParametersSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("1a2b3c45-5e65-7a85-9c05-1e2f3a4b5c65"))  
        self.received_rewards = []  # Store the received data here
    def on_message_received(self, msg: IncomingMessage) -> None:
        pass
    def send_hyperparameters(self, float_list):
        msg = OutgoingMessage()
        msg.write_int32(len(float_list))  # Write the count of float values
        for value in float_list:
            msg.write_float32(value)  # Write each float value
        self.queue_message_to_send(msg)
def load_env(env_path,seed,rep_length,worker_id=5):
    hyperparameters = [5, 0.5, 0.5, 2.28, 1, rep_length]
    hyperparametersChannel = HyperParametersSideChannel()
    hyperparametersChannel.send_hyperparameters(hyperparameters)
    stats_side_channel = StatsSideChannel()
    side_channels = [hyperparametersChannel,stats_side_channel]
    env = UnityEnvironment(file_name=env_path, side_channels=side_channels,worker_id=worker_id,no_graphics=True,seed=seed)
    env = UnityParallelEnv(env)
    return env