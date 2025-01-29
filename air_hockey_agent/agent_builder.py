import numpy as np
import torch

from air_hockey_challenge.framework import AgentBase, AirHockeyChallengeWrapper



def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    #raise NotImplementedError
    return DummyAgent(env_info, **kwargs)


class DummyAgent(AgentBase):
    def __init__(self, env_info, value, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.primitive_variable = value  # Primitive python variable
        self.numpy_vector = np.array([1, 2, 3]) * value  # Numpy array
        self.list_variable = [1, 'list', [2, 3]]  # Numpy array

        # Dictionary
        self.dictionary = dict(some='random', keywords=2, fill='the dictionary')

        # Building a torch object
        data_array = np.ones(3) * value
        data_tensor = torch.from_numpy(data_array)
        self.torch_object = torch.nn.Parameter(data_tensor)

        # A non serializable object
        self.object_instance = object()

        # A variable that is not important e.g. a buffer
        self.not_important = np.zeros(10000)

        # Here we specify how to save each component
        self._add_save_attr(
            primitive_variable='primitive',
            numpy_vector='numpy',
            list_variable='primitive',
            dictionary='pickle',
            torch_object='torch',
            object_instance='none',
            # The '!' is to specify that we save the variable only if full_save is True
            not_important='numpy!',
        )

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action


if __name__ == '__main__':
    env = AirHockeyChallengeWrapper("custom")
    
    # Construct Agent
    args = {'value': 0.5}
    agent_save = build_agent(env.env_info, **args)
    print(agent_save)