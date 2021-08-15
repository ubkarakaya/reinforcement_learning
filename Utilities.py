import random
from collections import deque
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from gym.wrappers import Monitor


def show_videos():
    mp4list = glob.glob('video/*.mp4')
    mp4list.sort()
    for mp4 in mp4list:
        print(f"\nSHOWING VIDEO {mp4}")
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))


def wrap_env(env, video_callable=None):
    env = Monitor(env, './video', force=True, video_callable=video_callable)
    return env


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward) -> object:
        self.memory.append(
            (state, action, next_state, reward))  # Add the tuple (state, action, next_state, reward) to the queue

    def sample(self, batch_size):
        batch_size = min(batch_size, len(
            self))  # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        return random.sample(self.memory, batch_size)  # Randomly select "batch_size" samples

    def __len__(self):
        return len(self.memory)  # Return the number of samples currently stored in the memory

