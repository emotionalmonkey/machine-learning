"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 1 # epsilon-greedy parameter for training
TESTING_EP = 1  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training
ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here
    action_index, object_index = None, None    
    if np.random.random() < epsilon:
        action_index, object_index = np.random.choice(NUM_ACTIONS), np.random.choice(NUM_OBJECTS)
    else:
        max_index = np.argmax(q_func[state_1, state_2,:,:])
        action_index, object_index = np.unravel_index(max_index, (NUM_ACTIONS, NUM_OBJECTS))
    return (action_index, object_index)

# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    # TODO Your code here    
    max_Q = 0 if terminal else np.max(q_func[next_state_1, next_state_2, :, :])
    sample = reward + GAMMA * max_Q
    q_func[current_state_1, current_state_2, 
           action_index, object_index] = ALPHA * sample + (1 - ALPHA) * q_func[current_state_1, current_state_2, action_index, object_index]  # TODO Your update here
    
    return None  # This function shouldn't return anything

# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0.0
    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()    
    step_count = 0
    while not terminal:
        # Choose next action and execute
        # TODO Your code here
        sr, sq = dict_room_desc[current_room_desc], dict_quest_desc[current_quest_desc]
        action_index, object_index = epsilon_greedy(sr, sq, q_func, epsilon)
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc, action_index, object_index)
                
        next_sr = dict_room_desc[next_room_desc]

        if for_training:
            # update Q-function.
            # TODO Your code here
            # Next Quest is also sq. Because QUEST remains unchanged within each episode 
            tabular_q_learning(q_func, sr, sq, action_index, object_index, reward, next_sr, sq, terminal)

        if not for_training:
            # update reward
            # TODO Your code here
            epi_reward = epi_reward + (GAMMA ** step_count) * reward            

        # prepare next step
        # TODO Your code here
        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc
        step_count += 1

    if not for_training:
        return epi_reward


# pragma: coderesponse end
def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description("Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each
    # state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' % (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()


"""
Avg reward: 0.479168 | Ewma reward: 0.506410: 100%|█| 200/200 [00:04<00:00, 48.7
Avg reward: 0.494843 | Ewma reward: 0.507415: 100%|█| 200/200 [00:03<00:00, 50.2
Avg reward: 0.502690 | Ewma reward: 0.512356: 100%|█| 200/200 [00:04<00:00, 48.2
Avg reward: 0.495625 | Ewma reward: 0.516579: 100%|█| 200/200 [00:04<00:00, 45.8
Avg reward: 0.505080 | Ewma reward: 0.499214: 100%|█| 200/200 [00:03<00:00, 50.2
Avg reward: 0.496904 | Ewma reward: 0.511149: 100%|█| 200/200 [00:04<00:00, 48.1
Avg reward: 0.500119 | Ewma reward: 0.522306: 100%|█| 200/200 [00:04<00:00, 43.2
Avg reward: 0.494638 | Ewma reward: 0.511368: 100%|█| 200/200 [00:04<00:00, 47.9
Avg reward: 0.501216 | Ewma reward: 0.519421: 100%|█| 200/200 [00:04<00:00, 48.7
Avg reward: 0.476696 | Ewma reward: 0.492471: 100%|█| 200/200 [00:04<00:00, 47.4


0.506225

"""