import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
import numpy as np
from tqdm import tqdm
from skimage import transform
import os
import random
from collections import deque
from vizdoom import ScreenFormat
from convnet import ConvNet
from convnet import ConvNet2

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

is_training = True
is_rendering = False
double_dqn_enabled = False
is_basic_scenario = True
reset_weights = True

# Set the hyperparameters for basic scenario
if is_basic_scenario:
    MODEL_PATH = "basic_scenario.pth"
    # Set hyperparameters
    gamma = 0.99  # Discount factor for future rewards
    epsilon_min = 0.1  # Minimum epsilon value
    epsilon_start = 1
    epsilon_phase_1_steps = 20000  # Number of steps while epsilon is 1.0
    epsilon_phase_2_steps = 100000  # Number of steps to decay epsilon to 0.1
    lr = 0.001
    num_train_epochs = 150
    num_total_training_steps_per_epoch = 2000
    batch_size = 32  # replay memory batch size
    target_update_frequency = 10000  # After how many time steps the target net is updated
    replay_memory_size = 50000
    # according to original vizdoom paper, best agent had a frame repeat as such
    frame_repeat = 4
else:
    MODEL_PATH = "health_gather_supreme.pth"
    # Set hyperparameters
    gamma = 1  # Discount factor for future rewards - original vizdoom paper
    epsilon_min = 0.1  # Minimum epsilon value
    epsilon_start = 1
    epsilon_phase_1_steps = 4000  # Number of steps while epsilon is equal to epsilon_start
    epsilon_phase_2_steps = 100000  # Number of steps to decay epsilon to epsilon_min
    lr = 0.0001 # learning rate
    num_train_epochs = 500
    num_total_training_steps_per_epoch = 2000
    batch_size = 64  # replay memory batch size
    target_update_frequency = 10000  # After how many time steps the target net is updated
    replay_memory_size = 200000
    frame_repeat = 10
    #frame_repeat = 7

if is_basic_scenario:
    stack_size = 4
else:
    stack_size = 1

class ReplayMemory:
    def __init__(self):
        self.memory = deque([], maxlen=replay_memory_size)
    def append(self, transition):
        self.memory.append(transition)
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    def __len__(self):
        return  len(self.memory)

def save_model(model, path):
    """Save the model to the given path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load the model from the given path."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}. Starting from scratch.")

def delete_model(path):
    """Delete the saved model file at the given path."""
    if os.path.exists(path):
        os.remove(path)
        print(f"Model deleted from {path}")
    else:
        print(f"No model found at {path} to delete.")

def stack_frames(stacked_frames,state,is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((30, 45), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames,axis=0)

    return stacked_state,stacked_frames

def preprocess_frame(img):
    """Down samples image to resolution"""
    # Crop 80 rows from top and 78 from bottom of img np.array
    img = img[80:-78, :]
    # Normalize the pixel values
    img=img/255.0
    img = transform.resize(img, (30,45))
    # the default img is of dtype float64 so convert to 32 for efficiency
    img = img.astype(np.float32)
    return img

def act(stacked_state_tensor, epsilon, actions, model):
    # # Epsilon-greedy action selection
    if random.random() < epsilon:
        random_action = random.choice(actions)
        return random_action  # Exploration: choose a random action
    else:
        with torch.no_grad():
            q_values = model(stacked_state_tensor.to(DEVICE))  # Get Q-values from the model
            action = actions[torch.argmax(q_values).item()]
            #print("Action: ", action)
            return action  # Exploitation: choose the action with the highest Q-value

def optimize(mini_batch, policy_network, target_network, optimizer, loss_fn):
    # Unzip the mini_batch into separate lists
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    # Convert lists to tensors and reshape them properly
    # The squeeze(1) removes the extra dimension, changing [32, 1, 4, 60, 90] to [32, 4, 60, 90]
    states = torch.stack(states).squeeze(1).to(DEVICE)
    actions = torch.stack(actions).to(DEVICE)
    next_states = torch.stack(next_states).squeeze(1).to(DEVICE)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
    if double_dqn_enabled:
        best_action_from_policy_n = policy_network(next_states).argmax(dim=1)
        target_q_values = (rewards + (1 - dones) *
                           gamma * target_network(next_states).gather(dim=1, index = best_action_from_policy_n.unsqueeze(1)).squeeze())
    else:
        # Calculate next state values with target network
        with torch.no_grad():
            target_q_values = rewards + (1 - dones) * gamma * target_network(next_states).max(dim=1)[0]

    current_q_values = policy_network(states).gather(dim=1, index = actions.argmax(dim=1).unsqueeze(dim=1)).squeeze()

    # Calculate loss and optimize
    loss = loss_fn(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def prepare_game():
    game = vzd.DoomGame()
    if is_basic_scenario:
        config_path = os.path.join(vzd.scenarios_path, "basic.cfg")
        game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    else:
        config_path = os.path.join(vzd.scenarios_path, "health_gathering_supreme.cfg")
        game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "health_gathering_supreme.wad"))

    print(config_path)
    game.load_config(config_path)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_window_visible(is_rendering)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(True)

    return game

def populate_memory(memory, actions, policy_net, game):
    step = 0
    epsilon = 1
    game.init()
    print("Initializing replay memory buffer...")
    while step < batch_size:
        game.new_episode()
        done = False
        stacked_frames = deque([np.zeros((30, 45), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)

        while not done:
            state = game.get_state()
            state = state.screen_buffer
            stacked_state, stacked_frames = stack_frames(stacked_frames, state, game.is_new_episode())
            stacked_state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0)
            action = act(stacked_state_tensor, epsilon, actions, policy_net)
            reward = game.make_action(action, frame_repeat)
            action = torch.tensor(action)
            done = game.is_episode_finished()

            # this is the step where we are adding new experiences to memory
            if done:
                next_state = np.zeros((480, 640), dtype=np.uint8)
                next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                next_stacked_state_tensor = torch.tensor(next_stacked_state, dtype=torch.float32).unsqueeze(0)
                transition = stacked_state_tensor, action, reward, next_stacked_state_tensor, done
                memory.append(transition)
            else:
                next_state = game.get_state().screen_buffer
                next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, game.is_new_episode())
                next_stacked_state_tensor = torch.tensor(next_stacked_state, dtype=torch.float32).unsqueeze(0)
                transition = stacked_state_tensor, action, reward, next_stacked_state_tensor, done
                memory.append(transition)
            step += 1

def show_trained_agent(game, actions, policy_net):
    print("======================================")
    print("Training finished. It's time to watch!")
    print("*** DQN agent ***")

    epsilon = 0.05
    game.close()

    game.set_window_visible(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()
    avg_reward = 0
    eps = 0
    policy_net.eval()
    episodes = 10
    rewards_per_episode = np.zeros(episodes)
    # Run episodes for testing the trained model
    for _ in range(episodes):
        game.new_episode()
        total_reward = 0

        stacked_frames = deque([np.zeros((30, 45), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
        while not game.is_episode_finished():
            state = game.get_state()

            img = state.screen_buffer
            stacked_state, stacked_frames = stack_frames(stacked_frames, img, game.is_new_episode())
            with torch.no_grad():
                stacked_state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0)
                reward = game.make_action(act(stacked_state_tensor, epsilon, actions, policy_net))
            total_reward += reward

        # Sleep between episodes
        print("Episode finished.")
        print("************************")
        avg_reward += total_reward
        eps += 1
        rewards_per_episode[_] = total_reward
        print("Total reward for this episode: ", total_reward)
        #time.sleep(2)
    print("************************")
    print("************************")
    print("************************")
    game.close()

def set_available_actions(game):
    if is_basic_scenario:
        actions = np.identity(3)
    else:
        available_buttons = [vzd.TURN_LEFT, vzd.TURN_RIGHT, vzd.MOVE_FORWARD]
        game.set_available_buttons(available_buttons)
        n = game.get_available_buttons_size()
        actions = np.identity(n, dtype=int).tolist()

    # Print or use the actions list as needed
    print(f"Number of valid actions: {len(actions)}")
    for action in actions:
        print(action)

    return actions

def handle_target_update(general_steps, policy_net, target_net):
    if general_steps % target_update_frequency == 0:
        target_net.load_state_dict(policy_net.state_dict())

def run():
    game= prepare_game()
    actions = set_available_actions(game)

    # initialize policy network and target network
    if is_basic_scenario:
        policy_net = ConvNet(len(actions)).to(DEVICE)
        target_net = ConvNet(len(actions)).to(DEVICE)
    else:
        policy_net = ConvNet2().to(DEVICE)
        target_net = ConvNet2().to(DEVICE)

    if reset_weights and is_training:
        delete_model(MODEL_PATH)

    load_model(policy_net, MODEL_PATH)
    # copy the policy network parameters to target network
    target_net.load_state_dict(policy_net.state_dict())

    # loss function is mean squared error loss
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(policy_net.parameters(), lr)

    # initialize epsilon to 1 (100% random actions)
    epsilon = epsilon_start

    # initialize memory replay buffer to capacity
    memory_buffer = ReplayMemory()

    if is_training:
        populate_memory(memory_buffer, actions, policy_net, game)

        general_steps = 0 # This counter is used for epsilon and target network updates
        for _ in range(num_train_epochs):
            print()
            print(f"Epoch {_ + 1}/{num_train_epochs}...")
            episode_rewards = []  # List to store total rewards for each episode
            training_steps = 0

            pbar = tqdm(total=num_total_training_steps_per_epoch, desc="Training Progress", disable=False)
            while training_steps < num_total_training_steps_per_epoch:
                game.new_episode()
                done = game.is_episode_finished()
                stacked_frames = deque([np.zeros((30, 45), dtype=np.float32) for _ in range(stack_size)], maxlen=stack_size)
                if not is_basic_scenario:
                    prev_health = game.get_game_variable(vzd.HEALTH) # variable needed for rewards shaping during health gathering scenario

                while not done:
                    state = game.get_state()
                    state = state.screen_buffer
                    stacked_state, stacked_frames = stack_frames(stacked_frames, state, game.is_new_episode())
                    stacked_state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0)
                    action = act(stacked_state_tensor, epsilon, actions, policy_net)
                    reward = game.make_action(action, frame_repeat)

                    if not is_basic_scenario:
                        current_health = game.get_game_variable(vzd.HEALTH)
                        health_diff = current_health - prev_health
                        if health_diff < -8:
                            health_reward = -100  # Penalize poison vial health loss
                        elif health_diff > 5:
                            health_reward = 150  # Explicitly give high reward for sparse health gains and incentivize agent to survive
                        else:
                            health_reward = 0

                        prev_health = current_health
                        reward = reward + health_reward

                    action = torch.tensor(action)
                    # check if episode is done
                    done = game.is_episode_finished()

                    # this is the step where we are adding new experiences to memory
                    if done:
                        next_state = np.zeros((480, 640), dtype=np.uint8)
                        next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        next_stacked_state_tensor = torch.tensor(next_stacked_state, dtype=torch.float32).unsqueeze(
                            0)
                        transition = stacked_state_tensor, action, reward, next_stacked_state_tensor, done
                        memory_buffer.append(transition)
                    else:
                        next_state = game.get_state().screen_buffer
                        next_stacked_state, stacked_frames = stack_frames(stacked_frames, next_state,
                                                                          game.is_new_episode())
                        next_stacked_state_tensor = torch.tensor(next_stacked_state, dtype=torch.float32).unsqueeze(
                            0)
                        transition = stacked_state_tensor, action, reward, next_stacked_state_tensor, done
                        memory_buffer.append(transition)

                    # pick a random sample from experience replay buffer
                    minibatch = memory_buffer.sample(batch_size)
                    optimize(minibatch, policy_net, target_net, optimizer, loss_fn)
                    if general_steps < epsilon_phase_1_steps:
                        epsilon = epsilon_start  # Keep epsilon at 1.0 for the first 100,000 steps
                    elif general_steps == epsilon_phase_1_steps:
                        print("\nEpsilon decay started")
                    elif general_steps < epsilon_phase_1_steps + epsilon_phase_2_steps:
                        # After 100,000 steps, decay epsilon from epsilon_start to epsilon_min for the next 100,000 steps
                        decay_progress = (general_steps - epsilon_phase_1_steps) / epsilon_phase_2_steps
                        epsilon = max(epsilon_min,
                                      epsilon_start - (epsilon_start - epsilon_min) * min(decay_progress, 1))
                    elif general_steps == epsilon_phase_1_steps + epsilon_phase_2_steps:
                        print("\nEpsilon decay ended")
                        # single step learning rate decay for hgs scenario
                        if not is_basic_scenario:
                            l = 0.00003
                            optimizer = optim.AdamW(policy_net.parameters(), l)
                    handle_target_update(general_steps, policy_net, target_net)

                    training_steps += 1
                    general_steps += 1

                    if training_steps >= num_total_training_steps_per_epoch:
                        break
                    pbar.update(1)
                episode_rewards.append(game.get_total_reward())
            print()
            avg_reward_per_episode = sum(episode_rewards) / len(episode_rewards)
            print("Average Reward per episode: ", avg_reward_per_episode)

            pbar.close()
            save_model(policy_net, MODEL_PATH)

        game.close()

    show_trained_agent(game, actions, policy_net)
    game.close()

run()