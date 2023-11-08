from unityagents import UnityEnvironment
import numpy as np
import numpy as np
from ddpg_agent import Agent
from collections import deque
import torch


def init_environment():
    # initialise the headless unity environment
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

    # get the default environment, called brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # retrieve the facts about the unity environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    init_output = \
    """##Reacher Environment##

        # Environment Details
        - Number of Agents: {num_agents}
        - Size of Action (Continuous): {action_size} 
        - Number of state variables: {state_size}
        
        # Hyper Parameters

    """.format(num_agents=num_agents, action_size=action_size,state_size=state_size)
    print(init_output)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
    
    return agent, env, brain_name


def ddpg(n_episodes=1000, max_t=1000, window_size=100):
    """DDQN Algorithm.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): frequency of printing information throughout iteration """
   
    agent, env, brain_name = init_environment()
    scores = []
    scores_deque = deque(maxlen=window_size)
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)                      # select an action
       
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done, t) # take step with agent (including learning)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        
        scores_deque.append(score)       # save most recent score
        scores.append(score)             # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            with open('last_scores.txt', 'w') as score_file:
                for element in scores:
                    score_file.write(str(element))
                    score_file.write("\n")

        if np.mean(scores_deque)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            with open('final_scores.txt', 'w') as score_file:
                for element in scores:
                    score_file.write(str(element))
                    score_file.write("\n")
            break
            
    return scores

scores = ddpg(n_episodes=1000)

