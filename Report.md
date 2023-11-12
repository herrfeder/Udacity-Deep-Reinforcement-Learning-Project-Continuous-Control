### Training Specs

  * I used the following environment incoporating only **one training agent**. 
The hyperparameters where choosen with some assumptions but in the end the best results where found by trying different combinations:

```
##Reacher Environment##

# Environment Details
- Number of Agents: 1
- Size of Action (Continuous): 4 
- Number of state variables: 33

# Hyper Parameters
{'batch_size': 256,
 'buffer_size': 100000,
 'gamma': 0.99,
 'lin_full_con_01': 128,
 'lin_full_con_02': 128,
 'lr_actor': 0.001,
 'lr_critic': 0.001,
 'noise_scalar': 0.25,
 'tau': 0.001,
 'weight_decay': 0}
```

  * The training was done with a `Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz` in a headless VPS Machine.
  * It took about 6 hours and 252 episodes to finish the training:
  
![](images/screenshot_finished_training.png)

### Important Parts



### Tuned Hyperparameters

 * **Dimension Fully Connected Layer** 
   * tried different sizes of the Fully Connected Layers
     * multiples of state size -> didn't converge at all
     * 128 -> best results
 * **Learning Rate Actor and Critic**
  * use the learning rates according to the original DDPG Paper
   
