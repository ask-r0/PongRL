# Settings
The folder `settings` is reserved for different settings-profiles to be used in training.

One setting-profile is stored as a json file, with the following parameters (see Settings Parameters).

## Setting Parameters

| Parameter         | Explanation                                                                       | Example              |
|-------------------|-----------------------------------------------------------------------------------|----------------------|
| env_name          | Name of gym environment                                                           | "PongNoFrameskip-v4" |
| save_to_file      | **true**: training params and neural net is saved to file, else **false**         | true                 |
| model_path        | File path the model (neural net) should be saved to                               | "save/nn.pth"        |
| params_path       | File path of where the params should be saved                                     | "save/params.json"   |
| gamma             | Gamma (discount rate)                                                             | 0.99                 |
| learning_rate     | Learning rate                                                                     | 0.00025              |
| epsilon_initial   | Initial epsilon value, typically 1.0                                              | 1.0                  |
| epsilon_min       | Epsilon will never go below this value                                            | 0.05                 |
| epsilon_decay     | Over how many frames epsilon should decay from epsilon_initial to epsilon_min     | 100000               |
| batch_size        | Size of the batch taken from replay memory (which is used to train neural net)    | 32                   |
| memory_size       | Amount of memories in replay memory                                               | 1000000              |
| memory_size_min   | Minimum number of memories in replay memory before training starts                | 800000               |
| max_frames        | Maximum total frames for training session                                         | 1000000              |
| logging_rate      | Number of frames between each logging to console                                  | 1000                 |
| target_net_update | Number of frames between each update of the target network                        | 1000                 |
| optimizer         | Optimizer to be used. Available choices: `adam`, `rmsprop`.                       | "adam"               |
| loss_function     | Loss function to be used. Available choices: `mse`, `huber`.                      | "mse"                |
| target_q_equation | How target Q-value will be calculated. Available choices: `dqn`, `ddqn`.          | "ddqn"               |
| network           | Neural network to be used in training. Available choices: `cnn`, `dueling`.       | "cnn"                |
| preprocessing     | Size of result image after preprocessing (84x84 etc.). Available choices: 68, 84. | 84                   |

 