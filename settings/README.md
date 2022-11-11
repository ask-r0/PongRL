# Settings
The folder `settings` is reserved for different settings-profiles to be used in training.

One setting-profile is stored as a json file, with the following parameters (see Settings Parameters).

## Setting Parameters

| Parameter         | Explanation                                                                    | Example              |
|-------------------|--------------------------------------------------------------------------------|----------------------|
| env_name          | Name of gym environment                                                        | "PongNoFrameskip-v4" |
| save_to_file      | **true**: training params and neural net is saved to file, else **false**      | true                 |
| model_path        | File path the model (neural net) should be saved to                            |                      |
| params_path       | File path of where the params should be saved                                  |                      |
| gamma             | Gamma (discount rate)                                                          |                      |
| learning_rate     | Learning rate                                                                  |                      |
| epsilon_initial   | Initial epsilon value, typically 1.0                                           |                      |
| epsilon_min       | Epsilon will never go below this value                                         |                      |
| epsilon_decay     | Over how many frames epsilon should decay from epsilon_initial to epsilon_min  |                      |
| batch_size        | Size of the batch taken from replay memory (which is used to train neural net) |                      |
| memory_size       | Amount of memories in replay memory                                            |                      |
| memory_size_min   | Minimum number of memories in replay memory before training starts             |                      |
| max_frames        | Maximum total frames for training session                                      |                      |
| logging_rate      | Number of frames between each logging to console                               |                      |
| target_net_update | Number of frames between each update of the target network                     |                      |
| optimizer         | Optimizer to be used. Available choices: `adam`, `rmsprop`.                    | "adam"               |
| loss_function     | Loss function to be used. Available choices: `mse`, `huber`.                   | "mse"                |

 