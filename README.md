# Direct Preference Optimization

## Used resourses:
* NVIDIA GeForce RTX 3090
* Python 3.12.8

## Some parameters/hyperparameters:
* GPT2 model
* Anthropic/hh-rlhf dataset with 3000 samples for TRAIN/VAL
* Anthropic/hh-rlhf dataset with 500 samples for TEST
* max_length = 256
* batch_size = 16
* device = cuda

## [Link to ckeckpoints](https://disk.yandex.ru/d/VwVExhptubkUWw)

## Instructions to run DPO
- Load checkpoints to use for test
- Run docker and commands:
```
docker build -t dpo .
docker run -it -v $(pwd):/app dpo

# run train
python main.py --mode train

# run test
python main.py --mode test --checkpoint_dir checkpoints/checkpoint_beta0.3
```

## Implementation details
DOP loss formula used:
<img width="1128" height="105" alt="image" src="https://github.com/user-attachments/assets/f2ea9e40-7578-457a-87e0-a682317c6d7c" />

### Train

I saved checkpoints for every epochs and for multiple beta parameters - (0.3, 0.5, 0.7)

Overall, after testing possible hyperparameters beta and lr, chosen reward and rejected reward both were negative. Still model ranked chosen higher than rejected. And reward always was positive that says about correct learning.

With param beta=0.3, lr=1e-4, epochs=5:
While analysing train and val statistic, after 3 epochs already model started to overfit, but reward only increased.
![alt text](assets/beta0.3_1.png)

With param beta=0.3, lr=1e-5, epochs=10:
I wanted to test if for lower beta we need lower lr. 
While analysing train and val statistic, after 5 epochs already model started to overfit, but reward only increased.
![alt text](assets/beta0.3_2.png)

Beta = 0.3 with different lr achived loss ~ 0.61.

With param beta=0.5, lr=1e-4, epochs=5:
While analysing train and val statistic, after 2 epochs already model started to overfit, but reward only increased.
I saved in the end for test checkpoint for epoch2.
![alt text](assets/beta0.5.png)

With param beta=0.7, lr=1e-5, epochs=10:
I wanted to test if for lower beta we need lower lr. 
While analysing train and val statistic, after 4 epochs already model started to overfit, but reward only increased.
![alt text](assets/beta0.7.png)

### Evaluation
I wanted to interpret results in the text format and to show that if to give to trained model prompt and 2 responses, it will rank more preferable higher.
But this is identical as numerical interpretation.
So I have just took 500 samples from Anthropic/hh-rlhf dataset and the best performance checkpoint - checkpoint_beta0.3.

Results show in avg correct preference ranking and small reward_margin = 0.17 distinguishes preferences, but weakly:

Test metrics: {'loss': 0.66, 'chosen_reward': -1.88, 'rejected_reward': -2.06, 'reward_margin': 0.17}

## Observations
* All configurations overfit after 2-5 epochs, that can be cause of limited dataset.

Even after overfit reward margin were increased, that means that model prioritized reward maximization over generalization.

* Different beta regularization did not help with overfiting

**Effect of beta:**

    - Lower beta (0.3) achieved better margins (0.17 vs. 0.1 for beta=0.7).

    - Higher beta penalizes divergence more, limiting optimization.

