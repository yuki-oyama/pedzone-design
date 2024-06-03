# Optimal design of pedestrian zones
Python code for multiobjective optimization to design pedestrian zones in a city center network solved by Adaptive Large Neighborhood Search (ALNS) algorithm.

## Paper
For more details, please see the paper

Oyama, Y., Murakami, S., Chikaraishi, M., Parady, G. (2024) [Designing pedestrian zones within city center networks considering policy objective trade-offs](https://www.sciencedirect.com/science/article/pii/S0965856424001678?via%3Dihub). Transportation Research Part A: Policy and Practice 185: 104119. 

If you find this code useful, please cite the paper:
```
@article{oyama2024pedzone,
  title = {Designing pedestrian zones within city center networks considering policy objective trade-offs},
  journal = {Transportation Research Part A: Policy and Practice},
  volume = {185},
  pages = {104119},
  year = {2024},
  author = {Oyama, Yuki and Murakami, Soichiro and Chikaraishi, Makoto and Parady, Giancarlos},
}
```

## Quick Start
Just run the ```run.py``` code.

```
python run.py
```

You can test different parameters such as the maximum number of zones, the accuracy threshold for UE assignment convergence, and the number of temparature changes by specifying them as

```
python run.py --maxZones 10 --accuracy 0.01 --maxChanges 200
```

