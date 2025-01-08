# Requirements

torch==1.9.0
tqdm==4.61.2
torch_scatter==2.0.7
torch_geometric==2.0.2
torch_sparse==0.6.10
pandas==1.1.5
matplotlib==3.3.4
scipy==1.5.4
numpy==1.16.2
loguru==0.5.3
scikit-learn==1.0.1



# Dataset

[JD](https://jdata.jd.com/html/detail.html?id=8): This is an open dataset collected from JD, which is one of the largest e-commerce platforms in China. There are three types of behaviors in this dataset, including view, tag-as-favorite, and purchase. Dataset link: https://jdata.jd.com/html/detail.html?id=8



# Start

Run the .sh script file directly

for example : 

```
. I_CRGCN.sh
```

Running different data sets requires changes to the code in the script file



# References

```
@article{yan2023cascading,
  title={Cascading residual graph convolutional network for multi-behavior recommendation},
  author={Yan, Mingshi and Cheng, Zhiyong and Gao, Chen and Sun, Jing and Liu, Fan and Sun, Fuming and Li, Haojie},
  journal={ACM Transactions on Information Systems},
  volume={42},
  number={1},
  pages={1--26},
  year={2023},
  publisher={ACM New York, NY}
}
```

Its github repository: https://github.com/MingshiYan/CRGCN



```
@article{ding2022causal,
  title={Causal incremental graph convolution for recommender system retraining},
  author={Ding, Sihao and Feng, Fuli and He, Xiangnan and Liao, Yong and Shi, Jun and Zhang, Yongdong},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={35},
  number={4},
  pages={4718--4728},
  year={2022},
  publisher={IEEE}
}
```

Its github repository: https://github.com/Dingseewhole/CI_LightGCN_master/