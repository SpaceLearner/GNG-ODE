# GNG-ODE
This is the code for CIKM 2022 paper Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation. 

## Abstract

Session-based recommendation (SBR) aims to predict the user’s next action based on the ongoing sessions. Recently, there has been an increasing interest in modeling the user preference evolution to capture the fine-grained user interests. While latent user preferences behind the sessions drift continuously over time, most existing approaches still model the temporal session data in discrete state spaces, which are incapable of capturing the fine-grained preference evolution and result in sub-optimal solutions. To this end, we propose Graph Nested GRU ordinary differential equation (ODE) namely GNG-ODE, a novel continuum model that extends the idea of neural ODEs to continuous-time temporal session graphs. The proposed model preserves the continuous nature of dynamic user preferences, encoding both temporal and structural patterns of item transitions into continuous-time dynamic embeddings. As the existing ODE solvers do not consider graph structure change and thus cannot be directly applied to the dynamic graph, we propose a time alignment technique, called t-Alignment, to align the updating time steps of the temporal session graphs within a batch. Empirical results on three benchmark datasets show that GNG-ODE significantly outperforms other baselines.


## Dataset

We have provided the processed datasets. If you want the original data please refer to the following links.

* [Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz)
* [Tmall](https://github.com/CCIIPLab/GCE-GNN/tree/master/datasets/Tmall)
* [Nowplaying](https://github.com/CCIIPLab/GCE-GNN/tree/master/datasets/Nowplaying)

## Environment



## Usage

### For Gowalla

```
python -u scripts/main_ode.py --dataset-dir ../datasets/gowalla --gnn GATConv
```

### For Tmall

```
python -u scripts/main_ode.py --dataset-dir ../datasets/tmall --gnn GATConv --solver dopri5
```

### For Nowplaying

```
python -u scripts/main_ode.py --dataset-dir ../datasets/nowplaying --gnn GATConv 
```

Generally using dopri5 solver will increase the performance and bring longer training time.


## Citation

```
@inproceedings{Guo2022EvolutionaryPL,
    title={Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation},
    author={Jiayan Guo and Peiyan Zhang and Chaozhuo Li and Xing Xie and Yan Zhang and Sunghun Kim},
    booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
    year={2022}
}
```
