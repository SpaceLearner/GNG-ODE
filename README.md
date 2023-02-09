# GNG-ODE
This is the code for CIKM 2022 paper Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation. 

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

Generally using dopri5 solver will increase the performance and bing longer training time.


## Citation

```
@inproceedings{Guo2022EvolutionaryPL,
    title={Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation},
    author={Jiayan Guo and Peiyan Zhang and Chaozhuo Li and Xing Xie and Yan Zhang and Sunghun Kim},
    booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
    year={2022}
}
```
