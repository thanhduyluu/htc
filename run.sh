python train.py -d vnexpress -mn hill -s 0 -b 4 -lr 1e-3 -k 3 -l 1e-3 -hd 768 -tp sum -ho bert
#python train.py -d wos -mn hill -s 0 -b 4 -lr 1e-3 -k 3 -l 1e-3 -hd 768 -tp sum -ho bert
# hill-t5-residual-contrastive
# hill-t5-tree-contrastive
# hill-t5-concat-contrastive
# hill-t5-only
# pm2 start run.sh -n hill-t5-residual-contrastive --no-autorestart
# pm2 start run.sh -n hill-t5-only-contrastive-loss --no-autorestart
# pm2 start run-hill.sh -n hill-t5-residual-contrastive --no-autorestart