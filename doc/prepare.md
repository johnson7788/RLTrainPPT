## 服务器准备
可以尝试智星云的按小时租赁服务器，前期拉取镜像期间可以不用GPU。

## 准备服务器上的clash代理
登录服务器，然后使用screen命令运行clash
```
screen -S clash
cd clash
./clash-linux-amd64-v1.10.0 -d .
```

##  尝试使用Areal的镜像
```
同步当前的仓库到服务器，然后拉取镜像
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --name areal ghcr.io/inclusionai/areal-runtime:v0.3.0.post2 sleep infinity
docker start areal
docker exec -it areal bash
cd /workspace/verl/RLTrainPPT/ART
pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 测试模型加载是否正常
```
cd doc
python load_model.py
```


