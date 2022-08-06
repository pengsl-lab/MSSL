#!/bin/bash

python degree_code.py --downstream DTI   --scenario warm
python PairDistance.py  --downstream DTI   --scenario warm
python SimCon.py  --downstream DTI   --scenario warm
python EdgeMask.py  --downstream DTI   --scenario warm
python PathClass.py  --downstream DTI   --scenario warm
python SimReg.py  --downstream DTI   --scenario warm
python ClusterPre.py   --downstream DTI   --scenario warm
