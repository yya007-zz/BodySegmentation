# -*- coding: utf-8 -*-
"""
@author: Yuang Yao
@email: yuang.yao@vanderbilt.edu
"""
import os
def makedir(add):
    if not os.path.exists(add):
        print "make ",add
        os.makedirs(add)
    else:
        print add," already exists"
        
makedir("../bigfile")
makedir("../res")
makedir("../network")
makedir("../Data3D")
makedir("../Data3D/train")
makedir("../Data3D/test")
