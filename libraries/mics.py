#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:27:00 2020

@author: zqy5086@AD.PSU.EDU
"""

def nearest(x,S):
    (_,ind)=S.query(x)
    z=S.data[ind]
    key=to_string(z)
    zstate=r1.Xfree.table[key]
    return z


if __name__=="__main__":    
    (_,ind)=r1.X_safe.query(r1.x)
    z=np.array(r1.X_safe.data[ind[0,0]])
    key=to_string(z)
    zstate=r1.Xfree.table[key]
    