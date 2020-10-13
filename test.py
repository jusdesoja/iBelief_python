#!/usr/bin/env python
# encoding: utf-8

from entropy import entropy
import numpy as np

m = np.array([0,0.4,0.1,0.5])
m_ign = np.array([0,0,0,1])
m_1 = np.array([0,0.1,0,0.9])
m_2 = np.array([0,1,0,0])
m_3 = np.array([0,0,0,0,0,0,0,1])
m_4 = np.array([0,0,0,0.5,0,0.5,0,0])
m_5 = np.array([0,0,0,0.3,0,0.3,0.4,0])
m_6 = np.array([0,0,0,0,0.4,0,0,0.6])
c = 5
print(entropy(m,c), entropy(m_ign,c),entropy(m_1,c),entropy(m_2,c))
print(entropy(m_3,c),entropy(m_4,c),entropy(m_5,c))
