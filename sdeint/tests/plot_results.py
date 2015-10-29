"""Run this script to plot the true solution against an approximated solution
from each integration algorithm, for several exactly solvable test systems"""

from test_integrate import *

t = Test_KP4446()
t.plot()

t = Test_KP4459()
t.plot()

t = Test_KPS445()
t.plot()

t = Test_R74()
t.plot()
