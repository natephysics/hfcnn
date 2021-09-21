# %%
from hyperopt import fmin, tpe, hp

def andrea(arg_dict):
    x, y, z = arg_dict 
    return x*y - z

space = [
    hp.uniform('x', -10, 10),
    hp.choice('y', [1, 2, 4, 7, 9]),
    hp.normal('z', 0, 1)
]

best = fmin(fn=andrea,
    space=space,
    algo=tpe.suggest,
    max_evals=100)
print(best)
# %%
