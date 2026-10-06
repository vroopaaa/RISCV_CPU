import json

with open("docs/superscalar-analysis/data/superscalar_sweep_n10.json") as f:
    d = json.load(f)

for lvl in ["O0", "O1", "O2", "O3"]:
    c_n = {"O0": 5, "O1": 4, "O2": 6, "O3": 6}[lvl]
    run = d[lvl]["runs"][f"n{c_n}"]
    print(f"[{lvl}]")
    print(f"  scalar cycles    : {d[lvl]['runs']['scalar']['cycles']:,}")
    print(f"  converged cycles : {run['cycles']:,}")
    print(f"  hist sum         : {sum(run['hist'].values()):,}")
    print(f"  instructions     : {run.get('instructions', 'N/A')}")