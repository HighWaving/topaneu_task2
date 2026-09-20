from scripts.astra6_e07.train import RUN
from scripts.astra6_e04 import run_e04 as e04
from scripts.astra6_e03.run_e03 import seed
seed();e04.BASE=RUN/'frozen_assignment';e04.infer(RUN)
