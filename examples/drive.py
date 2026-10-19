"""The simulated Go2 in a scanned ranch home, on zenoh: drive it with anything that publishes cmd_vel."""

import sys

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.module_coordinator import ModuleCoordinator

from dim_go2_sim import Go2Sim

drive = Go2Sim.blueprint(scene={"package": "supersplat/3f89bbd3"}).global_config(transport="zenoh")

if __name__ == "__main__":
    ModuleCoordinator.build(drive, BlueprintConfigParser(drive).parse(sys.argv[1:])).loop()
