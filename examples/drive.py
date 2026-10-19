"""Walk the simulated Go2 around a scanned ranch home with WASD in dimos-viewer."""

import sys

from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.visualization.vis_module import vis_module

from dim_go2_sim import Go2Sim

drive = autoconnect(
    Go2Sim.blueprint(scene={"package": "supersplat/3f89bbd3"}),  # the ranch home
    vis_module(viewer_backend=global_config.viewer),  # dimos-viewer: its WASD -> tele_cmd_vel
    MovementManager.blueprint(),  # tele_cmd_vel (and a planner's nav_cmd_vel) -> cmd_vel
)

if __name__ == "__main__":
    ModuleCoordinator.build(drive, BlueprintConfigParser(drive).parse(sys.argv[1:])).loop()
