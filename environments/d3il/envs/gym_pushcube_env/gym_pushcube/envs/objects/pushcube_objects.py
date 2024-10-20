import os

import numpy as np
import xml.etree.ElementTree as Et
from typing import Tuple

from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Sphere
from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject
from environments.d3il.d3il_sim.sims.mj_beta.MjLoadable import MjXmlLoadable
from environments.d3il.d3il_sim.utils import sim_path

init_end_eff_pos = [0.525, -0.35, 0.25]

box_pos = np.array([0.1, 0.15, 0.13])
box_quat = [0, 0, 0, 1]
target_pos = np.array([0.1, 0.05, 0.13])
target_quat = [0, 0, 0, 1]


class PushObject(SimObject, MjXmlLoadable):
    def __init__(self, file_name, object_name, pos, quat, root=sim_path.D3IL_DIR):
        if pos is None:
            pos = [0, 0, 0]
        else:
            assert len(pos) == 3, "Error, parameter pos has to be three dimensional."

        if quat is None:
            quat = [0, 0, 0, 0]
        else:
            assert len(quat) == 4, "Error, parameter quat has to be four dimensional."

        self.obj_dir_path = "./models/mj/common-objects/push_cube_box/" + file_name
        self.root = root
        self.pos = pos
        self.quat = quat
        self.name = object_name

        SimObject.__init__(self, name=self.name, init_pos=self.pos, init_quat=self.quat)
        MjXmlLoadable.__init__(self, os.path.join(root, self.obj_dir_path))

    def get_poi(self) -> list:
        """

        Returns:
            a list of points of interest for the scene to query
        """
        return [self.name]


def get_obj_list():

    push_box = PushObject(
        file_name="pushed_box.xml",
        object_name="pushed_box",
        pos=box_pos,
        quat=box_quat
    )

    target_box = PushObject(
        file_name="target_box.xml",
        object_name="target_box",
        pos=target_pos,
        quat=target_quat
    )

    platform_box = PushObject(
        file_name= "platform_box.xml",
        object_name="platform_box",
        pos=[0,0,0],
        quat=[0,0,0,1]
    )

    obj_list = [push_box, target_box , platform_box]

    return obj_list

    obj_list += [
        # TARGET
        Sphere(None, [0.35, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.35, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.20, 0], [0, 1, 0, 0], static=True, visual_only=True),
        Sphere(None, [0.7, 0.40, 0], [0, 1, 0, 0], static=True, visual_only=True),
        # WORKSPACE
        # Sphere(
        #     None,
        #     [0.30, -0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.30, 0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.8, -0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # Sphere(
        #     None,
        #     [0.8, 0.45, 0],
        #     [0, 1, 0, 0],
        #     static=True,
        #     visual_only=True,
        #     rgba=[0, 1, 0, 1],
        # ),
        # START
        Sphere(
            None,
            [0.35, -0.35, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.35, -0.1, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, -0.35, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
        Sphere(
            None,
            [0.7, -0.1, 0],
            [0, 1, 0, 0],
            static=True,
            visual_only=True,
            rgba=[0, 1, 1, 1],
        ),
    ]

    return obj_list
