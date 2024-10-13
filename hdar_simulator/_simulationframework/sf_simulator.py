import abc
import queue
from typing import List, Dict, Any
from typing import Callable, Tuple

from alr_sim.core.Scene import Scene
from alr_sim.sims.mj_beta import MjRobot
from alr_sim.sims.mj_beta import MjScene
from alr_sim.controllers import ControllerBase
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import MujocoObject
from simpub.sim.sf_publisher import SFPublisher

TaskQueue = queue.Queue[Tuple[Callable[..., None], Tuple[Any, ...]]]


class SFSimulator(abc.ABC):

    def __init__(
        self,
        task_name: str,
        host_address: str = None,
    ):
        # NOTE: for all the possible simulator
        self.task_name = task_name
        # MJSimFactory
        self.sim_factory = SimRepository.get_factory("mj_beta")
        self.mj_scene = self.create_scene()
        self.robot_dict = self.create_robots()
        self.task_queue: TaskQueue = TaskQueue()
        self.mj_scene.start()
        if host_address is not None:
            self.publisher = SFPublisher(self.mj_scene, host_address)

    def create_scene(
        self,
        dt=0.001,
        render=Scene.RenderMode.HUMAN,
        surrounding=None,
    ) -> MjScene:
        self.object_dict = self.create_objects()
        return self.sim_factory.create_scene(
            object_list=self.object_dict.values(),
            dt=dt,
            render=render,
            surrounding=surrounding,
        )

    def reset_objects(self, obj_state: Dict):
        for obj_name, obj_state in obj_state.items():
            self.mj_scene.set_obj_pos_and_quat(
                obj_name=obj_name,
                new_pos=obj_state["pos"],
                new_quat=obj_state["quat"],
            )

    def reset_robots(self, robot_state: Dict):
        for robot_name, robot_state in robot_state.items():
            self.robot_dict[robot_name].beam_to_joint_pos(
                robot_state["joint_pos"]
            )

    def add_task(self, func: Callable[..., None], *args: Any) -> None:
        self.task_queue.put((func, args))

    def reset_in_the_main_thread(self):
        self.task_queue.put((self.reset, None))

    @abc.abstractmethod
    def create_robots(self) -> Dict[str, MjRobot]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_objects(self) -> Dict[str, MujocoObject]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def before_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def after_step(self):
        raise NotImplementedError

    def assign_controller(self, controller_dict: Dict[str, ControllerBase]):
        self.controller_dict = controller_dict
        for name, robot in self.robot_dict.items():
            self.controller_dict[name].executeController(
                robot, maxDuration=1000, block=False
            )

    def execute_task_queue(self):
        while not self.task_queue.empty():
            task, args = self.task_queue.get(block=False)
            if args is None:
                task()
            else:
                task(*args)

    def next_step(self):
        self.before_step()
        self.mj_scene.next_step()
        self.execute_task_queue()
        self.after_step()

    def run(self):
        self.reset()
        while True:
            self.next_step()
