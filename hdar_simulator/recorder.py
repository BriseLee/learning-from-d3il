import abc
import os
import numpy as np
from typing import List,  Dict
from datetime import datetime
from dataclasses import dataclass
import pickle
import threading

# from simpub.core.log import logger


ObjectData = Dict[str, List[float]]
FrameData = Dict[str, ObjectData]


@dataclass
class RecordDataHeader:
    type: str
    simulator: str
    task: str
    skip_step: int
    simulation_dt: float
    sequence_length: int = 0


class RecordData:

    def __init__(self, header: RecordDataHeader = None) -> None:
        self.header: RecordDataHeader = header
        self.init_state: FrameData = {}
        self.frames: List[FrameData] = []
        self.state: Dict[str, Dict[str, np.ndarray]] = None

    def append_frame(self, frame: FrameData):
        self.frames.append(frame)
        self.header.sequence_length += 1

    def to_state_array(self):
        seq_length = self.header.sequence_length
        frames = self.frames
        state = {}
        for name, date in frames[0].items():
            state[name] = {}
            for attr, value in date.items():
                assert isinstance(value, np.ndarray)
                state[name][attr] = np.zeros((seq_length, len(value)))
        for index, frame in enumerate(frames):
            for name, date in frame.items():
                for attr, value in date.items():
                    state[name][attr][index] = np.array(value)
        self.state = state
        return self.state

    def save_to_file(self, save_path: str):
        if self.state is None:
            self.to_state_array()
        with open(save_path, "wb") as f:
            if self.state is None:
                self.to_state_array()
            pickle.dump(
                {
                    "header": self.header,
                    "init_state": self.init_state,
                    "state": self.state,
                }, f)

    def load_from_file(self, file_path: str):
        with open(file_path, "rb") as f:
            pickle_data = pickle.load(f)
        self.header = pickle_data["header"]
        # self.header = RecordDataHeader(
        #     pickle_data["header"]["type"],
        #     pickle_data["header"]["task"],
        #     pickle_data["header"]["skip_step"],
        #     pickle_data["header"]["simulation_dt"],
        #     pickle_data["header"]["sequence_length"],
        # )
        self.init_state = pickle_data["init_state"]
        self.state = pickle_data["state"]


# class Recorder(abc.ABC):

#     def __init__(
#         self,
#         task_name: str,
#         save_root_path: str,
#         # record_mode: bool,
#         skip_step: int,
#     ) -> None:
#         super().__init__()
#         # name the demonstration by date and time
#         record_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         self.task_name = task_name
#         self.save_path = os.path.join(
#             save_root_path,
#             f"{task_name}_{record_time}")
#         if not os.path.exists(self.save_path):
#             os.makedirs(self.save_path)
#         # self.record_mode = record_mode
#         self.skip_step = skip_step
#         # recording flag
#         self.on_recording = False
#         self.saving_thread = None
#         # counter for the demonstration and frames
#         self.demo_counter = 0
#         self.skip_counter = 0

#     def start_record(self):
#         # if not self.record_mode or self.on_recording:
#         #     return
#         if self.on_recording:
#             return
#         logger.info("Start recording")
#         self._start_record()
#         self.skip_counter = 0
#         self.on_recording = True

#     def stop_record(self):
#         # if not self.record_mode or not self.on_recording:
#         #     return
#         if not self.on_recording:
#             return
#         logger.info("Stop recording")
#         self.on_recording = False

#     def save_record(self):
#         # if not self.record_mode:
#         #     return
#         if self.saving_thread is not None:
#             return
#         self.stop_record()
#         file_name = "{}_{:03d}.pkl".format(self.task_name, self.demo_counter)
#         self.saving_thread = threading.Thread(
#             target=self._save_record, kwargs={"file_name": file_name}
#         ).start()
#         logger.info(f"Saving record to {file_name}")
#         self.demo_counter += 1

#     @abc.abstractmethod
#     def _start_record(self):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def _save_record(self, file_name: str):
#         logger.info(f"Finishing saving record to {file_name}")
#         self.saving_thread = None

#     @abc.abstractmethod
#     def _record(self):
#         raise NotImplementedError

#     def record(self):
#         # if not self.record_mode or not self.on_recording:
#         #     return
#         if not self.on_recording:
#             return
#         self.skip_counter += 1
#         if self.skip_counter % self.skip_step == 0:
#             self._record()
#             self.skip_counter = 0
#             return
