import abc

from .recorder import RecordData


class DataReplayer(abc.ABC):

    def __init__(self):
        self.index = 0
        self.sim_length = 0
        # self.load_recod_data(record_data_path)

    def load_recod_data(self, record_data_path: str):
        self.record_data = RecordData()
        self.record_data.load_from_file(record_data_path)
        sequence_length = self.record_data.header.sequence_length
        skip_step = self.record_data.header.skip_step
        self.sim_length = sequence_length * skip_step

    @abc.abstractmethod
    def create_simulator(self):
        pass

    @abc.abstractmethod
    def replay(self):
        while not self.isSequenceFinished():
            self.replay_step()

    @abc.abstractmethod
    def replay_step(self):
        raise NotImplementedError

    def isSequenceFinished(self):
        return self.index >= self.sim_length
