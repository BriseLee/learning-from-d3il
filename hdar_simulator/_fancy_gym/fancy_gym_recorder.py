from ..recorder import Recorder


class FancyGymRecorder(Recorder):
    def __init__(
        self,
        task_name: str,
        save_root_path="./ARHumanDemoData/",
        record_mode=False,
        record_step=1,
    ) -> None:
        super().__init__(task_name, save_root_path, record_mode, record_step)

    def _save_record(self, file_name):
        pass

    def record(self, data):
        pass
