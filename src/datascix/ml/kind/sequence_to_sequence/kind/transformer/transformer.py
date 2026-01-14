
from datascix.ml.kind.sequence_to_sequence.sequence_2_sequence import Sequence2Sequence
from utilix.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from utilix.os.file_system.file.file import File as OsFile
from utilix.os.file_system.path.file import File as FilePath


class Transformer(Sequence2Sequence):
    pass

if __name__ == '__main__':
    file_path = FilePath("/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic_sliced_from_1_to_300000/time_position_sequence.npz")
    os_file = OsFile.init_from_path(file_path)
    storage = NpMultiValued(os_file, False)
    storage.load()
    ram = storage.get_ram()
    print(ram)