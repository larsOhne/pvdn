from typing import List, Tuple


class Keypoint:
    """
    A simple data structure, describing any Keypoint that has a position and a unique ID
    """
    position: Tuple[int, int]
    oid: int

    def __init__(self, position: Tuple[int, int] = None, oid: int = -1):
        """
        Creates a new Keypoint
        :param position: the position of this keypoint
        :param oid: a unique id
        """
        self.position = position
        self.oid = oid


class Instance(Keypoint):
    vehicle: Keypoint
    direct: bool
    rear: bool

    def __init__(self, position: Tuple[int, int] = None, oid: int = 0, vehicle: Keypoint = None,
                 direct: bool = False, rear: bool = False):
        super(Instance, self).__init__(position, oid)

        self.vehicle = vehicle
        self.direct = direct
        self.rear = rear

    def from_dict(self, d: dict, vehicle: Keypoint):
        """
        builds this object from a dictionary, using the nomenclature defined by the annotation tool. This method is
        meant to be used when parsing annotation files, so that the vehicle object is available beforehand.
        :param d: a dictionary
        :param vehicle: The corresponding vehicle
        :return: Nothing
        """
        self.position = tuple(d["pos"])
        self.oid = d["iid"]
        self.vehicle = vehicle
        self.direct = d["direct"] if "direct" in d.keys() else False
        self.rear = d["rear"] if "rear" in d.keys() else False

    def __eq__(self, other):
        if type(other) is not Instance:
            return False
        else:
            return self.oid == other.oid and self.vehicle.oid == other.vehicle.oid

    def __repr__(self):
        return "Instance: iid={}, oid={}, direct={}, rear={}".format(self.oid, self.vehicle.oid, self.direct, self.rear)


class Vehicle(Keypoint):
    direct: bool
    instances: List[Instance]

    def _init__(self, position: Tuple[int, int] = None, oid: int = 0, direct: bool = False,
                instances: List[Instance] = [], *args, **kwargs):
        super(Vehicle, self).__init__(position, oid)

        # initialize subclass attributes
        self.direct = direct
        self.instances = instances

    def from_dict(self, d: dict):
        """Builds this object from a dictionary.

        This method is meant to be used when parsing annotation files.
        :param d: a dictionary
        :return: Nothing
        """
        # read vehicle specific data
        self.position = tuple(d["pos"])
        self.oid = d["oid"]

        self.direct = d["direct"] if "direct" in d.keys() else False

        # create instances
        self.instances = []
        for inst_d in d["instances"]:
            inst = Instance()
            inst.from_dict(inst_d, self)
            self.instances.append(inst)

    def __eq__(self, other):
        if type(other) is not Vehicle:
            return False
        else:
            return self.oid == other.oid

    def __repr__(self):
        return "Vehicle: pos:{}, #inst: {}, {}".format(self.position,
                                                       len(self.instances),
                                                       "direct" if self.direct else "indirect")
