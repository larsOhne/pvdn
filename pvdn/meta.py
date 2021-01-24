from copy import deepcopy
from enum import IntEnum
from typing import List, Optional, Dict, Any


class Direction(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "direction"

    See Marenzi (2018) p. 82
    """
    DIR_MOENSHEIM = 0
    DIR_IPTINGEN = 1
    DIR_EZW = 2
    ENTRANCE_MOENSHEIM = 3
    ENTRANCE_IPTINGEN = 4
    ENTRANCE_EZW = 5


class StreetStyle(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "street style"

    See Marenzi (2018) p. 82
    """
    RIGHT_TURN = 0
    LEFT_TURN = 1
    RIGHT_LEFT_TURN = 2
    LEFT_RIGHT_TURN = 3
    LONG_RIGHT_TURN = 4
    LONG_LEFT_TURN = 5
    STRAIGHT_STREET = 6


class DriverBehaviour(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "Lightdim"

    See Marenzi (2018) p. 82
    """
    CORRECT = 0
    TOO_LATE = 1
    TOO_EARLY = 2
    INCORRECT = 3
    EXCEPTIONAL = 4


class RoadType(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "road construction"

    See Marenzi (2018) p. 82
    """
    SINGLE_LANE_ROAD_MARKING = 0
    SINGLE_LANE_NO_ROAD_MARKING = 1


class Visibility(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "visibility"

    (meaning: how much of the vehicle's environment can be perceived)
    See Marenzi (2018) p. 82
    """
    WIDE_VIEW = 0
    SHORT_VIEW = 1


class WeatherConditions(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "weather"

    See Marenzi (2018) p. 82
    """
    DRY = 0
    WET = 1
    RAIN = 2
    SNOW = 3
    FOG = 4


class EnvironmentLighting(IntEnum):
    """ Enum-Class which specifies all possible labels and values for attribute "environment lighting"

    See Marenzi (2018) p. 82
    """
    BRIGHT = 0
    DUSK = 1
    DARK = 2


class CameraConfiguration:
    """ Describes the camera setup used to capture an image.

    It mainly captures:
    - The physical camera used to take the image,
    - the exposure used.

    """
    cid: int
    camera: Optional[str]
    cycle: Optional[str]

    def __init__(self, cid: int = 0, camera: str = None, cycle: str = None):
        """ Used to initialize this object.

        Meant to be used only together with a dataset class

        :param cid: an unique ID assigned to the physical camera
        :param camera: a textual description of the physical camera
        :param cycle: the exposure cycle used in this configuration
        """
        self.cid = cid
        self.camera = camera
        self.cycle = cycle

    def __repr__(self):
        return "{} at {}".format(self.camera, self.cycle)

    def to_plain_dict(self) -> Dict[str, Any]:
        """ Compresses the object into a minimal dictionary

        :return: the dictionary
        :rtype: Dict[str,Any]
        """
        plain_dict = deepcopy(vars(self))
        del plain_dict["cid"]
        return plain_dict


class Category:
    def __init__(self, supercategory=None, id=None, name=None):
        self.supercategory = supercategory
        self.id = id
        self.name = name


class Annotation:
    def __init__(self, id, category, image_id, **kwargs):
        # TODO make category object
        self.id = id
        self.category = category
        self.image_id = image_id
        self.tags = kwargs

    def __repr__(self):
        return "Annotation(id={}, category={}, image_id={}, tags={})".format(self.id, self.category, self.image_id,
                                                                             self.tags
                                                                             )

    def to_plain_dict(self):
        """ Compresses the object into a minimal dictionary

        :return: the dictionary
        :rtype: Dict[str,Any]
        """
        plain_dict = {
            "image_id": self.image_id,
            "id": self.id,
            "category": self.category
        }

        # add additional tags
        plain_dict = {**plain_dict, **self.tags}

        return plain_dict


class ImageInformation:
    def __init__(self, licence: int = 0, file_name: str = None, height: int = 0,
                 width: int = 0, date_captured: str = None, timestamp: int = 0,
                 id: int = 0, camera_config: CameraConfiguration = None, sequence=None, **kwargs):
        self.licence = licence
        self.file_name = file_name
        self.height = height
        self.width = width
        self.date_captured = date_captured
        self.timestamp = timestamp
        self.id = id
        self.camera_config = camera_config
        self.annots = []
        self.sequence = sequence

    def to_plain_dict(self):
        """ Compresses the object into a minimal dictionary

        :return: the dictionary
        :rtype: Dict[str,Any]
        """
        plain_dict = deepcopy(vars(self))
        plain_dict["camera_configuration"] = self.camera_config.cid
        del plain_dict["camera_config"]
        del plain_dict["annots"]
        del plain_dict["sequence"]

        return plain_dict

    def add_annotation(self, annot: Annotation):
        self.annots.append(annot)

    def __repr__(self):
        return "image id: {}, file: {}, size: ({},{}), camera: {}, annots: {}".format(self.id, self.file_name,
                                                                                      self.width,
                                                                                      self.height,
                                                                                      self.camera_config, self.annots)


class SequenceInformation:
    """ A class describing metadata of an image sequence.

    Usage is mostly restricted to the dataset classes to provide an object-oriented schema for loading
    annotations and metadata
    """

    def __init__(self, id: int = 0, dir: str = "", start_time: int = 0, end_time: int = 0,
                 num_images: int = 0, image_ids: List[int] = [], proband_id: int = 0, sector: int = 0,
                 direction: int = 0, street_style: int = 0, dome: bool = False, proband_behaviour: int = 0,
                 road_type: int = 0, view: int = 0, weather: int = 0, environment_lighting: int = 0) -> None:
        """ Initializes a compact object representing metadata about a specific sequence.

        :param id: The sequence's ID.
        :param dir: The path (relative to the dataset's base directory) to the directory containing the images
        in this sequence .
        :param start_time: The relative time in ms when the sequence starts.
        :param end_time: The relative time in ms when the sequence ends.
        :param num_images: The total number of images in this sequence.
        :param image_ids: A list containing all image IDs in this sequence. Sorting before passing is not necessary.
        :param proband_id: A unique ID assigned to the driver of the camera vehicle.
        :param sector: A number representing the sector (position on the testing route) this sequence was recorded.
        :param direction: A number representing the direction the vehicle is driving in this sequence.
        :param street_style: A number representing the type of the road markings.
        :param dome: True if the camera vehicle approaches a dome
        :param proband_behaviour: A number describing if the driver acted providently
        :param road_type: TODO
        :param view: A number representing the drivers field of view
        :param weather: A number representing the weather during this sequence
        :param environment_lighting: A number representing the ambient lights in this sequence
        """
        # initialize instance variables
        self.sid = id
        self.directory = dir
        self.start_time = start_time
        self.end_time = end_time
        self.num_images = num_images
        self.image_ids = sorted(image_ids)
        self.driver_id = proband_id
        self.sector = sector
        self.direction = Direction(direction)
        self.street_style = StreetStyle(street_style)
        self.dome = dome
        self.driver_behaviour = DriverBehaviour(proband_behaviour)
        self.road_type = RoadType(road_type)
        self.view = Visibility(view)
        self.weather = WeatherConditions(weather)
        self.environment_lighting = EnvironmentLighting(environment_lighting)

        self.used = False

    def sequence_contains(self, image_id: int) -> bool:
        """
        Checks whether an image represented by its ID belongs to this sequence
        :param image_id: int, the image's id
        :return: True if image belongs to Sequence, False otherwise
        """
        return image_id in self.image_ids

    def to_plain_dict(self, ):
        """ Compresses the object into a minimal dictionary

        :return: the dictionary
        :rtype: Dict[str,Any]
        """
        plain_dict = deepcopy(vars(self))
        for key in plain_dict:
            value = plain_dict[key]
            if isinstance(value, IntEnum):
                plain_dict[key] = int(value)

        del plain_dict["used"]
        return plain_dict
