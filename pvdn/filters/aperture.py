from pvdn.filters import Filter
from pvdn import ImageInformation


class NightCycleFilter(Filter):
    """ Filters images taken with the camera's night aperture

    Requires that the camera configuration has a cycle named "night"
    """

    def evaluate(self, image_info: ImageInformation):
        return image_info.camera_config.cycle == "night"


class DayCycleFilter(NightCycleFilter):
    """ Filters images taken with the camera's day aperture

    Requires that the camera configuration has a cycle named "day"
    """

    def evaluate(self, image_info: ImageInformation):
        return not super(DayCycleFilter, self).evaluate(image_info)
