#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "math.h"

class HeadLampObject
{
public:
    HeadLampObject(int x, int y, int w, int h, int c);
    HeadLampObject();
    int center_pos_x;
    int center_pos_y;
    int width;
    int height;
    double conf;

    int source;
    
    bool is_neighbor(int x, int y, int nms_distance);
    int *bbox();
    int *distance_vec_to_point(int x, int y);
    void from_bb(int bb[4]);
};