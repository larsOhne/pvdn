#include "HeadLampObject.h"


HeadLampObject::HeadLampObject(){
    center_pos_x = 0;
    center_pos_y = 0;
    width = 0;
    height = 0;
    conf = 0;
    
    source = -1;
}

HeadLampObject::HeadLampObject(int x, int y, int w, int h, int c)
{
    center_pos_x = x;
    center_pos_y = y;
    width = w;
    height = h;
    conf = c;

    source = -1;
}


bool HeadLampObject::is_neighbor(int x, int y, int nms_distance)
{
    int *vec;
    vec = distance_vec_to_point(x, y);
    int distance = std::max(std::abs(vec[0]), std::abs(vec[1]));

    return distance < nms_distance;
}


int *HeadLampObject::bbox()
{
    int *bb = new int[4];
    
    bb[0] = ceil((double)center_pos_x - (double)width/2.0);
    bb[1] = ceil((double)center_pos_y - (double)height/2.0);
    bb[2] = ceil((double)center_pos_x + (double)width/2.0);
    bb[3] = ceil((double)center_pos_y + (double)height/2.0);


    // bb[0] = center_pos_x - width/2;
    // bb[1] = center_pos_y - height/2;
    // bb[2] = center_pos_x + width/2;
    // bb[3] = center_pos_y + height/2;
    return bb;
}


int *HeadLampObject::distance_vec_to_point(int x, int y)
{
    int *bb;
    bb = bbox();

    int dx = std::max(bb[0] - x, x - bb[2]);
    dx = std::max(dx, 0);

    int dy = std::max(bb[1] - y, y - bb[3]);
    dy = std::max(dy, 0);

    int *vec = new int[2];
    vec[0] = dx;
    vec[1] = dy;
    
    return vec;
}


void HeadLampObject::from_bb(int bb[4])
{
    int new_w = bb[2] - bb[0];
    int new_h = bb[3] - bb[1];

    int new_center_x = floor((double)bb[0] + (double)new_w / 2.0);
    int new_center_y = floor((double)bb[1] + (double)new_h / 2.0);

    // update instance variables
    center_pos_x = new_center_x;
    center_pos_y = new_center_y;
    width = new_w;
    height = new_h;
}