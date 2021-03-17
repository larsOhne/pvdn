//#include "python3.8/Python.h"
#include "python2.7/Python.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <vector>
#include "HeadLampObject.h"
#include <numeric>


// using namespace std;

extern "C"
{
    int min (int a, int b)
    {
        int result;
        if (a < b)
        {
            result = a;
        }
        else
        {
            result = b;
        }
        return result;
    }


    int max (int a, int b)
    {
        int result;
        if (a > b)
        {
            result = a;
        }
        else
        {
            result = b;
        }
        return result;
    }

    int bin_img_at(uint8_t *img, int x, int y, int height, int width)
    {
        int index = y * width + x;
        return img[index];
    }

    float img_at(float *img, int x, int y, int height, int width)
    {
        int index = y * width + x;
        return img[index];
    }

    float ii_at(float *img, int x, int y, int height, int width)
    {
        int index = y * width + x;
        return img[index];
    }

    double threshold (float *img, float *ii, int x, int y, int h, int w, double k, int window, float eps)
    {
        int xmin = std::max(0, x - window / 2);
        int ymin = std::max(0, y - window / 2);
        int xmax = std::min(w-1, x + window / 2);
        int ymax = std::min(h-1, y + window / 2);
        
        int area = (xmax - xmin) * (ymax - ymin);

        double m;
        m = (double)ii_at(ii, xmin, ymin, h, w);
        m = m + (double)ii_at(ii, xmax, ymax, h, w);
        m = m - (double)ii_at(ii, xmax, ymin, h, w);
        m = m - (double)ii_at(ii, xmin, ymax, h, w);
        m = m / (double)area;

        // local deviation
        double delta = img_at(img, x, y, h, w) - m;

        // dynamic threshold + small number for numerial stability
        double t = m * (1. + k * (1.0 - delta / (1.0 - delta + eps)));
        
        return t;
    }


    void binarize (float *img, uint8_t *bin_img, float *ii, int h, int w, int window, float k, float eps)
    {
        int x = 0;
        int y = 0;

        int i = 0;

        // iterate over all pixels in image
        for (y = 0; y < h; y++)
        {
            for (x = 0; x < w; x++)
            {
                bin_img[y*w + x] = (uint8_t)((double)(img_at(img, x, y, h, w)) > threshold(img, ii, x, y, h, w, (double)k, window, eps));
            }
        }
    }


    std::vector<int> sort_indexes(std::vector<int> &v) {

    // initialize original index locations
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    stable_sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] < v[i2];});

    return idx;
    }


    std::vector<int> filter_idxs(int *bb1, std::vector<int> bb_list_orig, std::vector<int> idxs, int last, std::vector<int> area, double overlap_thresh)
    {
        int i = 0;
        int offset = 0;
        int h;
        int w;
        double overlap;
        int xx1, yy1, xx2, yy2;

        std::vector<int> swap_idxs;
        std::vector<int> erase_idxs = {last};
        for (i = 0; i < last; i++)
        {
            offset = i * 4;

            xx1 = std::max(bb1[0], bb_list_orig.at(idxs.at(i)*4+0));
            yy1 = std::max(bb1[1], bb_list_orig.at(idxs.at(i)*4+1));
            xx2 = std::min(bb1[2], bb_list_orig.at(idxs.at(i)*4+2));
            yy2 = std::min(bb1[3], bb_list_orig.at(idxs.at(i)*4+3));


            h = std::max(0, xx2 - xx1 + 1);
            w = std::max(0, yy2 - yy1 + 1);
            overlap = (double)(w * h) / (double)area.at(idxs.at(i));

            if (overlap <= overlap_thresh)
            {
                swap_idxs.push_back(idxs.at(i));

            }
            else{
                erase_idxs.push_back(i);
            }
        }

        return swap_idxs;
    }


    std::vector<HeadLampObject*> non_maximum_supression(std::vector<HeadLampObject*> bbs, int length, double overlap_thresh)
    {
        std::vector<int> bboxes(length*4);
        std::vector<int> area(length);
        std::vector<int> y2(length);

        // convert HeadLampObjects to bounding boxes
        // also compute the area of each bounding box
        int i = 0;
        int offset = 0;
        for (i = 0; i < length; i++)
        {
            int *bb = bbs.at(i)->bbox();
            offset = i * 4;
            bboxes.at(offset+0) = bb[0];
            bboxes.at(offset+1) = bb[1];
            bboxes.at(offset+2) = bb[2];
            bboxes.at(offset+3) = bb[3];

            area.at(i) = (bb[2] - bb[0] + 1 ) * (bb[3] - bb[1] + 1);
            y2.at(i) = bb[3];
        }

        i = 0;
        // init list of picked indexes
        std::vector<int> pick;

        // sort bboxes by the bottom-right y-coordinate
        std::vector<int> idxs;
        idxs = sort_indexes(y2);

        int h = 0;
        int w = 0;
    
        int bb[4];
        int idx = 0;
        while (idxs.size() > 0)
        {
            // grab the last index in the indexes list and add the
            // index value to the list of picked indexes
            int last = idxs.size() - 1;
            idx = idxs.at(last);
            pick.push_back(idx);

            // find the largest (x, y) coordinates for the start of
            // the bounding box and the smallest (x, y) coordinates
            // for the end of the bounding box
            bb[0] = bboxes.at(idx*4+0);
            bb[1] = bboxes.at(idx*4+1);
            bb[2] = bboxes.at(idx*4+2);
            bb[3] = bboxes.at(idx*4+3);

            idxs = filter_idxs(bb, bboxes, idxs, last, area, overlap_thresh);
        }

        std::vector<int> picked_bbs(pick.size()*4);
        std::vector<HeadLampObject*> proposed(pick.size());
        
        i = 0;
        offset = 0;
        idx = 0;
        int xx1, yy1, xx2, yy2;
        for (i = 0; i < pick.size(); i++)
        {
            idx = pick.at(i);
            offset = idx * 4;
            
            xx1 = bboxes.at(idx*4+0);
            yy1 = bboxes.at(idx*4+1);
            xx2 = bboxes.at(idx*4+2);
            yy2 = bboxes.at(idx*4+3);

            int bb[4] = {xx1, yy1, xx2, yy2};
            HeadLampObject *hlo = new HeadLampObject();
            hlo->from_bb(bb);
            proposed.at(i) = hlo;
        }
        
        return proposed;
    }


    void find_proposals(int *final_proposals, uint8_t *bin_img, int h, int w, int padding, int nms_distance)
    {
        double overlap_thresh = 0.05;    // TODO: make it parameter
        int x = padding;
        int y = padding;
        int i;
        bool match_found;
        std::vector<HeadLampObject*> to_consider = {};
        std::vector<HeadLampObject*> proposals = {};

        // flood fill algorithm
        for (y = padding; y < (h - padding); y++)
        {
            std::vector<HeadLampObject*> updated = {};
            for (x = padding; x < (w - padding); x++)
            {
                if (bin_img_at(bin_img, x, y, h, w) == 1)
                {
                    match_found = false;
                    i = 0;
                    for (i = 0; i < to_consider.size(); i++)
                    {
                        if (to_consider.at(i)->is_neighbor(x, y, nms_distance) == true)
                        {
                            match_found = true;

                            // add point to proposal
                            int *proposal_bb;
                            proposal_bb = to_consider.at(i)->bbox();
                            int xmin = std::min(proposal_bb[0], x);
                            int ymin = std::min(proposal_bb[1], y);
                            int xmax = std::max(proposal_bb[2], x + 1);
                            int ymax = std::max(proposal_bb[3], y + 1);

                            int new_bb[4] = {xmin, ymin, xmax, ymax};
                            to_consider.at(i)->from_bb(new_bb);
                            updated.push_back(to_consider.at(i));

                            // no need to look for more
                            break;
                        }
                    }
                    if (match_found == false)
                    {
                        // create new HeadLampObject with default height and width of 3
                        HeadLampObject *new_proposal = new HeadLampObject();
                        new_proposal->center_pos_x = x;
                        new_proposal->center_pos_y = y;
                        new_proposal->width = 3;
                        new_proposal->height = 3;
                        new_proposal->conf = 1;
                        proposals.push_back(new_proposal);
                        updated.push_back(new_proposal);
                    }
                }
            }
            to_consider = updated;
        }

        // apply non maximum supression
        std::vector<HeadLampObject*> picked;
        std::vector<HeadLampObject*> filtered;
        picked = non_maximum_supression(proposals, proposals.size(), overlap_thresh);

        // filter out very small proposals
        i = 0;
        for (i = 0; i < picked.size(); i++)
        {
            if (picked.at(i)->width * picked.at(i)->height > 10)
            {
                if (picked.at(i)->width > 5)
                {
                    picked.at(i)->source = 0;
                    filtered.push_back(picked.at(i));
                }
            }
        }

        // convert HeadLampObjects to bounding boxes
        i = 0;
        for (i = 0; i < filtered.size(); i++)
        {
            int *bb = filtered.at(i)->bbox();
            final_proposals[i*4+0] = bb[0];
            final_proposals[i*4+1] = bb[1];
            final_proposals[i*4+2] = bb[2];
            final_proposals[i*4+3] = bb[3];
        }

        // call destructor on all HeadLampObjects
        i = 0;
        for (i = 0; i < proposals.size(); i++)
        {
            if (i < picked.size())
            {
                delete[] picked.at(i);
            }
            delete[] proposals.at(i);
        }
    }
}