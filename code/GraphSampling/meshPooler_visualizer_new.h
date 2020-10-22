//
// Created by zhouyi on 6/13/19.
//

#ifndef POINTSAMPLING_MESHPOOLER_VISUALIZER_NEW_H
#define POINTSAMPLING_MESHPOOLER_VISUALIZER_NEW_H

#endif //POINTSAMPLING_MESHPOOLER_VISUALIZER_H


//#include "meshPooler.h"
#include "meshCNN.h"
#include <opencv2/opencv.hpp>
#include <math.h>

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}

double cos_sin_to_radius(double cos, double sin )
{
    double arcsin = asin(sin);
    double pi = 3.1415926585;
    double radius=0;
    if(sin>=0)
    {
        if(cos>=0)
        {
            radius = arcsin;
        }
        else
        {
            radius = pi-arcsin;
        }
    }
    else
    {
        if(cos>=0)
            radius= 2*pi + arcsin;
        else
        {
            radius = pi - arcsin;
        }
        
    }
    
}

class MeshPooler_Visualizer
{
public:
    
    void save_colored_obj_receptive_field(const string &file_dir, Mesh mesh, MeshCNN meshCNN, int layer_index)
    {
        cout<<"Start visualizing layer "<<layer_index<<"\n";
        
        cout<<"Initialize.\n";
        
        int center_point_num = meshCNN._meshPoolers[layer_index]._pool_map.size();
        cout<<"Center point number: "<<center_point_num<<"\n";
        vector<vector<float>> last_receptions_lst;

        for(int i=0;i<center_point_num;i++)
        {
            vector<float> receptions;
            for (int j=0;j<center_point_num;j++)
            {
                if(i!=j)
                    receptions.push_back(0);
                else
                    receptions.push_back(1);
            }
            last_receptions_lst.push_back(receptions);
        }

        cout<<"compute receptions_lst.\n";

        vector<vector<float>> receptions_lst;

        

        for(int n=layer_index; n>=0;n--)
        {
            //cout<<"Layer "<<n<<"\n";
            int input_point_num = meshCNN._meshPoolers[n]._connection_map.size();
            int output_point_num = meshCNN._meshPoolers[n]._pool_map.size();

            receptions_lst.clear();

            for(int i=0; i<input_point_num;i++)
            {
                vector<float> receptions;
                for(int k = 0; k<center_point_num; k++)
                {
                    receptions.push_back(0);
                }

                receptions_lst.push_back(receptions);
            }
            
            //recepoints_lst input_point_num*center_point_num
            //last_receptions_lst  output_point_num*center_point_num
            /*
            for(int i=0; i<output_point_num;i++)
            {
                vector<Int2> connected_old_index_lst = meshCNN._meshPoolers[n]._pool_map[i]; 
                int connected_vertices_num = connected_old_index_lst.size();

                vector<float> last_receptions = last_receptions_lst[i];

                for (int j=0;j<connected_vertices_num;j++)
                {
                    int old_index= connected_old_index_lst[j][0];

                    for(int k=0; k<center_point_num;k++)
                    {
                        receptions_lst[old_index][k] += last_receptions[k]/connected_vertices_num;
                    }
                }
            }*/
            for(int i=0;i<input_point_num;i++)
            {
                vector<Int2> connected_output_lst =  meshCNN._meshPoolers[n]._unpool_map[i];
                int  connected_vertices_num = connected_output_lst.size();
                for(int j=0;j<connected_vertices_num;j++)
                {
                    int connected_vertex_id = connected_output_lst[j][0];
                    vector<float> last_receptions = last_receptions_lst[connected_vertex_id];
                    for(int k=0;k<center_point_num;k++)
                    {
                        receptions_lst[i][k] += last_receptions[k]/ float(connected_vertices_num) ;
                    }
                }
            }

            last_receptions_lst = receptions_lst;
        }

        for(int i =0;i<receptions_lst.size(); i++)
        {
            float sum = 0; 
            for(int k=0;k<center_point_num;k++)
            {
                sum+=receptions_lst[i][k];
            }
            if(abs(sum-1.0)>0.0001)
                cout<<"ERROR "<<i<<" "<<sum<<"\n";
        }

        cout<<"receptions_lst"<<receptions_lst.size()<<"\n";

        cout<<"save colored obj for visualizing pooling receptive field\n";


        //create color plate
        vector<Vec3<float>> color_plate; //the vector (cos(h), sin(h))
        vector<Vec3<float>> color_plate_rgb;
        for (int i=0;i<center_point_num;i++)
        {
            float h = (i)*3.1415926585*2/center_point_num;// - 3.1415926585/center_point_num/2;

            color_plate.push_back(Vec3<float>(cos(h), sin(h),0));

            //float h2 = cos_sin_to_radius(cos(h), sin(h));
            hsv hsv_color;
            hsv_color.h= h*180/3.1415926585;
            hsv_color.s = 1;
            hsv_color.v=1;
            rgb rgb_color = hsv2rgb(hsv_color);

            color_plate_rgb.push_back(Vec3<float>(rgb_color.r, rgb_color.g, rgb_color.b));
            //cout<<h/3.1415926585*180<<" "<<h2/3.1415926585*180<<" "<<cos(h)<<" "<<sin(h)<<"\n";
        }
        Vec3<float> color_2 = color_plate_rgb[2];
        color_plate_rgb[2] = color_plate_rgb[1];
        color_plate_rgb[1]=color_2;

        //for(int i=0;i<center_point_num; i++)
        //    cout<<color_plate_rgb[i][0]<<" "<<color_plate_rgb[i][1]<<" "<<color_plate_rgb[i][2]<<"\n";
        
        //random_shuffle(color_plate.begin(), color_plate.end());

        vector<Vec3<float>> colors;
        int max_color_num=0;
        for (int i=0;i<receptions_lst.size();i++)
        {
            
            Vec3<float> color_sum(0,0,0);
            float color_num=0;
            for(int k=0;k<receptions_lst[i].size();k++)
            {
                color_sum  += color_plate_rgb[k]*receptions_lst[i][k];
                color_num += receptions_lst[i][k];
            }
            Vec3<float> h_vector=color_sum/color_num;
            
            /*
            
            int max_k=0;
            float max_weight=0;

            for(int k=0;k<receptions_lst[i].size();k++)
            {
               if(receptions_lst[i][k]>max_weight)
               {
                    max_weight = receptions_lst[i][k];
                    max_k = k;
               }
            }
            Vec3<float> h_vector = color_plate_rgb[max_k];
            */

            /*
            h_vector.Normalize();
                
            //cout<<"color_num "<<color_num<<"\n";
            //if(color_num>max_color_num)
            //    max_color_num=color_num;
            
            double h = cos_sin_to_radius(h_vector[0], h_vector[1]);
            
            
            hsv hsv_color;
            hsv_color.h = h*180/3.1415926585;
            if(hsv_color.h<0)
                hsv_color.h = hsv_color.h+180;
            hsv_color.s=1;
            hsv_color.v=1;
            
            
            rgb rgb_color = hsv2rgb(hsv_color);

            

            colors.push_back(Vec3<float>(rgb_color.r, rgb_color.g, rgb_color.b));
            */
            colors.push_back(h_vector);
        }

        //for(int i=0;i<colors.size();i++)
        //    colors[i]=colors[i]/max_color_num;

        mesh.SaveOBJ(file_dir, mesh.points, colors, mesh.triangles);
    }


    void save_center_mesh(const string &file_dir, Mesh mesh, MeshCNN meshCNN, int layer_index)
    {
        vector<vector<int>> center_center_map = meshCNN._meshPoolers[layer_index]._center_center_map;
        int center_num = center_center_map.size();
        vector<int> center_old_index_lst;
        for (int i=0;i<center_num;i++)
            center_old_index_lst.push_back(i);

        for(int n=layer_index;n>=0;n--)
        {
            vector<vector<Int2>> pool_lst= meshCNN._meshPoolers[n]._pool_map;
            for(int i=0;i<center_num;i++)
            {
                center_old_index_lst[i] = pool_lst[center_old_index_lst[i]][0][0];
            }
        }

         //create color plate
        vector<Vec3<float>> color_plate; //the vector (cos(h), sin(h))
        vector<Vec3<float>> color_plate_rgb;
        for (int i=0;i<center_num;i++)
        {
            float h = (i)*3.1415926585*2/center_num;// - 3.1415926585/center_point_num/2;

            color_plate.push_back(Vec3<float>(cos(h), sin(h),0));

            //float h2 = cos_sin_to_radius(cos(h), sin(h));
            hsv hsv_color;
            hsv_color.h= h*180/3.1415926585;
            hsv_color.s = 1;
            hsv_color.v=1;
            rgb rgb_color = hsv2rgb(hsv_color);

            color_plate_rgb.push_back(Vec3<float>(rgb_color.r, rgb_color.g, rgb_color.b));
            //cout<<h/3.1415926585*180<<" "<<h2/3.1415926585*180<<" "<<cos(h)<<" "<<sin(h)<<"\n";
        }
        Vec3<float> color_2 = color_plate_rgb[2];
        color_plate_rgb[2] = color_plate_rgb[1];
        color_plate_rgb[1]=color_2;



        vector<Vec3<float> > points;
        vector<Vec3<float> > colors;
        for(int i=0;i<center_num;i++)
        {
            points.push_back(mesh.points[center_old_index_lst[i]]);
            //colors.push_back(Vec3<float>(1,0,0));
            colors.push_back(color_plate_rgb[i]);
        }

        for(int i=0;i<center_num;i++)
        {
            points.push_back(mesh.points[center_old_index_lst[i]]);
            //colors.push_back(Vec3<float>(1,0,0));
            colors.push_back(color_plate_rgb[i]);
        }

        vector<Vec3<int> > triangles;

        for(int i=0;i<center_num;i++)
        {
            vector<int> connections = center_center_map[i];
            for(int j=1;j<connections.size();j++)
            {
                triangles.push_back(Vec3<int>(i, connections[j], i+center_num));
            }
        }

        mesh.SaveOBJ(file_dir, points, colors, triangles);

    }



};