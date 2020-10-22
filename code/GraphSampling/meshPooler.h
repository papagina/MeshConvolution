//
// Created by zhouyi on 6/12/19.
//

#ifndef POINTSAMPLING_MESHPOOLER_H
#define POINTSAMPLING_MESHPOOLER_H

#endif //POINTSAMPLING_MESHPOOLER_H

#include "meshLoader.h"
#include <queue>
#include <list>
#include "set"
#include <math.h>


class MultiComponents{
private:
    vector<int> _cover_lst;//point_num
    vector<vector<int>> _components;
    vector<vector<int>>* _connection_map;
    const Mesh *_mesh;

    //return a vector of point index
    //stride: distance between center points
    //radius, sample the center points such that there are no other center points in the radius
    //BFS
    vector<int> get_point_lst_in_a_component(int start_id) {
        vector<int> point_lst;
        // Create a queue for BFS
        list<int> queue;


        // Mark the points in the _must_include_center_lst as center and enqueue it
        int s=start_id;
        _cover_lst[s]=1;
        queue.push_back(s);
        point_lst.push_back(s);
        
        while (!queue.empty()) {
            // Dequeue a vertex from queue and print it
            s = queue.front();
            queue.pop_front();

            vector<int> s_connection = (*_connection_map)[s];

            for (int i = 0; i < s_connection.size(); i++) {
                int p = s_connection[i];
                if (_cover_lst[p] == 1) //visited
                    continue;
                _cover_lst[p] = 1;
                queue.push_back(p);
                point_lst.push_back(p);
            }
        }

        return point_lst;
    }

    int get_first_unvisited_point()
    {
        for (int i=0;i<_cover_lst.size();i++)
        {
            if(_cover_lst[i]==0)
            {
                return i;
            }
        }
        return -1;
    }
    //this function will update _componenets
    void find_components()
    {
        int start_id = get_first_unvisited_point();
        while(start_id!=-1)
        {
            vector<int> point_lst =  get_point_lst_in_a_component(start_id);
            _components.push_back(point_lst);
            cout<<"Find component "<<_components.size()<<" with size "<<point_lst.size()<<"\n";

            start_id = get_first_unvisited_point();
        }

    }


    int find_component_id(int id, vector<set<int>> &components_of_component)
    {
        for(int i=0;i<components_of_component.size();i++)
        {
            if(components_of_component[i].find(id)!=components_of_component[i].end())
            {
                return i;
            }
        }
        return -1;
    }


    //this function will add (M-1) more edges in _connection_map
    void add_extra_edges(vector<Int2> &extra_edges)
    {
        vector<Int2> component_edges;

        for(int i=0; i<_components.size();i++)
        {
            Int2 extra_edge(-1,-1); 
            Int2 edge_between_component(-1,-1);
            float min_dist = 12345678;
            for(int j=0;j<_components[i].size();j++)
            {
                int src_point_id = _components[i][j];
                for(int k=0;k<_components.size();k++)
                {
                    if(k==i) continue;
                    for (int l=0; l<_components[k].size();l++)
                    {
                        int tar_point_id = _components[k][l];
                        Vec3<float> diff = _mesh->points[tar_point_id] - _mesh->points[src_point_id];
                        float dist = diff.GetNorm();
                        if(dist< min_dist)
                        {
                            extra_edge[0]=src_point_id;
                            extra_edge[1]=tar_point_id;
                            edge_between_component[0] = i;
                            edge_between_component[1] = k;
                            min_dist= dist;
                        }
                    }
                }
            }

            cout<<"Find closest component "<<edge_between_component[0]<<" "<<edge_between_component[1]<<"\n";

            //check if extra edge exists in extra edges if not add it
            bool edge_exist=false;
            for(int j=0;j<extra_edges.size();j++)
            {
                Int2 e = extra_edges[j];
                if(((e[0]==extra_edge[0])&&(e[1]==extra_edge[1]))||((e[0]==extra_edge[1])&&(e[1]==extra_edge[0])))
                    edge_exist=true;
            }
            if(edge_exist==false) extra_edges.push_back(extra_edge);
            
            bool component_edge_exist=false;
            for(int j=0;j<component_edges.size();j++)
            {
                Int2 e= component_edges[j];
                if(((e[0]==edge_between_component[0])&&(e[1]==edge_between_component[1]))||((e[0]==edge_between_component[1])&&(e[1]==edge_between_component[0])))
                    component_edge_exist=true;
            }
            if(component_edge_exist==false) component_edges.push_back(edge_between_component);
        }

        //merge components
        vector<set<int>> components_of_component;
        
        for(int i=0;i<component_edges.size();i++)
        {
            int c1= component_edges[i][0];
            int c2= component_edges[i][1];
            int cc_id1 = find_component_id(c1, components_of_component);
            int cc_id2 = find_component_id(c2, components_of_component);
            if ((cc_id1==-1) &&(cc_id2==-1))
            {
                set<int> components;
                components.insert(c1);
                components.insert(c2);
                components_of_component.push_back(components);
            }
            else if ((cc_id1>-1) &&(cc_id2>-1) &&(cc_id1!=cc_id2))
            {
                //merge cc_id2 to cc_id1;
                for (set<int>::iterator it = components_of_component[cc_id2].begin();it != components_of_component[cc_id2].end(); ++it)
                {
                    int c_id = *it;
                    components_of_component[cc_id1].insert(c_id);
                }
                components_of_component[cc_id2].clear();
                components_of_component.erase(components_of_component.begin()+cc_id2);
            }
            else if  ((cc_id1>-1) &&(cc_id2>-1) &&(cc_id1==cc_id2))
            {
                continue;
            }
            else if (cc_id1>-1)
                components_of_component[cc_id1].insert(c2);
            else components_of_component[cc_id2].insert(c1);
        }

        vector<vector<int>> new_components;
        for (int i=0;i<components_of_component.size();i++)
        {
            vector<int> points;
            for (set<int>::iterator it = components_of_component[i].begin();it != components_of_component[i].end(); ++it)
            {
                int c_id = *it;
                for(int j=0;j<_components[c_id].size();j++)
                    points.push_back(_components[c_id][j]);
            }
            new_components.push_back(points);
        }

        

        _components.clear();
        for(int i=0;i<new_components.size();i++)
        {
            vector<int> point_lst;
            for(int j=0;j<new_components[i].size();j++)
            {
                point_lst.push_back(new_components[i][j]);
            }
            _components.push_back(point_lst);
        }
        cout<<"After merging, there are "<<_components.size()<<" of components, ";
        int sub_components_num=0;
        for(int i=0;i<components_of_component.size();i++)
            sub_components_num+=components_of_component[i].size();
        cout<<sub_components_num<<" of sub components. ";

        int p_num=0;
        for(int i=0;i<_components.size();i++)
            p_num+=_components[i].size();
        cout<<p_num<<" points in total.\n";
        if(_components.size()>1)
        {
            add_extra_edges(extra_edges);
        }


    }
public:
    //virtual edges mean that we will add (M-1) more edges in connection map. but won't change the obj's topology. M is the number of components
    void add_virtual_edges_between_components(int point_num, vector<vector<int>>* connection_map, const Mesh* mesh)
    {
        cout<<"###Start add_virtual_edges_between_components. \n";
        for (int i = 0; i < point_num; i++) {
            _cover_lst.push_back(0);
        }
        _connection_map = connection_map;
        _mesh = mesh;
        cout<<"find_components. \n";
        find_components();

        cout<<"add_extra_edges. \n";
        vector<Int2> extra_edges;
        add_extra_edges(extra_edges);

        if(_components[0].size()!=point_num)
            cout<<"ERROR!!! _components[0].size()!=point_num "<<_components[0].size()<<" "<<point_num<<"\n";

        for(int i=0;i<extra_edges.size();i++)
        {
            int id1 = extra_edges[i][0];
            int id2 = extra_edges[i][1];
            if(id1==id2) cout<<"ERROR!!!! Extra edges contain id1=id2 edge.\n";

            (*connection_map)[id1].push_back(id2);
            (*connection_map)[id2].push_back(id1);
        }

        cout<<"add "<<extra_edges.size()<<" extra edges to connection_map.\n";

    }


};


class MeshPooler {
public:

    vector<vector<int> > _connection_map; // a list of the size of the point, per entry is a list of indices of neighboring points. symmetric. connection radius =1
    vector<int> _must_include_center_lst; //the old index of the points that must be included in center_lst
    vector<int> _must_include_center_lst_new_index; //the new index of the points that must be included in center_lst
    vector<int> _center_lst;
    vector<vector<Int2>> _pool_map ;//_pool_map[i] contains the list of old indices of connected points for new center i
                                    //map[i][j] is [index, distance]
    vector<vector<Int2>> _unpool_map;
    vector<vector<int>> _center_center_map;
    vector<int> _old2new_index_lst;

    bool _use_full_search_radius_for_center_to_center_map;

    MeshPooler(bool use_full_search_radius_for_center_to_center_map=true)
    {
        _use_full_search_radius_for_center_to_center_map=use_full_search_radius_for_center_to_center_map;
    }


    void set_new_edge_to_connection_map(int p1, int p2, vector<vector<int> > &connection_map) {
        vector<int> p1_connection = connection_map[p1];
        bool exist = false;
        for (int i = 0; i < p1_connection.size(); i++) {
            if (p1_connection[i] == p2) {
                exist = true;
                break;
            }
        }
        if (exist == false) {
            connection_map[p1].push_back(p2);
            connection_map[p2].push_back(p1);
        }
    }

    void set_connection_map_from_mesh(const Mesh & mesh) {
        cout<<"Start setting connection map from mesh\n";
        _connection_map.clear();
        int point_num = mesh.points.size();
        _connection_map.resize(point_num);

        for (int i = 0; i < mesh.triangles.size(); i++) {
            Vec3<int> p = mesh.triangles[i];
            set_new_edge_to_connection_map(p[0], p[1], _connection_map);
            set_new_edge_to_connection_map(p[1], p[2], _connection_map);
            set_new_edge_to_connection_map(p[2], p[0], _connection_map);
        }

        //MultiComponents mc;
        //mc.add_virtual_edges_between_components(point_num, &_connection_map, &mesh);
    }

    void set_must_include_center_lst_from_mesh(const Mesh & mesh){
        cout<<"Start setting must_include_center_lst from mesh (the red points)\n";
        _must_include_center_lst.clear();

        int point_num = mesh.colors.size();

        for(int i =0; i< point_num; i++)
        {
            Vec3<float> c = mesh.colors[i];
            if((c[0]==1) && (c[1]==0) && (c[2]==0))
            {
                _must_include_center_lst.push_back(i);
            }

        }

        cout <<"There are "<<_must_include_center_lst.size()<<" red points.\n";
    }

    bool is_connection_map_good()
    {
        int p_num = _connection_map.size();
        if(p_num<3)
        {
            cout<<"ERROR: connection map receives less than 3 points.\n";
            return false;
        }

        //check symmetry
        for (int i=0;i<p_num;i++)
        {
            int p=i;
            vector<int> p_connection_lst = _connection_map[i];
            for(int j =0; j<p_connection_lst.size();j++)
            {
                int q = p_connection_lst[j];
                vector<int> q_connection_lst = _connection_map[q];
                bool find_p = false;
                for(int k=0;k<q_connection_lst.size();k++)
                {
                    if(q_connection_lst[k]==p)
                    {
                        find_p =true;
                        break;
                    }
                }
                if(find_p==false) {
                    cout<<"ERROR: the connection map is not symmetric.\n";
                    return false;
                }
            }
        }

        return true;
    }

    //after connection_map is set up
    //I recommend to use stride=2, radius_pool >= stride-1, radius_unpool = n*stride-1, n=1,2,...
    //unpool radius is also respected to the original connection map
    void compute_pool_and_unpool_map(int stride=2, int radius_pool=1, int radius_unpool=1)
    {
        //first check connection map is ok
        if(is_connection_map_good()==false)
            return;

        cout<<"stride: "<<stride<<", radius pool: "<<radius_pool<<", radius unpool: "<<radius_unpool;
        //cout<<"Get center points lst.\n";
        //step 1 get sample point list with stride S (for both pool and unpool)
        //sample the centers so that no other centers in radius (S-1) of each center
        //center_points_lst contains the old indices of the center points

        
        _center_lst = get_center_points_lst(stride);



        cout<<"Center points number: "<<_center_lst.size()<<"\n";
        //also get old to new list
        int original_num=_connection_map.size();

        for(int i=0;i<original_num;i++)
            _old2new_index_lst.push_back(-1);
        for(int i=0;i<_center_lst.size();i++)
            _old2new_index_lst[_center_lst[i]]=i;



        //cout<<"Get pool map.\n";
        //step 2 get pc to center map (center*pc) with radius radius_pool for pooling
        //length center num
        //map[i] contains the list of old indices of connected points for new center i
        //map[i][j] is [index, distance]
        _pool_map = get_center_to_pc_map( _center_lst, radius_pool);


        //cout<<"Get unpool map.\n";
        //step 3 get center to pc map (pc*center) with radius radius_unpool for unpooling
        //length center num
        //map[i] contains the list of old indices of connected points for new center i
        //map[i][j] is [index, distance]

        vector<vector<Int2>> unpool_map_T = get_center_to_pc_map( _center_lst, radius_unpool);
        _unpool_map = get_transposed_map (unpool_map_T, original_num);



        //cout<<"Get center to center map.\n";
        //step 4 get new (center to center) connection map
        //radius = 1 among centers.
        //length center num
        //map[i] contains the list of new indices of connected centers for center i
        _center_center_map = get_center_to_center_map(_center_lst, stride);

        
        //update _must_include_center_lst_nex_index
        _must_include_center_lst_new_index.clear();
        for (int i=0;i<_must_include_center_lst.size();i++)
        {
            int old_index= _must_include_center_lst[i];
            int new_index = _old2new_index_lst[old_index];
            if(new_index == -1)
            {
                cout<<"Error while updating _must_include_center_lst_new_index!!!! The index is -1 in _old2new_index_lst. Very likely they are not included in the center_lst!\n";
                return;
            }
            else
            {
                _must_include_center_lst_new_index.push_back(new_index);
            }
            
        }

        return;

    }

private:


    //***********Begin of getting center points list**************//

    //check if there are center points in the radius of p. if so return false, else return true
    bool can_be_center(int p, int radius, const vector<vector<int>> &connection_map, const vector<int> &cover_lst) {
        vector<int> interest_lst;
        interest_lst.push_back(p);

        set<int> visited;
        visited.insert(p);

        int r = 0;
        while (r < radius) {
            vector<int> new_interest_lst;
            //cout<<interest_lst.size()<<"\n";
            for (int i = 0; i < interest_lst.size(); i++) {
                int interest_p = interest_lst[i];
                vector<int> interest_p_connection_map = connection_map[interest_p];
                for (int j = 0; j < interest_p_connection_map.size(); j++) {
                    int neighbor_p = interest_p_connection_map[j];


                    if (cover_lst[neighbor_p] == 1) {

                        return false;
                    }

                    set<int>::iterator it = visited.find(neighbor_p);
                    if (it == visited.end()) { //not visited
                        //cout<<"neighbor ";

                        visited.insert(neighbor_p);
                        new_interest_lst.push_back(neighbor_p);
                    }
                }
            }
            interest_lst = new_interest_lst;
            r = r + 1;
        }
        return true;
    }

    //return a vector of point index
    //stride: distance between center points
    //radius, sample the center points such that there are no other center points in the radius
    //BFS
    vector<int> get_center_points_lst(int stride) {
        int radius = stride -1;
        // Mark all the vertices as not visited
        vector<int> cover_lst; //-1 not visited, 0 covered, 1 center
        for (int i = 0; i < _connection_map.size(); i++) {
            cover_lst.push_back(-1);
        }

        // Create a queue for BFS
        list<int> queue;

        // Mark the points in the _must_include_center_lst as center and enqueue it
        int s;
        
        for(int i=0; i<_must_include_center_lst.size();i++)
        {
            s = _must_include_center_lst[i];
            cover_lst[s] = 1;
            queue.push_back(s);
        }
        
        if(_must_include_center_lst.size()==0)
        {
            s=0;
            queue.push_back(s);
        }
        while (!queue.empty()) {
            // Dequeue a vertex from queue and print it
            s = queue.front();
            queue.pop_front();

            vector<int> s_connection = _connection_map[s];

            for (int i = 0; i < s_connection.size(); i++) {
                int p = s_connection[i];
                if (cover_lst[p] >= 0) //visited
                    continue;

                if (can_be_center(p, radius, _connection_map, cover_lst)) {
                    cover_lst[p] = 1;
                } else {
                    cover_lst[p] = 0;
                }
                queue.push_back(p);

            }

        }
        


        vector<int> sample_points_lst;
        for (int i = 0; i < cover_lst.size(); i++) {
            if (cover_lst[i] == 1)
                sample_points_lst.push_back(i);
        }

        return sample_points_lst;

    }

    //***********End of getting center points list**************//


    //***********Begin of getting center to pc map**************//

    vector<Int2> get_p_connection_lst(int p, int radius)
    {
        vector<int> interest_lst;
        interest_lst.push_back(p);

        set<int> visited;
        visited.insert(p);

        vector<Int2> p_connection_lst;
        Int2 connection=Int2(p,0);
        p_connection_lst.push_back(connection);

        int r = 0;
        while (r < radius) {
            vector<int> new_interest_lst;
            //cout<<interest_lst.size()<<"\n";
            for (int i = 0; i < interest_lst.size(); i++) {
                int interest_p = interest_lst[i];
                vector<int> interest_p_connection_map = _connection_map[interest_p];
                for (int j = 0; j < interest_p_connection_map.size(); j++) {
                    int neighbor_p = interest_p_connection_map[j];

                    set<int>::iterator it = visited.find(neighbor_p);
                    if (it == visited.end()) { //not visited
                        //cout<<"neighbor ";

                        visited.insert(neighbor_p);
                        new_interest_lst.push_back(neighbor_p);
                        Int2 connection=Int2(neighbor_p,r+1);
                        p_connection_lst.push_back(connection);

                    }
                }
            }
            interest_lst = new_interest_lst;
            r=r+1;
        }




        return p_connection_lst;

    }




    vector<vector<Int2>> get_center_to_pc_map(vector<int> center_lst, int radius) {

        vector<vector<Int2>> center_to_pc_map;
        for (int i=0;i<center_lst.size();i++)
        {
            int p = center_lst[i]; //old index of center p
            vector<Int2> p_connection_lst = get_p_connection_lst(p,radius);
            center_to_pc_map.push_back(p_connection_lst);
        }

        return center_to_pc_map;

    }

    //***********End of getting center to pc map**************//

    //***********Start of getting center to center map**************//


    vector<vector<int>> get_center_to_center_map(vector<int> center_lst, int stride)
    {
        int original_num=_connection_map.size();
        vector<int> is_center_lst, old2new_index_lst;
        for(int i=0;i<original_num;i++)
        {
            is_center_lst.push_back(0);
            old2new_index_lst.push_back(-1);
        }
        for(int i=0;i<center_lst.size();i++)
        {
            int old_index= center_lst[i];
            is_center_lst[old_index]=1;
            old2new_index_lst[old_index]=i;
        }


        vector<vector<int>> center_to_center_map;
        for (int i=0;i<center_lst.size();i++)
        {
            int p = center_lst[i]; //old index of center p
            int search_radius = stride;
            if(_use_full_search_radius_for_center_to_center_map == true)
            {
                search_radius = 2*stride-1;
            }
            vector<Int2> p_connection_lst_old_index = get_p_connection_lst(p,search_radius);
            /*
            int s = stride+1;
            while(s<=stride*2)
            {
                if(p_connection_lst_old_index.size()>2)
                    break;
                p_connection_lst_old_index = get_p_connection_lst(p,s);
                s+=1;
            }
            */
            vector<int> center_connection_lst;
            //cout<<p_connection_lst_old_index.size()<<" ";
            for(int j=0;j<p_connection_lst_old_index.size();j++)
            {
                //cout<<j<<"\n";

                Int2 qq = p_connection_lst_old_index[j];
                int q=qq[0];
                //cout<<qq[0]<<" "<<qq[1]<<"\n";
                if(is_center_lst[q]!=1)
                    continue;
                else
                    center_connection_lst.push_back(old2new_index_lst[q]);
            }
            //cout<<center_connection_lst.size()<<"\n";
            center_to_center_map.push_back(center_connection_lst);
        }

        return center_to_center_map;


    }


    vector<vector<Int2>> get_transposed_map(vector<vector<Int2>> map, int previous_point_num)
    {
        int after_point_num = map.size();

        vector<vector<Int2>> map_T = vector<vector<Int2>>(previous_point_num);

        for(int i=0;i< after_point_num; i++)
        {
            int center_id = i;
            for(int j=0;j<map[i].size(); j++)
            {
                int target_id = map[i][j][0];
                int dist = map[i][j][1];
                map_T[target_id].push_back(Int2(center_id, dist));

            }
        }
        return map_T;
    }

    //***********End of getting center to center map**************//







};


















