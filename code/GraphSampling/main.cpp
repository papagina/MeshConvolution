#include <iostream>
//#include "meshPooler.h"
#include "meshPooler_visualizer_new.h"


void set_7k_mesh_layers_dfaust(MeshCNN &meshCNN)
{
    /*
    **Attributes:
    int stride, int radius_pool, int radius_unpool, bool use_full_search_radius_for_center_to_center_map=true
    **TIPS:
    1. you can add the first layer with stride=1, pool_radius=1, unpool_radius=1 to use for computing the laplacian loss.
    2. In cases like tetrahedron meshes, to reduce edges and make the resulting graph less compact, you can set use_full_search_radius_for_center_to_center_map=false
    */
    meshCNN.add_pool_layer(1,1,1);// pool/unpool0
    meshCNN.add_pool_layer(2,2,2);// pool/unpool1
    meshCNN.add_pool_layer(1,1,1);// pool/unpool2
    meshCNN.add_pool_layer(2,2,2);// pool/unpool3
    meshCNN.add_pool_layer(1,1,1);// pool/unpool4
    meshCNN.add_pool_layer(2,2,2);// pool/unpool5
    meshCNN.add_pool_layer(1,1,1);// pool/unpool6
    meshCNN.add_pool_layer(2,2,2);// pool/unpool7
}


int main() {
    
    
    
    Mesh mesh;

    
    mesh.loadmesh_obj("../../data/DFAUST/template.obj");
    //mesh.loadmesh_obj("/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/D-FAUST/sample.obj");
    //mesh.loadmesh_obj("../../data/body_1m/template.obj");

    cout<<"#############################################################\n";
    cout<<"## Create pool and unpool layers ############################\n";
    MeshCNN meshCNN=MeshCNN(mesh);

    //TIPS, you can set layer 0 with stride=1, pool_radius=1, unpool_radius=1 to use for computing the laplacian loss.
    set_7k_mesh_layers_dfaust(meshCNN);

    cout<<"#############################################################\n";
    cout<<"## Save pool and unpool connection matrices in npy ##########\n";
    string folder="../../train/0422_graphAE_dfaust/ConnectionMatrices/";
    meshCNN.save_pool_and_unpool_neighbor_info_to_npz(folder);


    //Visualize the graph and receptive field of each down-sampling layer by vertex colors and dump in obj files.
    //Comment out the following code if you don't want to check the visualization.
    cout<<"#############################################################\n";
    cout<<"## Visualize the graphs and receptive fields in obj files. ##\n";
    MeshPooler_Visualizer mpv;

    
    for (int i=0;i<meshCNN._meshPoolers.size();i++)
    {
        mpv.save_colored_obj_receptive_field(folder+"vis_receptive_"+to_string(i)+".obj", mesh,meshCNN,i);

        mpv.save_center_mesh(folder+"vis_center_"+to_string(i)+".obj", mesh,meshCNN, i);
    } 
    return 0;



}
