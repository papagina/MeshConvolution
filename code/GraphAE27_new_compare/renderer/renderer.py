"""""""""""""""
Requirement:
right hand coordinate in meters.
A square area for my roughly 1.7 meters' high agent to move, with size around 8m*8m, centered at the origin. The floor (y=0) needs to be rendered. 
The whole area should be bright (might need multiple lights to light up the whole area)
Have alpha channel.
"""""""""""""""
from .render.gl.YiRender import YiRender
from .render.PespectiveCamera import PersPectiveCamera
from .render.CameraPose import CameraPose
import os
import numpy as np


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_faces_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    #norm = np.zeros((faces.shape[0],3,3), dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces] #faces_num, 3, 3
    #print ("tris",tris.shape)
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) #faces_num,3
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    #norm[faces[:, 0]] += n
    #norm[faces[:, 1]] += n
    #norm[faces[:, 2]] += n
    #normalize_v3(norm)
    
    n = n.reshape((-1,1,3)).repeat(3,1)
    
    #print (n.shape)
    return n


class Mesh:
    # facets numpy(or torch) facet_num*3

    # vertices numpy(or torch) vertice_num*3
    # faces    numpy(or torch) facet_num*3
    # color (R,G,B,A) the color for the whole mesh
    def __init__(self, vertices, faces, face_normals, color):

        self.vertices = vertices
        self.faces = faces
        self.faces_normals = compute_faces_normal(vertices, faces) #face_num*3*3
        self.colors = np.ones((faces.shape[0], 3))
        self.colors[:,0]=self.colors[:,0]*color[0]
        self.colors[:,1]=self.colors[:,1]*color[1]
        self.colors[:,2]=self.colors[:,2]*color[2]
        
        
        


# The sphere class
class Renderer:

    # facets numpy(or torch) facet_num*3
    def __init__(self, width=512, height=512, anti_alias=8):
        self.render = YiRender(width=width, height=height, multi_sample_rate=anti_alias)
        source_path = os.path.dirname(os.path.abspath(__file__))

        sh_file = os.path.join(source_path, 'render', 'gl', 'data', 'env_sh_norm.npy')
        self.shs = np.load(sh_file)
        # mesh_lst [Mesh] #the facet is indiced independently for each mesh

    # camera_pos 3 [x,y,z]
    # camera_direction 3 [x,y,z]
    # camera_fov 1
    # mesh_lst mesh
    # img_size 2 (width,height)
    # return an image.
    def render_one_frame(self, mesh, camera_pos, camera_direction, camera_fov, sh_id=3):
        
        vertices = mesh.vertices
        faces = mesh.faces
        color = mesh.colors
        faces_normals = mesh.faces_normals
        vert_data_total = vertices[faces.reshape([-1])]
        norm_data_total = faces_normals.reshape((-1,3))
        color_data_total = np.concatenate([color, color, color], axis=1).reshape([-1, 3])
            
        
        self.render.set_attrib(0, vert_data_total)
        self.render.set_attrib(1, norm_data_total)
        self.render.set_attrib(2, color_data_total)

        camera = PersPectiveCamera()
        camera.far =20
        camera.set_by_field_of_view(camera_fov)

        pose = CameraPose()
        pose.center = camera_pos
        pose.front = -camera_direction
        pose.sanity_check()
        
        #print(len(self.shs))

        uniform_dict = {
            'ModelMat': pose.get_model_view_mat(),
            'PerspMat': camera.get_projection_mat(),
            'SHCoeffs': self.shs[sh_id % len(self.shs)]
        }

        self.render.draw(
            uniform_dict
        )

        return self.render.get_color()

    
    

def get_floor_mesh(length, grid_num):
    vertices=[]
    l = length / (grid_num-1)
    for i in range(grid_num-1):
        
        for j in range(grid_num-1):
            if((j+i)%2==1):
                continue
            x = -length/2 + i*l 
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + i*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + i*l
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
    vertices=np.array(vertices) 
    faces = np.array(range(len(vertices))).reshape((-1,3))
    normals = np.zeros(vertices.shape)
    normals[:,1]=normals[:,1]+1
    mesh = Mesh(vertices, faces, normals, np.array([0.2,0.2,0.18]))
    mesh_lst =[mesh]
    
    vertices=[]
    l = length / (grid_num-1)
    for i in range(grid_num-1):
        
        for j in range(grid_num-1):
            if((j+i)%2==0):
                continue
            x = -length/2 + i*l
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + i*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + i*l
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + (j+1)*l
            vertices+=[[x, 0, z]]
            
            x = -length/2 + (i+1)*l
            z = -length/2 + j*l
            vertices+=[[x, 0, z]]
            
    vertices=np.array(vertices)
    faces = np.array(range(len(vertices))).reshape((-1,3))
    normals = np.zeros(vertices.shape)
    normals[:,1]=normals[:,1]+1
    mesh = Mesh(vertices, faces, normals, np.array([0.1,0.1,0.09]))
    mesh_lst +=[mesh]
    return mesh_lst

