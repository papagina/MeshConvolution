from .Render import *


class YiRender(Render):
    def __init__(self,
                 width, height,
                 multi_sample_rate=1
                 ):
        Render.__init__(self, width, height, multi_sample_rate, num_render_target=8)

    def _init_shader(self):
        self.shader = Shader(vs_file='yi.vs', fs_file='yi.fs', gs_file=None)

        # Declare all vertex attributes used in the program
        # layout (location = 0) in vec3 a_Position;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 1) in vec3 a_Normal;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))
        # layout (location = 3) in vec3 a_Color;
        self.vbo_list.append(VBO(type_code='f', gl_type=gl.GL_FLOAT))

        # Declare all uniforms used in the program
        self.shader.declare_uniform('ModelMat', type_code='f', gl_type=gl.glUniformMatrix4fv)
        self.shader.declare_uniform('PerspMat', type_code='f', gl_type=gl.glUniformMatrix4fv)

        self.shader.declare_uniform('SHCoeffs', type_code='f', gl_type=gl.glUniform3fv)
