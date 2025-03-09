#version 150 core

in vec3 pos;
in vec2 tex_coord;
out VertexData {
    vec2 tex_coord;
} v_out;

uniform mat4 view;
uniform mat4 proj;

void main() {
    gl_Position = proj * view * vec4(pos,1.0);
    v_out.tex_coord = tex_coord;
}
