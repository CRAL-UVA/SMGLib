#version 150 core

out vec4 outColor;

in VertexData {
    vec2 tex_coord;
} VertexIn;

uniform sampler2D texs[7];

uniform int agent_type;
uniform bool reactive;

void main() {
    if (agent_type == -1) {
        outColor = texture(texs[0], VertexIn.tex_coord).rgba;
    } else if (agent_type == 0) {
        outColor = texture(texs[1], VertexIn.tex_coord).rgba;
    } else if (agent_type == 1) {
        outColor = texture(texs[2], VertexIn.tex_coord).rgba;
    } else if (agent_type == 2) {
        outColor = texture(texs[3], VertexIn.tex_coord).rgba;
    } else if (agent_type == 3) {
        outColor = texture(texs[4], VertexIn.tex_coord).rgba;
    } else if (agent_type == 4) {
        outColor = texture(texs[5], VertexIn.tex_coord).rgba;
    } else {
        outColor = texture(texs[6], VertexIn.tex_coord).rgba;
    }
    if (!reactive) {
        float gray = dot(outColor.rgb, vec3(0.299, 0.587, 0.114));
        outColor.rgb = vec3(gray);
    }
}
