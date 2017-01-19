#version 150 core

uniform mat4 u_model_view_proj;
uniform mat4 u_model_view;

in vec3 position, normal;
in vec2 uv;

out vec2 v_TexCoord;
out vec3 _normal;

void main() {
    v_TexCoord = vec2(uv.x, uv.y);

    gl_Position = u_model_view_proj * vec4(position, 1.0);
    _normal = normalize(normal);
}
