#version 150 core

uniform mat4 u_model_view_proj;
uniform mat4 u_model_view;

in vec3 position;
in vec3 a_Color;
in vec3 normal;
out vec4 v_Color;

out vec3 _normal;

void main() {
    gl_Position = u_model_view_proj * vec4(position, 1.0);
    v_Color = vec4(a_Color, 1.0);
    _normal = normalize(normal);
}
