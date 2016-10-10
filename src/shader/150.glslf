#version 150 core

uniform vec3 u_light;
uniform vec4 u_ambientColor;
uniform vec3 u_eyeDirection;

in vec4 v_Color;

smooth in vec3 _normal;

out vec4 Target0;

void main() {
    float diffuse = clamp(dot(_normal, -u_light), 0.05f, 1.0f);
    vec3 halfLE = normalize(u_eyeDirection);
    float speular = pow(clamp(dot(_normal, halfLE), 0.0, 1.0), 50.0);
    Target0 = v_Color * vec4(vec3(diffuse), 1.0) + vec4(vec3(speular), 1.0) + u_ambientColor;
}
