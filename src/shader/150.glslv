#version 150 core

uniform mat4 u_model_view_proj;
uniform mat4 u_model_view;
uniform b_skinning {
    mat4 u_skinning[64];
};

in vec3 position, normal;
in vec2 uv;
in ivec4 joint_indices;
in vec4 joint_weights;


out vec2 v_TexCoord;
out vec3 _normal;

void main() {

    vec4 bindVertex = vec4(position, 1.0);
    vec4 bindNormal = vec4(normal, 0.0);
    vec4 v =  joint_weights.x * u_skinning[joint_indices.x] * bindVertex;
         v += joint_weights.y * u_skinning[joint_indices.y] * bindVertex;
         v += joint_weights.z * u_skinning[joint_indices.z] * bindVertex;
         v += joint_weights.a * u_skinning[joint_indices.a] * bindVertex;
    vec4 n = bindNormal * u_skinning[joint_indices.x] * joint_weights.x;
    n += bindNormal * u_skinning[joint_indices.y] * joint_weights.y;
    n += bindNormal * u_skinning[joint_indices.z] * joint_weights.z;
    n += bindNormal * u_skinning[joint_indices.a] * joint_weights.a;

    gl_Position = u_model_view_proj * v;
    v_TexCoord = uv;
    _normal = normalize(bindNormal).xyz;
}
