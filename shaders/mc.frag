#version 460 core

uniform vec3 lightPosition;
uniform vec3 La;
uniform vec3 Ld;
uniform vec3 Ka;
uniform vec3 Kd;
uniform uint shininess;
uniform float constantAttenuation;
uniform float linearAttenuation;
uniform float quadraticAttenuation;

in vec3 fragPosition;
in vec3 fragNormal;

out vec4 fragColor;

void main()
{
  // Ambient
  vec3 ambient = Ka * La;

  // Diffuse
  // Fragment normal has been interpolated, so it does not necessarily have norm equal to 1
  vec3 normalizedNormal = normalize(fragNormal);
  vec3 toLight = lightPosition - fragPosition;
  vec3 lightDir = normalize(toLight);
  float diff = max(dot(normalizedNormal, lightDir), 0.0);
  vec3 diffuse = Kd * Ld * diff;

  // Attenuation
  float distToLight = length(toLight);
  float attenuation = constantAttenuation +
                      linearAttenuation * distToLight +
                      quadraticAttenuation * distToLight * distToLight;
  
  vec3 greenColor = vec3(0.15f, 0.4f, 0.0f);
  vec3 result = (ambient + (diffuse / attenuation)) * greenColor;
  fragColor = vec4(result, 1.0);
}