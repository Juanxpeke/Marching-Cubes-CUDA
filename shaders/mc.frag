#version 460 core

uniform vec3 lightPosition; 
uniform vec3 viewPosition;
uniform vec3 La;
uniform vec3 Ld;
uniform vec3 Ls;
uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;
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

  // Specular
  vec3 viewDir = normalize(viewPosition - fragPosition);
  vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
  vec3 specular = Ks * Ls * spec;

  // Attenuation
  float distToLight = length(toLight);
  float attenuation = constantAttenuation +
                      linearAttenuation * distToLight +
                      quadraticAttenuation * distToLight * distToLight;
    
  vec3 result = (ambient + ((diffuse + specular) / attenuation)) * vec3(1.0f, 1.0f, 1.0f);
  fragColor = vec4(result, 1.0);
}