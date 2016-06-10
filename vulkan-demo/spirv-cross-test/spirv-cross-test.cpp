#include <stdlib.h>
#include <vector>
#include <random>
#include <chrono>

#include <vulkan/vulkan.h>
#include "vk_cpp.hpp"

#include "SPIRV/GlslangToSpv.h"

#include "SPIRV-Cross/spirv_glsl.hpp"

#define DEBUG _DEBUG

/*****************************************************************************
* DEBUG
*****************************************************************************/

// Only execute actions of D_ in DEBUG mode
// D_ for single line statements
#if DEBUG
#define D_(x) do { x; } while(0)
#else
#define D_(x) do {    } while(0)
#endif

/*****************************************************************************
* SHADERS
******************************************************************************/

std::string simpleVertShaderText =
R"vertexShader(
#version 440
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 col;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec2 tex;
layout(location = 1) out vec4 Col;
layout(location = 3) out vec2 Tex;
layout(binding = 0) uniform UBO
{
	mat4 proj;
	mat4 view;
	mat4 model;
} ubo;
void main() {
	vec3 lightdir = normalize(vec3(-.3, -.4, -.6));
	Col = col * clamp(dot((ubo.model * vec4(norm, 1.0)).xyz, -lightdir), 0.0, 1.0);
	Tex = tex;
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos, 1.0);
}
)vertexShader";

std::string simpleFragShaderText =
R"fragmentShader(
#version 440
layout(location = 1) in vec4 col;
layout(location = 3) in vec2 texcoord;
layout(location = 0) out vec4 out_Color;
layout(binding = 1) uniform sampler2D tex;
layout(binding = 2) uniform sampler2D tex2;
void main() {
  out_Color =  texture(tex, texcoord) * texture(tex2, texcoord) * col;
}
)fragmentShader";

std::string lodFragShaderText =
R"fragmentShader(
#version 440
layout(location = 1) in vec4 col;
layout(location = 3) in vec2 texcoord;
layout(location = 0) out vec4 out_Color;
layout(binding = 1) uniform sampler2D tex;
void main() {
  out_Color =  texture(tex, texcoord);
}
)fragmentShader";

std::string couchVertShaderText =
R"vertexShader(
#version 440

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 tang;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec2 tex;

layout(location = 1) out vec3 Tan;
layout(location = 2) out vec3 Norm;
layout(location = 3) out vec2 Tex;

layout(binding = 0) uniform UBO
{
	mat4 proj;
	mat4 view;
	mat4 model;
} ubo;

void main() {
	Tan = (ubo.model * vec4(tang.xyz, 0.0)).xyz;
	Norm = (ubo.model * vec4(norm, 0.0)).xyz;
	Tex = tex;

	vec3 pos2 = pos + gl_InstanceIndex * vec3(0.0, 0.0, 100.0) - vec3(0.0, 0.0, 50.0);
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(pos2, 1.0);
}
)vertexShader";

std::string couchFragShaderText =
R"fragmentShader(
#version 440

layout(location = 1) in vec3 tang;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec2 texcoord;

layout(location = 0) out vec4 out_Color;

layout(binding = 1) uniform sampler2D maskMap;
layout(binding = 2) uniform sampler2D leatherNormalMap;
layout(binding = 3) uniform sampler2D baseNormalMap;
layout(binding = 4) uniform sampler2D aoMap;
layout(binding = 5) uniform sampler2D leatherSpecularMap;
layout(binding = 6) uniform sampler2D leatherMap;

layout(std430, push_constant) uniform PushConstants
{
	vec3 lightColor;
	vec3 cameraPos;
} constants;

float ComputeLuminance(vec3 color)
{
    return color.x * 0.3 + color.y * 0.59 + color.z * 0.11;
}

vec3 desaturate(vec3 color, float factor)
{
    float lum = ComputeLuminance(color);
    return mix(color, vec3(lum, lum, lum), factor);
}

vec3 TangentSpaceTransform(vec3 normal_in) {
	vec3 vNormal = norm;
	vec3 vTangent = tang;
	vec3 vBiTangent = cross(vNormal, vTangent);
	return normalize(normal_in.x * vTangent + normal_in.y * vBiTangent + normal_in.z * vNormal);
}

float Pow4(float x)
{
    return (x*x)*(x*x);
}

vec2 LightingFuncGGX_FV(float dotLH, float roughness)
{
    float alpha = roughness*roughness;/*sf*/

    // F
    float F_a, F_b;
    float dotLH5 = Pow4(1.0-dotLH) * (1.0 - dotLH);
    F_a = 1.0;
    F_b = dotLH5;

    // V
    float vis;
    float k = alpha/2.0;
    float k2 = k*k;
    float invK2 = 1.0-k2;
    vis = 1.0/(dotLH*dotLH*invK2 + k2);

    return vec2(F_a*vis, F_b*vis);
}

float LightingFuncGGX_D(float dotNH, float roughness)
{
    float alpha = roughness*roughness;
    float alphaSqr = alpha*alpha;
    float pi = 3.14159;
    float denom = dotNH * dotNH *(alphaSqr-1.0) + 1.0;

    float D = alphaSqr/(pi * denom * denom);
    return D;
}

vec3 Lighting(vec3 normal, vec3 albedo, vec3 lightParam) {
	vec3 pos = gl_FragCoord.xyz;
	vec3 lightDir = vec3(-10.0, 10.0, 1.0);
	vec3 lightColor = constants.lightColor;
	vec3 cameraPos = constants.cameraPos;

	float brightness = clamp(dot(lightDir, normal), 0.0, 1.0);
	vec3 view = normalize(cameraPos - pos);
	
	float roughness_in = lightParam.x;
	float metallic_in = lightParam.y;
	float specular_in = lightParam.z;

	vec3 L = lightDir;
	vec3 H = normalize(view + L);
	
	float dotNL = clamp(dot(normal, L), 0.01, 0.99);
	float dotLH = clamp(dot(L, H), 0.01, 0.99);
	float dotNH = clamp(dot(normal, H), 0.01, 0.99);

	float D = LightingFuncGGX_D(dotNH,roughness_in);
    vec2 FV_helper = LightingFuncGGX_FV(dotLH,roughness_in);
    float FV = metallic_in*FV_helper.x + (1.0-metallic_in)*FV_helper.y;
    float specular = dotNL * D * FV * specular_in;
    float highlight = specular;

	return lightColor * 
                        (albedo * (brightness + 0.7)*(1.0-metallic_in) + 
                        mix(albedo, vec3(1.0), 1.0 - metallic_in) * (highlight ));
}

void main() {
	vec2 normalCoord = texcoord * 5.79;
	vec3 mask = texture(maskMap, texcoord).xyz;

	vec2 macroNormalCoord = texcoord * 0.372;
	vec3 macroNormal = (texture(leatherNormalMap, macroNormalCoord).xyz * 2.0 - vec3(1.0, 1.0, 1.0)) * vec3(0.274, 0.274, 0.0);
	vec3 leatherNormal = (texture(leatherNormalMap, normalCoord).xyz * 2.0 - vec3(1.0, 1.0, 1.0)) * vec3(1.0, 1.0, 0.0);
	vec3 normal = normalize(texture(baseNormalMap, texcoord).xyz * 2.0 - vec3(1.0, 1.0, 1.0) + (leatherNormal + macroNormal) * mask.x);

	vec3 aoTex = texture(aoMap, texcoord).xyz;
	vec3 specTex = texture(leatherSpecularMap, normalCoord).xyz;
	float wearFactor = mask.z * 0.381;

	float Roughness = mix(mix(mix(0.2, mix(mix(0.659, 2.01, specTex.x),
								-0.154, wearFactor), mask.x), 0.0, mask.y), 0.0, aoTex.y);
	float Metallic = mix(0.5, 0.1, specTex.x);
	float Specular = 1.0;

	float ao = aoTex.x;
	vec3 Color1 = vec3(0.0, 0.0, 0.0);
	float Desaturation2 = 0.0;
	float Desaturation2WearSpot = 0.0896;
	vec3 Color2 = vec3(1.0, 0.86, 0.833);
	vec3 Color2WearSpot = vec3(0.628, 0.584, 0.584);
	vec3 Color3 = vec3(0.823, 0.823, 0.823);
	vec3 SeamColor = vec3(0.522, 0.270, 0.105);
	vec3 albedo = mix(mix(mix(Color1,desaturate(texture(leatherMap, normalCoord).xyz,
            mix(Desaturation2, Desaturation2WearSpot, wearFactor)) * 
            mix(Color2, Color2WearSpot, wearFactor), mask.x), 
            Color3, mask.y), SeamColor, aoTex.y) * ao;

    vec3 normalTransform = TangentSpaceTransform(normal);
	vec3 lighting = Lighting(normalTransform, albedo, vec3(Roughness, Metallic, Specular));
    
	vec3 normalVis = (normalTransform  + vec3(1.0)) / 2.0;
	out_Color = vec4(lighting, 1.0);
}
)fragmentShader";

EShLanguage find_language(const vk::ShaderStageFlagBits shader_type) {
	switch (shader_type) {
	case vk::ShaderStageFlagBits::eVertex: return EShLangVertex;
	case vk::ShaderStageFlagBits::eTessellationControl: return EShLangTessControl;
	case vk::ShaderStageFlagBits::eTessellationEvaluation: return EShLangTessEvaluation;
	case vk::ShaderStageFlagBits::eGeometry: return EShLangGeometry;
	case vk::ShaderStageFlagBits::eFragment: return EShLangFragment;
	case vk::ShaderStageFlagBits::eCompute: return EShLangCompute;
	default: return EShLangVertex;
	}
}

void init_resources(TBuiltInResource &Resources) {
	Resources.maxLights = 32;
	Resources.maxClipPlanes = 6;
	Resources.maxTextureUnits = 32;
	Resources.maxTextureCoords = 32;
	Resources.maxVertexAttribs = 64;
	Resources.maxVertexUniformComponents = 4096;
	Resources.maxVaryingFloats = 64;
	Resources.maxVertexTextureImageUnits = 32;
	Resources.maxCombinedTextureImageUnits = 80;
	Resources.maxTextureImageUnits = 32;
	Resources.maxFragmentUniformComponents = 4096;
	Resources.maxDrawBuffers = 32;
	Resources.maxVertexUniformVectors = 128;
	Resources.maxVaryingVectors = 8;
	Resources.maxFragmentUniformVectors = 16;
	Resources.maxVertexOutputVectors = 16;
	Resources.maxFragmentInputVectors = 15;
	Resources.minProgramTexelOffset = -8;
	Resources.maxProgramTexelOffset = 7;
	Resources.maxClipDistances = 8;
	Resources.maxComputeWorkGroupCountX = 65535;
	Resources.maxComputeWorkGroupCountY = 65535;
	Resources.maxComputeWorkGroupCountZ = 65535;
	Resources.maxComputeWorkGroupSizeX = 1024;
	Resources.maxComputeWorkGroupSizeY = 1024;
	Resources.maxComputeWorkGroupSizeZ = 64;
	Resources.maxComputeUniformComponents = 1024;
	Resources.maxComputeTextureImageUnits = 16;
	Resources.maxComputeImageUniforms = 8;
	Resources.maxComputeAtomicCounters = 8;
	Resources.maxComputeAtomicCounterBuffers = 1;
	Resources.maxVaryingComponents = 60;
	Resources.maxVertexOutputComponents = 64;
	Resources.maxGeometryInputComponents = 64;
	Resources.maxGeometryOutputComponents = 128;
	Resources.maxFragmentInputComponents = 128;
	Resources.maxImageUnits = 8;
	Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
	Resources.maxCombinedShaderOutputResources = 8;
	Resources.maxImageSamples = 0;
	Resources.maxVertexImageUniforms = 0;
	Resources.maxTessControlImageUniforms = 0;
	Resources.maxTessEvaluationImageUniforms = 0;
	Resources.maxGeometryImageUniforms = 0;
	Resources.maxFragmentImageUniforms = 8;
	Resources.maxCombinedImageUniforms = 8;
	Resources.maxGeometryTextureImageUnits = 16;
	Resources.maxGeometryOutputVertices = 256;
	Resources.maxGeometryTotalOutputComponents = 1024;
	Resources.maxGeometryUniformComponents = 1024;
	Resources.maxGeometryVaryingComponents = 64;
	Resources.maxTessControlInputComponents = 128;
	Resources.maxTessControlOutputComponents = 128;
	Resources.maxTessControlTextureImageUnits = 16;
	Resources.maxTessControlUniformComponents = 1024;
	Resources.maxTessControlTotalOutputComponents = 4096;
	Resources.maxTessEvaluationInputComponents = 128;
	Resources.maxTessEvaluationOutputComponents = 128;
	Resources.maxTessEvaluationTextureImageUnits = 16;
	Resources.maxTessEvaluationUniformComponents = 1024;
	Resources.maxTessPatchComponents = 120;
	Resources.maxPatchVertices = 32;
	Resources.maxTessGenLevel = 64;
	Resources.maxViewports = 16;
	Resources.maxVertexAtomicCounters = 0;
	Resources.maxTessControlAtomicCounters = 0;
	Resources.maxTessEvaluationAtomicCounters = 0;
	Resources.maxGeometryAtomicCounters = 0;
	Resources.maxFragmentAtomicCounters = 8;
	Resources.maxCombinedAtomicCounters = 8;
	Resources.maxAtomicCounterBindings = 1;
	Resources.maxVertexAtomicCounterBuffers = 0;
	Resources.maxTessControlAtomicCounterBuffers = 0;
	Resources.maxTessEvaluationAtomicCounterBuffers = 0;
	Resources.maxGeometryAtomicCounterBuffers = 0;
	Resources.maxFragmentAtomicCounterBuffers = 1;
	Resources.maxCombinedAtomicCounterBuffers = 1;
	Resources.maxAtomicCounterBufferSize = 16384;
	Resources.maxTransformFeedbackBuffers = 4;
	Resources.maxTransformFeedbackInterleavedComponents = 64;
	Resources.maxCullDistances = 8;
	Resources.maxCombinedClipAndCullDistances = 8;
	Resources.maxSamples = 4;
	Resources.limits.nonInductiveForLoops = 1;
	Resources.limits.whileLoops = 1;
	Resources.limits.doWhileLoops = 1;
	Resources.limits.generalUniformIndexing = 1;
	Resources.limits.generalAttributeMatrixVectorIndexing = 1;
	Resources.limits.generalVaryingIndexing = 1;
	Resources.limits.generalSamplerIndexing = 1;
	Resources.limits.generalVariableIndexing = 1;
	Resources.limits.generalConstantMatrixVectorIndexing = 1;
}

// GLSL to SPIR-V compiler modeled after the VKCPP wrappers
template <typename Allocator = std::allocator<unsigned int>>
typename std::vector<unsigned int, Allocator> GLSLtoSPV(const vk::ShaderStageFlagBits shader_type, const std::string& pshader) {
	std::vector<unsigned int> shaderSPV;

	glslang::InitializeProcess();

	EShLanguage stage = find_language(shader_type);
	glslang::TShader  shader(stage);
	glslang::TProgram program;

	const char *shaderStrings[1];

	// Get resource limits
	TBuiltInResource resources;
	init_resources(resources);

	// Enable SPIR-V and Vulkan rules
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

	// Set up shader
	shaderStrings[0] = pshader.c_str();
	shader.setStrings(shaderStrings, 1);

	// Parse shader
	if (!shader.parse(&resources, 100, false, messages)) {
		std::string err_message = (std::string)shader.getInfoLog() + "\n\n" + (std::string)shader.getInfoDebugLog();
		throw std::runtime_error(err_message);
	}

	program.addShader(&shader);

	// Process program
	if (!program.link(messages)) {
		std::string err_message = (std::string)shader.getInfoLog() + "\n\n" + (std::string)shader.getInfoDebugLog(); // should this be program?
		throw std::runtime_error(err_message);
	}

	glslang::GlslangToSpv(*program.getIntermediate(stage), shaderSPV);

	glslang::FinalizeProcess();

	return shaderSPV;
}

#include <stdarg.h>
void print_indent(size_t indent_level, const char* format, ...) {
	va_list vl;
	va_start(vl, format);

	printf("%s", std::string(2*indent_level, ' ').c_str());
	vprintf(format, vl);

	va_end(vl);
}

/*****************************************************************************
* MAIN FUNCTION
*****************************************************************************/

int main() {
	std::string vertShaderText = simpleVertShaderText;
	std::string fragShaderText = simpleFragShaderText;

	std::vector<unsigned int> vertShaderSPV;
	std::vector<unsigned int> fragShaderSPV;

	try {
		vertShaderSPV = GLSLtoSPV(vk::ShaderStageFlagBits::eVertex, vertShaderText);
		fragShaderSPV = GLSLtoSPV(vk::ShaderStageFlagBits::eFragment, fragShaderText);
	}
	catch (std::runtime_error e) {
		fprintf(stderr, "glslang: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	spirv_cross::CompilerGLSL vertShaderGLSL(std::move(vertShaderSPV));
	spirv_cross::CompilerGLSL fragShaderGLSL(std::move(fragShaderSPV));

	spirv_cross::ShaderResources vertShaderResources = vertShaderGLSL.get_shader_resources();
	spirv_cross::ShaderResources fragShaderResources = fragShaderGLSL.get_shader_resources();

	print_indent(0, "Vertex Shader:\n");
	print_indent(1, "Inputs:\n");

	std::vector<std::string> vtxInputs;
	for (auto &resource : vertShaderResources.stage_inputs) {
		uint32_t binding = vertShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationBinding);
		uint32_t location = vertShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationLocation);
		uint32_t offset = vertShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationOffset);

		spirv_cross::SPIRType type = vertShaderGLSL.get_type(resource.type_id);

		print_indent(2, "Resource ID: %u\n", resource.id);
		print_indent(3, "Name: %s\n", vertShaderGLSL.get_name(resource.id).c_str());
		print_indent(3, "Fallback Name: %s\n", vertShaderGLSL.get_fallback_name(resource.id).c_str());
		print_indent(3, "Type Name: %s\n", vertShaderGLSL.get_name(resource.type_id).c_str());
		print_indent(3, "Type Fallback Name: %s\n", vertShaderGLSL.get_fallback_name(resource.type_id).c_str());
		print_indent(3, "Type ID: %u\n", resource.type_id);
		print_indent(3, "Type: %d\n", type.basetype);
		print_indent(3, "Type width: %d\n", type.width);
		print_indent(3, "Type vecsize: %d\n", type.vecsize);
		print_indent(3, "Type columns: %d\n", type.columns);
		print_indent(3, "Binding: %u\n", binding);
		print_indent(3, "Location: %u\n", location);
		print_indent(3, "Offset: %u\n", offset);

		std::string curIn = "layout(location = " + std::to_string(location) + ") in ";
		if (type.basetype == spirv_cross::SPIRType::Float) {
			if (type.width == 32) {
				if (type.vecsize > 1) {
					curIn += "vec" + std::to_string(type.vecsize) + " ";
					if (type.vecsize == 4) {
						print_indent(4, "%s\n", "vk::Format::eR32G32B32A32Sfloat");
					}
				}
				else {
					curIn += "float ";
				}

				curIn += resource.name + ";";
			}
		}

		vtxInputs.push_back(curIn);
	}

	for (int k = 0; k < vtxInputs.size(); k++) {
		printf("%s\n", vtxInputs.at(k).c_str());
	}

	printf("\n");

	print_indent(0, "Fragment Shader:\n");
	print_indent(1, "Outputs\n");
	for (auto &resource : fragShaderResources.stage_outputs) {
		uint32_t binding = fragShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationBinding);
		uint32_t location = fragShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationLocation);
		uint32_t offset = fragShaderGLSL.get_decoration(resource.id, spv::Decoration::DecorationOffset);

		spirv_cross::SPIRType type = fragShaderGLSL.get_type(resource.type_id);

		print_indent(2, "Resource ID: %u\n", resource.id);
		print_indent(3, "Name: %s\n", fragShaderGLSL.get_name(resource.id).c_str());
		print_indent(3, "Fallback Name: %s\n", fragShaderGLSL.get_fallback_name(resource.id).c_str());
		print_indent(3, "Type Name: %s\n", fragShaderGLSL.get_name(resource.type_id).c_str());
		print_indent(3, "Type Fallback Name: %s\n", fragShaderGLSL.get_fallback_name(resource.type_id).c_str());
		print_indent(3, "Type ID: %u\n", resource.type_id);
		print_indent(3, "Binding: %u\n", binding);
		print_indent(3, "Location: %u\n", location);
		print_indent(3, "Offset: %u\n", offset);
	}

	printf("\n");

	printf("\n---\n\n");

	std::string source = vertShaderGLSL.compile();
	printf("%s\n", source.c_str());

	system("pause");

	return 0;
}