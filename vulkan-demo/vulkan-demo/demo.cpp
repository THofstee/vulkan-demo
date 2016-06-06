#include <stdlib.h>
#include <vector>
#include <random>

#include <vulkan/vulkan.h>
#include "vk_cpp.hpp"

#include "SPIRV/GlslangToSpv.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define DEBUG true
#define DEBUG_CLEANUP true

#define WIDTH  640
#define HEIGHT 480
#define VSYNC false

// I don't want windows.h
#ifndef DWORD
typedef unsigned long DWORD;
#endif

// Use the NVIDIA GPU if the system has Optimus
extern "C" {
	_declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

bool quit;

/*****************************************************************************
* DEBUG
*****************************************************************************/

// Function pointers
//PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT;

// Only execute actions of D_/D__ in DEBUG mode
// D_ for single line statements
#if DEBUG
#define D_(x) do { x; } while(0)
#else
#define D_(x) do {    } while(0)
#endif

// D__ is to be treated like #if DEBUG ... #endif
#if DEBUG
#define D__(x) do { x } while(0)
#else
#define D__(x) do {   } while(0)
#endif

// GLFW Error Callback
void error_callback(int error, const char* description)
{
	(void)error;
	fprintf(stderr, "%s\n", description);
}

// Debug callback direct to stderr
VKAPI_ATTR VkBool32 VKAPI_CALL DebugToStderrCallback(
	VkDebugReportFlagsEXT      flags,
	VkDebugReportObjectTypeEXT objectType,
	uint64_t                   object,
	size_t                     location,
	int32_t                    messageCode,
	const char*                pLayerPrefix,
	const char*                pMessage,
	void*                      pUserData)
{
	(void)flags; (void)objectType; (void)object; (void)location; (void)messageCode; (void)pUserData;
	fprintf(stderr, "%s: %s\n", pLayerPrefix, pMessage);
	return VK_FALSE;
}

std::vector<VkDebugReportCallbackEXT> callbacks;

/*****************************************************************************
* STRUCTS/CLASSES
******************************************************************************/

struct float2 {
	float u, v;
};

struct float3 {
	float x, y, z;
};

struct float4 {
	float x, y, z, w;
};

struct vertex {
	float3 pos;
	float4 col;
	float3 norm;
	float2 tex;
};

/*****************************************************************************
* SHADERS
******************************************************************************/

std::string vertShaderText =
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

std::string fragShaderText =
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

	return shaderSPV;
}

/*****************************************************************************
* HELPER FUNCTIONS
*****************************************************************************/

// GLFW Window Close Callback
void winclose_callback(GLFWwindow* window)
{
	(void)window;
	quit = true;
}

GLFWwindow* init_glfw() {
	// Initialize GLFW
	if (glfwInit() == GLFW_FALSE) {
		fprintf(stderr, "ERROR: GLFW failed to initialize.\n");
		system("pause");
		exit(-1);
	}

	// Set GLFW Error Callback
	glfwSetErrorCallback(error_callback);

	// Window creation
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Not using OpenGL so no need to create context
	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "instance", nullptr, nullptr);

	// Set GLFW Window Close Callback
	glfwSetWindowCloseCallback(window, winclose_callback);

	// Check for Vulkan support
	if (glfwVulkanSupported() == GLFW_FALSE) {
		fprintf(stderr, "ERROR: Vulkan is not supported.\n");
		system("pause");
		exit(-1);
	}

	return window;
}

void init_glslang() {
	glslang::InitializeProcess();
}

void finalize_glslang() {
	glslang::FinalizeProcess();
}

// Create callbacks for debugging
void create_callbacks(const vk::Instance& instance) {
	PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT =
		reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>
		(instance.getProcAddr("vkCreateDebugReportCallbackEXT"));

	vk::DebugReportCallbackCreateInfoEXT callbackCreateInfo(
		vk::DebugReportFlagsEXT(
			vk::DebugReportFlagBitsEXT::eError |
			vk::DebugReportFlagBitsEXT::eWarning |
			vk::DebugReportFlagBitsEXT::ePerformanceWarning
		),
		&DebugToStderrCallback,
		nullptr
	);

	VkDebugReportCallbackEXT callback;
	//TODO: See if this can be wrapped in VKCPP again, if not do error checking
	VkResult result = vkCreateDebugReportCallbackEXT(instance, &(VkDebugReportCallbackCreateInfoEXT)callbackCreateInfo, nullptr, &callback);
	callbacks.push_back(callback);
	//instance.createDebugReportCallbackEXT(callbackCreateInfo);
}

vk::Instance create_instance() {
	// Check required InstanceExtensions for GLFW
	uint32_t count;
	const char** extensions = glfwGetRequiredInstanceExtensions(&count);

	// Application and Instance Info
	vk::ApplicationInfo app_info = {};
	app_info.sType = vk::StructureType::eApplicationInfo;
	app_info.pNext = nullptr;
	app_info.pApplicationName = "instance";
	app_info.pEngineName = nullptr;
	app_info.engineVersion = 1;
	app_info.apiVersion = VK_API_VERSION_1_0;

	// Enabled Layers
	std::vector<const char*> enabledInstanceLayers;
	D_(enabledInstanceLayers.push_back("VK_LAYER_LUNARG_standard_validation"));

	// Enabled Extensions
	std::vector<const char*> enabledInstanceExtensions;
	//enabledInstanceExtensions.push_back("VK_KHR_surface");
	D_(enabledInstanceExtensions.push_back("VK_EXT_debug_report"));

	// Add GLFW Extensions
	for (size_t k = 0; k < count; k++) {
		enabledInstanceExtensions.push_back(const_cast<char*>(extensions[k]));
	}

	vk::InstanceCreateInfo inst_info(
		vk::InstanceCreateFlags(),
		&app_info,
		(uint32_t)enabledInstanceLayers.size(),
		enabledInstanceLayers.data(),
		(uint32_t)enabledInstanceExtensions.size(),
		enabledInstanceExtensions.data()
	);

	vk::Instance instance;

	// Create the instance
	try {
		instance = vk::createInstance(inst_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create callbacks
	D_(create_callbacks(instance));

	return instance;
}

std::vector<vk::PhysicalDevice> get_devices(const vk::Instance& instance) {
	std::vector<vk::PhysicalDevice> physicalDevices;

	try {
		physicalDevices = instance.enumeratePhysicalDevices();
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	return physicalDevices;
}

// Create Logical Device from Physical Device and Queue Info
vk::Device create_device(const vk::PhysicalDevice& physical_device, const std::vector<vk::DeviceQueueCreateInfo>& deviceQueueInfoVec) {
	std::vector<const char*> enabledDeviceLayers;
	D_(enabledDeviceLayers.push_back("VK_LAYER_LUNARG_standard_validation"));

	std::vector<const char*> enabledDeviceExtensions;
	enabledDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	vk::PhysicalDeviceFeatures enabledDeviceFeatures;
	try {
		enabledDeviceFeatures = physical_device.getFeatures();
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::DeviceCreateInfo deviceInfo(
		vk::DeviceCreateFlags(),
		(uint32_t)deviceQueueInfoVec.size(),
		deviceQueueInfoVec.data(),
		(uint32_t)enabledDeviceLayers.size(),
		enabledDeviceLayers.data(),
		(uint32_t)enabledDeviceExtensions.size(),
		enabledDeviceExtensions.data(),
		&enabledDeviceFeatures
	);

	vk::Device device;

	try {
		device = physical_device.createDevice(deviceInfo);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	return device;
}

// Create Render Pass
vk::RenderPass create_render_pass(const vk::Device& device, const vk::Format& colorFormat) {
	std::vector<vk::AttachmentDescription> attachment_descriptions;
	attachment_descriptions.push_back( // framebuffer
		vk::AttachmentDescription(
			vk::AttachmentDescriptionFlags(),
			colorFormat,
			vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eColorAttachmentOptimal
		)
	);
	attachment_descriptions.push_back( // depth buffer
		vk::AttachmentDescription(
			vk::AttachmentDescriptionFlags(),
			vk::Format::eD16Unorm,
			vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare,
			vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eDepthStencilAttachmentOptimal,
			vk::ImageLayout::eDepthStencilAttachmentOptimal
		)
	);

	std::vector<vk::AttachmentReference> color_attachment_references;
	color_attachment_references.push_back(
		vk::AttachmentReference(
			0,
			vk::ImageLayout::eColorAttachmentOptimal
		)
	);

	std::vector<vk::AttachmentReference> depth_attachment_references;
	depth_attachment_references.push_back(
		vk::AttachmentReference(
			1,
			vk::ImageLayout::eDepthStencilAttachmentOptimal
		)
	);

	std::vector<vk::SubpassDescription> subpass_descriptions;
	subpass_descriptions.push_back(
		vk::SubpassDescription(
			vk::SubpassDescriptionFlags(),
			vk::PipelineBindPoint::eGraphics,
			0,
			nullptr,
			(uint32_t)color_attachment_references.size(),
			color_attachment_references.data(),
			nullptr,
			depth_attachment_references.data(),
			0,
			nullptr
		)
	);

	vk::RenderPassCreateInfo render_pass_create_info(
		vk::RenderPassCreateFlags(),
		(uint32_t)attachment_descriptions.size(),
		attachment_descriptions.data(),
		(uint32_t)subpass_descriptions.size(),
		subpass_descriptions.data(),
		0,
		nullptr
	);

	vk::RenderPass render_pass;

	try {
		render_pass = device.createRenderPass(render_pass_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	return render_pass;
}

// Update buffer memory
void update_memory(vk::Device device, vk::DeviceMemory device_memory, void* src, size_t size) {
	void* device_mapped_memory;
	try {
		device_mapped_memory = device.mapMemory(device_memory, 0, size, vk::MemoryMapFlags());
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	memcpy(device_mapped_memory, src, size);

	try {
		device.unmapMemory(device_memory);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}
}

uint32_t get_memory_type(vk::PhysicalDevice physical_device, uint32_t memory_type_bits, vk::MemoryPropertyFlags memory_property_flags) {
	uint32_t memory_type_index = 0;

	for (uint32_t k = 0; k < 32; k++) {
		if ((memory_type_bits & 1) == 1) {
			if ((physical_device.getMemoryProperties().memoryTypes[k].propertyFlags & memory_property_flags) == memory_property_flags) {
				memory_type_index = k;
				return memory_type_index;
			}
		}

		memory_type_bits >>= 1;
	}

	fprintf(stderr, "Couldn't find any matching memory types.\n");
	exit(-1);
}

// Create Shader Modules



/*****************************************************************************
* PRINT FUNCTIONS
*****************************************************************************/

// Print device info
void print_device_info(const std::vector<vk::PhysicalDevice>& physicalDevices) {
	for (uint32_t i = 0; i < physicalDevices.size(); i++) {
		// Print out device properties
		vk::PhysicalDeviceProperties deviceProperties = physicalDevices.at(i).getProperties();
		printf("Device ID:      %d\n", deviceProperties.deviceID);
		printf("Driver Version: %d\n", deviceProperties.driverVersion);
		printf("Device Name:    %s\n", deviceProperties.deviceName);
		printf("Device Type:    %d\n", deviceProperties.deviceType);
		printf("API Version:    %d.%d.%d\n",
			VK_VERSION_MAJOR(deviceProperties.apiVersion),
			VK_VERSION_MINOR(deviceProperties.apiVersion),
			VK_VERSION_PATCH(deviceProperties.apiVersion));

		printf("\n");

		// Print out device features
		printf("Supported Features:\n");
		vk::PhysicalDeviceFeatures deviceFeatures = physicalDevices.at(i).getFeatures();
		if (deviceFeatures.shaderClipDistance == VK_TRUE) printf("Shader Clip Distance\n");
		if (deviceFeatures.textureCompressionBC == VK_TRUE) printf("BC Texture Compression\n");

		printf("\n");

		// Print out device queue info
		std::vector<vk::QueueFamilyProperties> familyProperties = physicalDevices.at(i).getQueueFamilyProperties();

		for (int j = 0; j < familyProperties.size(); j++) {
			printf("Count of Queues: %d\n", familyProperties[j].queueCount);
			printf("Supported operations on this queue:\n");
			printf("%s\n", to_string(familyProperties[j].queueFlags).c_str());
			printf("\n");
		}

		// Readability
		printf("\n---\n\n");
	}
}

// Print surface capabilities
void print_surface_capabilities(const vk::PhysicalDevice& physical_device, const vk::SurfaceKHR& surface) {
	printf("Surface Capabilities:\n");

	vk::SurfaceCapabilitiesKHR surfaceCapabilities = physical_device.getSurfaceCapabilitiesKHR(surface).value;

	printf("Min Image Count: %d\n", surfaceCapabilities.minImageCount);
	printf("Max Image Count: %d\n", surfaceCapabilities.maxImageCount);

	vk::Extent2D surfaceResolution = surfaceCapabilities.currentExtent;
	printf("Width:  %d\n", surfaceResolution.width);
	printf("Height: %d\n", surfaceResolution.height);

	printf("Supported transforms: %s\n", to_string(surfaceCapabilities.supportedTransforms).c_str());

	std::vector<vk::PresentModeKHR> presentModes = physical_device.getSurfacePresentModesKHR(surface);
	std::string pmodestr = "";
	for (int k = 0; k < presentModes.size(); k++) {
		pmodestr += to_string(presentModes.at(k));
		pmodestr += " | ";
	}
	printf("Present modes supported: {%s}\n", pmodestr.substr(0, pmodestr.size() - 3).c_str());

	printf("Supported Usages: %s\n", to_string(surfaceCapabilities.supportedUsageFlags).c_str());

	printf("\n");
}

/*****************************************************************************
* VERTEX TEST FUNCTIONS
******************************************************************************/

// Generates a quad
std::vector<vertex> test_quad() {
	return{
		{ { -1.0f, -1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 1.0f, -1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f } },
		{ { -1.0f,  1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 1.0f } },
		{ { -1.0f,  1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 1.0f } },
		{ { 1.0f, -1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 1.0f, 0.0f } },
		{ { 1.0f,  1.0f, 0.0f },{ 1.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 1.0f, 1.0f } }
	};
}

// Generates a triangle
std::vector<vertex> test_triangle() {
	return{
		{ { -0.8f,  0.8f, -2.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f, -2.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f, -2.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }
	};
}

// Generates a set of vertices to test zbuffer
std::vector<vertex> test_zbuffer() {
	return{
		{ { -0.7f,  0.7f,  0.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f,  0.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f,  0.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f,  0.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f,  0.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f, -0.7f,  0.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f, -0.7f, -1.0f },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //these normals
		{ { -0.7f,  0.7f, -1.0f },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //are actually
		{ { 0.7f,  0.7f,  1.0f },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }  //wrong.
	};
}

// Test zbuffer with random colors
std::vector<vertex> test_zbuffer_rc() {
	std::random_device rd;
	std::default_random_engine gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	return{
		{ { -0.7f,  0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f, -0.7f,  0.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f, -0.7f, -1.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //these normals
		{ { -0.7f,  0.7f, -1.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //are actually
		{ { 0.7f,  0.7f,  1.0f },{ (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }  //wrong.
	};
}

// Test projection matrix
std::vector<vertex> test_proj() {
	return{
		{ { -0.7f,  0.7f, -3.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f, -3.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f, -3.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { -0.7f, -0.7f, -3.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f,  0.7f, -3.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 0.7f, -0.7f, -3.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } },
		{ { 1.2f, -1.6f, -5.0f },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //these normals
		{ { -1.6f,  1.2f, -5.0f },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }, //are acutally
		{ { 0.3f,  0.3f, -2.0f },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }  //wrong.
	};
}

//             _.-+.
//        _.-""     '.
//    +:""            '.
//    J \               '.
//     L \             _.-+
//     |  '.       _.-"   |
//     J    \  _.-"       L
//      L    +"          J
//      +    |           |
//       \   |          .+
//        \  |       .-'
//         \ |    .-'
//          \| .-'
//           +'   hs
std::vector<vertex> test_cube() {
	float mag = 0.7f;

	return{
		//Front (red)
		{ { -mag,  mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },{ 1.0f, 0.0f, 0.0f, 1.0f },{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		//Left (green)
		{ { -mag,  mag,  mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },{ 0.0f, 1.0f, 0.0f, 1.0f },{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		//Right (blue)
		{ { mag,  mag,  mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },{ 0.0f, 0.0f, 1.0f, 1.0f },{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		//Top (yellow)
		{ { -mag,  mag,  mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag,  mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },{ 1.0f, 1.0f, 0.0f, 1.0f },{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		//Bottom (magenta)
		{ { -mag, -mag,  mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },{ 1.0f, 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		//Back (cyan)
		{ { -mag,  mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },{ 0.0f, 1.0f, 1.0f, 1.0f },{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } }
	};

}

//             _.-+.
//        _.-""     '.
//    +:""            '.
//    J \               '.
//     L \             _.-+
//     |  '.       _.-"   |
//     J    \  _.-"       L
//      L    +"          J
//      +    |           |
//       \   |          .+
//        \  |       .-'
//         \ |    .-'
//          \| .-'
//           +'   hs
// single color cube
std::vector<vertex> test_cube_solid() {
	std::random_device rd;
	std::default_random_engine gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	float4 col = { (float)dis(gen), (float)dis(gen), (float)dis(gen), 1.0f };

	float mag = 0.7f;

	return{
		//Front
		{ { -mag,  mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },col,{ 0.0f,  0.0f,  1.0f },{ 0.0f, 0.0f } },
		//Left
		{ { -mag,  mag,  mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },col,{ -1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		//Right
		{ { mag,  mag,  mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },col,{ 1.0f,  0.0f,  0.0f },{ 0.0f, 0.0f } },
		//Top
		{ { -mag,  mag,  mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag,  mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag,  mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag,  mag, -mag },col,{ 0.0f,  1.0f,  0.0f },{ 0.0f, 0.0f } },
		//Bottom
		{ { -mag, -mag,  mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag,  mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag,  mag },col,{ 0.0f, -1.0f,  0.0f },{ 0.0f, 0.0f } },
		//Back
		{ { -mag,  mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag,  mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { -mag, -mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } },
		{ { mag, -mag, -mag },col,{ 0.0f,  0.0f, -1.0f },{ 0.0f, 0.0f } }
	};

}

// Use TinyObjLoader to load .obj files
std::vector<vertex> load_obj(const char* inputfile, const char* mtldir) {
	std::vector<vertex> vertices;

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string err;
	bool ret = tinyobj::LoadObj(shapes, materials, err, inputfile, mtldir, tinyobj::triangulation | tinyobj::calculate_normals);

	if (!err.empty()) {
		fprintf(stderr, "%s\n", err.c_str());
	}
	if (!ret) {
		fprintf(stderr, "tinyobjloader failed to load obj\n");
		system("pause");
		exit(-1);
	}

	printf("Num shapes:    %llu\n", shapes.size());
	printf("Num materials: %llu\n", shapes.size());

	std::random_device rd;
	std::default_random_engine gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	for (size_t i = 0; i < shapes.size(); i++) {
		printf("shape[%llu].name = %s\n", i, shapes[i].name.c_str());
		printf("Size of shape[%llu].indices: %llu\n", i, shapes[i].mesh.indices.size());
		printf("Size of shape[%llu].material_ids: %llu\n", i, shapes[i].mesh.material_ids.size());
		printf("shape[%llu].vertices: %llu\n", i, shapes[i].mesh.positions.size());
		printf("shape[%llu].texcoords: %llu\n", i, shapes[i].mesh.texcoords.size());

		size_t indexOffset = 0;
		for (size_t n = 0; n < shapes[i].mesh.num_vertices.size(); n++) {
			for (size_t f = 0; f < shapes[i].mesh.num_vertices[n]; f++) {
				unsigned int v = shapes[i].mesh.indices[indexOffset + f];

				float Kd[3] = { 1 };
				if (shapes[i].mesh.material_ids[n] >= 0) {
					memcpy(&Kd[0], &materials[shapes[i].mesh.material_ids[n]].diffuse[0], 3 * sizeof(float));
				}

				vertices.push_back({
					{ shapes.at(i).mesh.positions[3 * v + 0], shapes.at(i).mesh.positions[3 * v + 1], shapes.at(i).mesh.positions[3 * v + 2] },
					{ Kd[0], Kd[1], Kd[2], 1.0f },
					{ shapes.at(i).mesh.normals[3 * v + 0], shapes.at(i).mesh.normals[3 * v + 1], shapes.at(i).mesh.normals[3 * v + 2] },
					{ shapes.at(i).mesh.texcoords[2 * v + 0], shapes.at(i).mesh.texcoords[2 * v + 1] }
				});
			}

			indexOffset += shapes[i].mesh.num_vertices[n];
		}
	}


	printf("\n---\n\n");

	return vertices;
}

// Read vertex data from vtx file
#include <fstream>
enum class VertexDataType
{
	Half, Float, Char, UChar, Short, UShort, Int,
	UInt4_10_10_10_2
};
struct vertexAttribute {
	std::wstring name;
	VertexDataType type;
	int components;
	bool normalized;
};
std::vector<vertex> load_vtx(const char* inputfile) {
	std::vector<vertex> vertices;

	std::ifstream fin(inputfile, std::ifstream::binary);

	// Read bounding box
	float3 min, max;
	fin.read((char*)&min.x, sizeof(float));
	fin.read((char*)&min.y, sizeof(float));
	fin.read((char*)&min.z, sizeof(float));
	fin.read((char*)&max.x, sizeof(float));
	fin.read((char*)&max.y, sizeof(float));
	fin.read((char*)&max.z, sizeof(float));

	printf("Bounds:\tmin: { %.2f, %.2f, %.2f }\n\tmax: { %.2f, %.2f, %.2f }\n", min.x, min.y, min.z, max.x, max.y, max.z);

	// Read attributes
	uint32_t attributes;
	fin.read((char*)&attributes, sizeof(uint32_t));
	printf("Attributes: %u\n", attributes);

	std::vector<vertexAttribute> vattrs;
	for (int i = 0; i < attributes; i++) {
		vertexAttribute vattr;
		uint32_t name_len;
		fin.read((char*)&name_len, sizeof(uint32_t));
		wchar_t* cname = new wchar_t[name_len+1];
		char comp;
		bool norm;
		uint16_t type;

		fin.read((char*)cname, name_len * sizeof(wchar_t));
		cname[name_len] = '\0';
		fin.read((char*)&comp, sizeof(char));
		fin.read((char*)&norm, sizeof(bool));
		fin.read((char*)&type, sizeof(uint16_t));

		vattr.name = cname;
		vattr.components = comp;
		vattr.normalized = norm != 0;
		vattr.type = (VertexDataType)type;
		
		printf("name_len: %u\n", name_len);
		printf("name:"); wprintf(vattr.name.c_str()); printf("\n");
		printf("comp: %d\n", vattr.components);
		printf("norm: %d\n", vattr.normalized);
		printf("type: %d\n", vattr.type);

		vattrs.push_back(vattr);
	}

	int32_t vertSize; fin.read((char*)&vertSize, sizeof(int32_t));
	printf("vert size: %d\n", vertSize);

	while (fin.good()) {
		float3 pos;
		float3 norm;
		float2 tex;
		float4 col = { 1.0f, 1.0f, 1.0f, 1.0f };

		for (int k = 0; k < vattrs.size(); k++) {
			if (vattrs.at(k).name == L"vert_position") {
				if (vattrs.at(k).components >= 3) {
					fin.read((char*)&pos.x, 3 * sizeof(float));
				}
				else {
					fprintf(stderr, "vtx loader: file format is incompatible\n");
					exit(-1);
				}

				if (vattrs.at(k).components > 3) {
					fin.ignore((vattrs.at(k).components - 3) * sizeof(float));
				}
			}
			else if (vattrs.at(k).name == L"vert_normal") {
				if (vattrs.at(k).components >= 3) {
					fin.read((char*)&norm.x, 3 * sizeof(float));
				}
				else {
					fprintf(stderr, "vtx loader: file format is incompatible\n");
					exit(-1);
				}

				if (vattrs.at(k).components > 3) {
					fin.ignore((vattrs.at(k).components - 3) * sizeof(float));
				}
			}
			else if (vattrs.at(k).name == L"vert_texCoord0") {
				if (vattrs.at(k).components >= 2) {
					fin.read((char*)&tex.u, 2 * sizeof(float));
				}
				else {
					fprintf(stderr, "vtx loader: file format is incompatible\n");
					exit(-1);
				}

				if (vattrs.at(k).components > 2) {
					fin.ignore((vattrs.at(k).components - 2) * sizeof(float));
				}
			}
			else {
				fin.ignore((vattrs.at(k).components) * sizeof(float));
			}
		}

		vertices.push_back({ pos, col, norm, tex });
	}

	printf("\n---\n\n");

	return vertices;
}

/*****************************************************************************
* MAIN FUNCTION
*****************************************************************************/

int main() {
	// Vertices
	std::vector<vertex> vertices;
	//vertices = test_quad();
	//vertices = load_obj("../meshes/cube/cube.obj", "../meshes/cube/");
	vertices = load_vtx("../meshes/couch/couch.vtx");

	// Uniforms
	struct {
		glm::mat4 projection_matrix;
		glm::mat4 view_matrix;
		glm::mat4 model_matrix;
	} uboVS;

	uboVS.projection_matrix = glm::scale(glm::vec3(1, -1, 1)) * glm::perspective(glm::radians(60.0f), (float)WIDTH / (float)HEIGHT, 1.0f, 512.0f);
	//uboVS.view_matrix = glm::lookAt(glm::vec3(0, 100, 250), glm::vec3(0, 100, 0), glm::vec3(0, 1, 0)); //sackboy
	//uboVS.view_matrix = glm::lookAt(glm::vec3(0, 0, 2), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)); //cube
	uboVS.view_matrix = glm::lookAt(glm::vec3(-90, 50, 0), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)); //couch
	uboVS.model_matrix = glm::mat4();

	// Initialize GLFW
	GLFWwindow* window = init_glfw();

	// Initialize glslang
	init_glslang();

	// Create Vulkan Instance
	vk::Instance instance = create_instance();

	// Choose Physical Device
	std::vector<vk::PhysicalDevice> physicalDevices = get_devices(instance);
	print_device_info(physicalDevices);
	vk::PhysicalDevice physical_device = physicalDevices.at(0);

	// Initialize Queues
	std::vector<vk::DeviceQueueCreateInfo> deviceQueueInfoVec;
	vk::DeviceQueueCreateInfo deviceQueueInfo;
	deviceQueueInfo.sType = vk::StructureType::eDeviceQueueCreateInfo;
	deviceQueueInfo.pNext = nullptr;
	deviceQueueInfo.flags = vk::DeviceQueueCreateFlags();
	deviceQueueInfo.queueFamilyIndex = 0; // use the first queue in the family list

	float queuePriorities[] = { 1.0f };
	deviceQueueInfo.queueCount = 1;
	deviceQueueInfo.pQueuePriorities = queuePriorities;

	deviceQueueInfoVec.push_back(deviceQueueInfo);

	// Create Logical Device
	vk::Device device = create_device(physical_device, deviceQueueInfoVec);

	// Query for Vulkan presentation support
	VkPhysicalDevice native_physicalDevice = physical_device;
	if (glfwGetPhysicalDevicePresentationSupport(instance, native_physicalDevice, 0) == GLFW_FALSE) {
		fprintf(stderr, "ERROR: Selected queue does not support image presentation.\n");
		system("pause");
		exit(-1);
	}

	// Create Vulkan surface
	VkInstance native_instance = instance;
	VkSurfaceKHR native_surface;
	VkResult native_res = glfwCreateWindowSurface(native_instance, window, nullptr, &native_surface);
	if (native_res) {
		fprintf(stderr, "ERROR: Window surface creation failed.\n");
		system("pause");
		exit(-1);
	}

	vk::SurfaceKHR surface = native_surface;

	// Make validation layer happy, this is same check as glfwGetPhysicalDevicePresentationSupport
	physical_device.getSurfaceSupportKHR(0, surface);

	// Print surface capabilities
	print_surface_capabilities(physical_device, surface);

	// Get Physical Device Color Format and Space
	vk::Format colorFormat;
	std::vector<vk::SurfaceFormatKHR> surfaceFormats = physical_device.getSurfaceFormatsKHR(surface);

	if ((surfaceFormats.size() == 1) && (surfaceFormats.at(0).format == vk::Format::eUndefined)) {
		colorFormat = vk::Format::eB8G8R8A8Unorm;
	}
	else {
		colorFormat = surfaceFormats.at(0).format;
		printf("Preferred ");
	}

	vk::ColorSpaceKHR colorSpace = surfaceFormats.at(0).colorSpace;

	printf("Color Format: %s\n", to_string(colorFormat).c_str());
	printf("Color Space: %s\n", to_string(colorSpace).c_str());
	printf("\n");

	// Select presentation mode
	vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo; // Fifo presentation mode is guaranteed
	std::vector<vk::PresentModeKHR> presentModes = physical_device.getSurfacePresentModesKHR(surface);
	for (int k = 0; k < presentModes.size(); k++) {
		// If we can use mailbox, use it.
		if (presentModes.at(k) == vk::PresentModeKHR::eMailbox) {
			presentMode = vk::PresentModeKHR::eMailbox;
			if (VSYNC == true) {
				break;
			}
		}
		if (VSYNC == false) {
			if (presentModes.at(k) == vk::PresentModeKHR::eImmediate) {
				// If we don't care about VSYNC, Immediate is the lowest latency mode
				presentMode = vk::PresentModeKHR::eImmediate;
				break;
			}
			else if (presentModes.at(k) == vk::PresentModeKHR::eFifoRelaxed) {
				// Not sure if this is preferrable to Mailbox
				presentMode = vk::PresentModeKHR::eFifoRelaxed;
			}
		}
	}

	printf("Presentation Mode: %s\n", to_string(presentMode).c_str());
	///std::cout << "Presentation Mode: " << to_string(presentMode) << std::endl;

	// Select number of images in swap chain
	vk::SurfaceCapabilitiesKHR surfaceCapabilities = physical_device.getSurfaceCapabilitiesKHR(surface).value;

	uint32_t desiredSwapchainImages = 2; // Default double buffering
	if (presentMode != vk::PresentModeKHR::eImmediate) {
		desiredSwapchainImages = 3; // If are using a VSYNC presentation mode, triple buffer
	}

	if (desiredSwapchainImages < surfaceCapabilities.minImageCount) {
		desiredSwapchainImages = surfaceCapabilities.minImageCount;
	}
	else if (desiredSwapchainImages > surfaceCapabilities.maxImageCount) {
		desiredSwapchainImages = surfaceCapabilities.maxImageCount;
	}

	printf("Num swapchain images: %d\n", desiredSwapchainImages);
	printf("\n");

	// Select swapchain size
	vk::Extent2D swapchainExtent = {};
	if (surfaceCapabilities.currentExtent.width == -1) {
		swapchainExtent.width = WIDTH;
		swapchainExtent.height = HEIGHT;
	}
	else {
		swapchainExtent = surfaceCapabilities.currentExtent;
	}

	// Select swapchain pre-transform
	// (Can be useful on tablets, etc.)
	vk::SurfaceTransformFlagBitsKHR preTransform = surfaceCapabilities.currentTransform;
	if (surfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
		// Select identity transform if we can
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}

	// Create swapchain
	vk::SwapchainCreateInfoKHR swapchainCreateInfo(
		vk::SwapchainCreateFlagsKHR(),
		surface,
		desiredSwapchainImages,
		colorFormat,
		colorSpace,
		swapchainExtent,
		1,
		vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment),
		vk::SharingMode::eExclusive,
		0,
		nullptr,
		preTransform,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		presentMode,
		VK_TRUE,
		VK_NULL_HANDLE
	);

	vk::SwapchainKHR swapchain;

	try {
		swapchain = device.createSwapchainKHR(swapchainCreateInfo);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	//
	// END CREATE SWAPCHAIN
	//

	// Create semaphores
	vk::SemaphoreCreateInfo semaphore_create_info;
	vk::Semaphore imageAvailableSemaphore = device.createSemaphore(semaphore_create_info);
	vk::Semaphore renderingFinishedSemaphore = device.createSemaphore(semaphore_create_info);

	// Get swapchain images
	std::vector<vk::Image> images = device.getSwapchainImagesKHR(swapchain);

	// Create Render Pass
	vk::RenderPass render_pass = create_render_pass(device, colorFormat);

	// Create depth buffer
	// Create depth image
	vk::ImageCreateInfo depth_image_create_info(
		vk::ImageCreateFlags(),
		vk::ImageType::e2D,
		vk::Format::eD16Unorm,
		vk::Extent3D(WIDTH, HEIGHT, 1),
		1,
		1,
		vk::SampleCountFlagBits::e1,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlags(vk::ImageUsageFlagBits::eDepthStencilAttachment),
		vk::SharingMode::eExclusive,
		0,
		NULL,
		vk::ImageLayout::eUndefined
	);

	vk::Image depth_image;
	try {
		depth_image = device.createImage(depth_image_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Get image memory requirements
	vk::MemoryRequirements depth_memory_requirements;
	try {
		depth_memory_requirements = device.getImageMemoryRequirements(depth_image);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	uint32_t depth_memory_type_index = get_memory_type(physical_device, depth_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);

	// Allocate device memory for image
	vk::MemoryAllocateInfo depth_memory_allocate_info(
		depth_memory_requirements.size,
		depth_memory_type_index
	);

	vk::DeviceMemory depth_image_memory;
	try {
		depth_image_memory = device.allocateMemory(depth_memory_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Bind memory to device
	try {
		device.bindImageMemory(depth_image, depth_image_memory, (vk::DeviceSize)0);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create depth image view
	vk::ImageViewCreateInfo depth_image_view_create_info(
		vk::ImageViewCreateFlags(),
		depth_image,
		vk::ImageViewType::e2D,
		depth_image_create_info.format,
		vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
		vk::ImageSubresourceRange(
			vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth),
			0,
			1,
			0,
			1
		)
	);

	vk::ImageView depth_image_view;
	try {
		depth_image_view = device.createImageView(depth_image_view_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Framebuffer
	std::vector<vk::ImageView> imageViews(images.size());
	std::vector<vk::Framebuffer> framebuffers(images.size());
	for (int k = 0; k < images.size(); k++) {
		// Create image views
		vk::ImageViewCreateInfo colorAttachmentView(
			vk::ImageViewCreateFlags(),
			images.at(k),
			vk::ImageViewType::e2D,
			colorFormat,
			vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
			vk::ImageSubresourceRange(vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor), 0, 1, 0, 1)
		);

		try {
			imageViews.at(k) = device.createImageView(colorAttachmentView);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		std::vector<vk::ImageView> framebuffer_attachments;
		framebuffer_attachments.push_back(imageViews.at(k));
		framebuffer_attachments.push_back(depth_image_view);

		// Create framebuffers
		vk::FramebufferCreateInfo framebuffer_create_info(
			vk::FramebufferCreateFlags(),
			render_pass,
			(uint32_t)framebuffer_attachments.size(),
			framebuffer_attachments.data(),
			WIDTH,
			HEIGHT,
			1
		);

		try {
			framebuffers[k] = device.createFramebuffer(framebuffer_create_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}
	}

	// Create Shader Modules
	std::vector<unsigned int> vertShaderSPV;
	try {
		vertShaderSPV = GLSLtoSPV(vk::ShaderStageFlagBits::eVertex, vertShaderText);
	}
	catch (std::runtime_error e) {
		fprintf(stderr, "glslang: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	std::vector<unsigned int> fragShaderSPV;
	try {
		fragShaderSPV = GLSLtoSPV(vk::ShaderStageFlagBits::eFragment, fragShaderText);
	}
	catch (std::runtime_error e) {
		fprintf(stderr, "glslang: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::ShaderModuleCreateInfo vert_shader_module_create_info(
		vk::ShaderModuleCreateFlags(),
		vertShaderSPV.size() * sizeof(unsigned int),
		reinterpret_cast<const uint32_t*>(vertShaderSPV.data())
	);

	vk::ShaderModuleCreateInfo frag_shader_module_create_info(
		vk::ShaderModuleCreateFlags(),
		fragShaderSPV.size() * sizeof(unsigned int),
		reinterpret_cast<const uint32_t*>(fragShaderSPV.data())
	);

	vk::ShaderModule vert_shader_module;
	try {
		vert_shader_module = device.createShaderModule(vert_shader_module_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::ShaderModule frag_shader_module;
	try {
		frag_shader_module = device.createShaderModule(frag_shader_module_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Pipeline Descriptions
	// Create Shader Stage Description
	std::vector<vk::PipelineShaderStageCreateInfo> shader_stage_create_infos;

	vk::PipelineShaderStageCreateInfo vert_stage_create_info(
		vk::PipelineShaderStageCreateFlags(),
		vk::ShaderStageFlagBits::eVertex,
		vert_shader_module,
		"main",
		nullptr
	);

	vk::PipelineShaderStageCreateInfo frag_stage_create_info(
		vk::PipelineShaderStageCreateFlags(),
		vk::ShaderStageFlagBits::eFragment,
		frag_shader_module,
		"main",
		nullptr
	);

	shader_stage_create_infos.push_back(vert_stage_create_info);
	shader_stage_create_infos.push_back(frag_stage_create_info);

	// Create Vertex Input Description
	std::vector<vk::VertexInputBindingDescription> vertex_input_binding_descriptions;
	vertex_input_binding_descriptions.push_back(
		vk::VertexInputBindingDescription(
			0,
			sizeof(vertices.front()),
			vk::VertexInputRate::eVertex
		)
	);

	std::vector<vk::VertexInputAttributeDescription> vertex_input_attribute_descriptions;
	vertex_input_attribute_descriptions.push_back( //Position
		vk::VertexInputAttributeDescription(
			0,
			0,
			vk::Format::eR32G32B32Sfloat,
			0
		)
	);
	vertex_input_attribute_descriptions.push_back( // Color
		vk::VertexInputAttributeDescription(
			1,
			0,
			vk::Format::eR32G32B32A32Sfloat,
			sizeof(vertices.front().pos)
		)
	);
	vertex_input_attribute_descriptions.push_back( // Normal
		vk::VertexInputAttributeDescription(
			2,
			0,
			vk::Format::eR32G32B32Sfloat,
			(sizeof(vertices.front().pos) + sizeof(vertices.front().col))
		)
	);
	vertex_input_attribute_descriptions.push_back( // Texcoords
		vk::VertexInputAttributeDescription(
			3,
			0,
			vk::Format::eR32G32Sfloat,
			(sizeof(vertices.front().pos) + sizeof(vertices.front().col) + sizeof(vertices.front().norm))
		)
	);

	vk::PipelineVertexInputStateCreateInfo vertex_input_state_create_info(
		vk::PipelineVertexInputStateCreateFlags(),
		(uint32_t)vertex_input_binding_descriptions.size(),
		vertex_input_binding_descriptions.data(),
		(uint32_t)vertex_input_attribute_descriptions.size(),
		vertex_input_attribute_descriptions.data()
	);

	// Create Input Assembly Description
	vk::PipelineInputAssemblyStateCreateInfo input_assembly_state_create_info(
		vk::PipelineInputAssemblyStateCreateFlags(),
		vk::PrimitiveTopology::eTriangleList,
		VK_FALSE
	);

	// Create Viewport Description
	std::vector<vk::Viewport> viewports;
	vk::Viewport viewport(
		0.0f,
		0.0f,
		(float)WIDTH,
		(float)HEIGHT,
		0.0f,
		1.0f
	);
	viewports.push_back(viewport);

	std::vector<vk::Rect2D> scissors;
	vk::Rect2D scissor(
		vk::Offset2D(0, 0),
		vk::Extent2D(WIDTH, HEIGHT)
	);
	scissors.push_back(scissor);

	vk::PipelineViewportStateCreateInfo viewport_state_create_info(
		vk::PipelineViewportStateCreateFlags(),
		(uint32_t)viewports.size(),
		viewports.data(),
		(uint32_t)scissors.size(),
		scissors.data()
	);

	// Create Rasterization Description
	vk::PipelineRasterizationStateCreateInfo rasterization_state_create_info(
		vk::PipelineRasterizationStateCreateFlags(),
		VK_FALSE,
		VK_FALSE,
		vk::PolygonMode::eFill,
		vk::CullModeFlags(vk::CullModeFlagBits::eBack),
		vk::FrontFace::eCounterClockwise,
		VK_FALSE,
		0.0f,
		0.0f,
		0.0f,
		1.0f
	);

	// Create Multisampling Description
	vk::PipelineMultisampleStateCreateInfo multisample_state_create_info(
		vk::PipelineMultisampleStateCreateFlags(),
		vk::SampleCountFlagBits::e1,
		VK_FALSE,
		1.0f,
		nullptr,
		VK_FALSE,
		VK_FALSE
	);

	// Create Depth Stencil Description
	vk::PipelineDepthStencilStateCreateInfo depth_stencil_create_info(
		vk::PipelineDepthStencilStateCreateFlags(),
		VK_TRUE,
		VK_TRUE,
		vk::CompareOp::eLessOrEqual,
		VK_FALSE,
		VK_FALSE,
		vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
		vk::StencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways, 0, 0, 0),
		0,
		0
	);

	// Create Blending State Description
	std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachment_states;
	vk::PipelineColorBlendAttachmentState color_blend_attachment_state(
		VK_FALSE,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::BlendFactor::eOne,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::ColorComponentFlags(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
	);
	color_blend_attachment_states.push_back(color_blend_attachment_state);

	vk::PipelineColorBlendStateCreateInfo color_blend_state_create_info(
		vk::PipelineColorBlendStateCreateFlags(),
		VK_FALSE,
		vk::LogicOp::eCopy,
		(uint32_t)color_blend_attachment_states.size(),
		color_blend_attachment_states.data(),
		{ 0.0f, 0.0f, 0.0f, 0.0f }
	);

	// Set up descriptor pool
	size_t num_textures = 2;
	std::vector<vk::DescriptorPoolSize> descriptor_pool_sizes;
	descriptor_pool_sizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1));
	descriptor_pool_sizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, (uint32_t)num_textures));

	vk::DescriptorPoolCreateInfo descriptor_pool_create_info(
		vk::DescriptorPoolCreateFlags(),
		1,
		(uint32_t)descriptor_pool_sizes.size(),
		descriptor_pool_sizes.data()
	);

	vk::DescriptorPool descriptor_pool;
	try {
		descriptor_pool = device.createDescriptorPool(descriptor_pool_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create descriptor set layout
	std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
	layout_bindings.push_back( // Vertex Shader Uniform Buffer
		vk::DescriptorSetLayoutBinding(
			0,
			vk::DescriptorType::eUniformBuffer,
			1,
			vk::ShaderStageFlags(vk::ShaderStageFlagBits::eVertex),
			nullptr
		)
	);
	layout_bindings.push_back(
		vk::DescriptorSetLayoutBinding(
			1,
			vk::DescriptorType::eCombinedImageSampler,
			1,
			vk::ShaderStageFlags(vk::ShaderStageFlagBits::eFragment),
			nullptr
		)
	);
	layout_bindings.push_back(
		vk::DescriptorSetLayoutBinding(
			2,
			vk::DescriptorType::eCombinedImageSampler,
			1,
			vk::ShaderStageFlags(vk::ShaderStageFlagBits::eFragment),
			nullptr
		)
	);

	vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
		vk::DescriptorSetLayoutCreateFlags(),
		(uint32_t)layout_bindings.size(),
		layout_bindings.data()
	);

	std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
	try {
		descriptor_set_layouts.push_back(device.createDescriptorSetLayout(descriptor_set_layout_create_info));
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Pipeline Layout
	vk::PipelineLayoutCreateInfo layout_create_info(
		vk::PipelineLayoutCreateFlags(),
		(uint32_t)descriptor_set_layouts.size(),
		descriptor_set_layouts.data(),
		0,
		nullptr
	);

	vk::PipelineLayout pipeline_layout;
	try {
		pipeline_layout = device.createPipelineLayout(layout_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Graphics Pipeline
	vk::GraphicsPipelineCreateInfo pipeline_create_info(
		vk::PipelineCreateFlags(),
		(uint32_t)shader_stage_create_infos.size(),
		shader_stage_create_infos.data(),
		&vertex_input_state_create_info,
		&input_assembly_state_create_info,
		nullptr,
		&viewport_state_create_info,
		&rasterization_state_create_info,
		&multisample_state_create_info,
		&depth_stencil_create_info,
		&color_blend_state_create_info,
		nullptr,
		pipeline_layout,
		render_pass,
		0,
		VK_NULL_HANDLE,
		-1
	);

	std::vector<vk::Pipeline> graphics_pipelines;
	try {
		graphics_pipelines = device.createGraphicsPipelines(VK_NULL_HANDLE, pipeline_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	//
	// BEGIN DRAW LOGIC
	//
	// Create command buffer memory pool
	vk::CommandPoolCreateInfo cmd_pool_create_info(
		vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
		0
	);

	vk::CommandPool command_pool;
	try {
		command_pool = device.createCommandPool(cmd_pool_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Setup command buffer allocation info
	vk::CommandBufferAllocateInfo setup_cmd_buffer_allocate_info(
		command_pool,
		vk::CommandBufferLevel::ePrimary,
		(uint32_t)1
	);

	// Create setup command buffers
	std::vector<vk::CommandBuffer> setup_command_buffers;
	try {
		setup_command_buffers = device.allocateCommandBuffers(setup_cmd_buffer_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}
	vk::CommandBuffer setup_command_buffer = setup_command_buffers.at(0);

	// Swapchain command buffers allocation info
	vk::CommandBufferAllocateInfo cmd_buffer_allocate_info(
		command_pool,
		vk::CommandBufferLevel::ePrimary,
		(uint32_t)images.size()
	);

	// Create swapchain command buffers
	std::vector<vk::CommandBuffer> command_buffers;
	try {
		command_buffers = device.allocateCommandBuffers(cmd_buffer_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Uniform Buffer
	vk::BufferCreateInfo uniform_buffer_create_info(
		vk::BufferCreateFlags(),
		vk::DeviceSize(sizeof(uboVS)),
		vk::BufferUsageFlags(vk::BufferUsageFlagBits::eUniformBuffer),
		vk::SharingMode::eExclusive,
		0,
		nullptr
	);

	vk::Buffer uniform_buffer;
	try {
		uniform_buffer = device.createBuffer(uniform_buffer_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::MemoryRequirements uniform_buffer_memory_requirements;
	try {
		uniform_buffer_memory_requirements = device.getBufferMemoryRequirements(uniform_buffer);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Allocate host visible (change to coherent?) memory
	uint32_t uniform_buffer_memory_type_index = get_memory_type(physical_device, uniform_buffer_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);

	vk::MemoryAllocateInfo uniform_buffer_allocate_info(
		uniform_buffer_memory_requirements.size,
		uniform_buffer_memory_type_index
	);

	vk::DeviceMemory uniform_buffer_device_memory;
	try {
		uniform_buffer_device_memory = device.allocateMemory(uniform_buffer_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	update_memory(device, uniform_buffer_device_memory, &uboVS, sizeof(uboVS));

	try {
		device.bindBufferMemory(uniform_buffer, uniform_buffer_device_memory, 0);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::DescriptorBufferInfo uniform_buffer_descriptor(
		uniform_buffer,
		0,
		sizeof(uboVS)
	);

	// Create texture samplers (currently assumes all textures are set up with the same properties)
	std::vector<std::string> texture_paths;
	texture_paths.push_back("../meshes/cube/default.png");
	texture_paths.push_back("../meshes/cube/tounge.png");

	std::vector<vk::Image> texture_images(num_textures);
	std::vector<vk::DeviceMemory> texture_image_memorys(num_textures);
	std::vector<vk::ImageView> texture_image_views(num_textures);
	std::vector<vk::Sampler> texture_samplers(num_textures);
	std::vector<vk::DescriptorImageInfo> texture_descriptors(num_textures);

	for (int k = 0; k < num_textures; k++) {
		int texture_width, texture_height, texture_comp;
		unsigned char* texture = stbi_load(texture_paths.at(k).c_str(), &texture_width, &texture_height, &texture_comp, 4);

		vk::ImageCreateInfo texture_image_create_info(
			vk::ImageCreateFlags(),
			vk::ImageType::e2D,
			vk::Format::eR8G8B8A8Unorm,
			vk::Extent3D(texture_width, texture_height, 1),
			1,
			1,
			vk::SampleCountFlagBits::e1,
			vk::ImageTiling::eLinear,
			vk::ImageUsageFlags(vk::ImageUsageFlagBits::eSampled),
			vk::SharingMode::eExclusive,
			0,
			nullptr,
			vk::ImageLayout::ePreinitialized
		);

		try {
			texture_images.at(k) = device.createImage(texture_image_create_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::MemoryRequirements texture_memory_requirements;
		try {
			texture_memory_requirements = device.getImageMemoryRequirements(texture_images.at(k));
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		uint32_t texture_memory_type_index = get_memory_type(physical_device, texture_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);

		vk::MemoryAllocateInfo texture_memory_allocate_info(
			texture_memory_requirements.size,
			texture_memory_type_index
		);


		try {
			texture_image_memorys.at(k) = device.allocateMemory(texture_memory_allocate_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		try {
			device.bindImageMemory(texture_images.at(k), texture_image_memorys.at(k), (vk::DeviceSize)0);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		// maybe need to convert image layout here?

		vk::ImageSubresource texture_image_subresource(
			vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor),
			0,
			0
		);

		vk::SubresourceLayout texture_subresource_layout;
		try {
			texture_subresource_layout = device.getImageSubresourceLayout(texture_images.at(k), texture_image_subresource);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		// copy texture to memory taking row pitch into account
		void* texture_mapped_memory;
		try {
			texture_mapped_memory = device.mapMemory(texture_image_memorys.at(k), 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		char* rowPtr = reinterpret_cast<char*>(texture_mapped_memory);
		for (int y = 0; y < texture_height; y++) {
			memcpy(rowPtr, &texture[y*texture_width * 4], texture_width * 4);
			rowPtr += texture_subresource_layout.rowPitch;
		}

		try {
			device.unmapMemory(texture_image_memorys.at(k));
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::ImageViewCreateInfo texture_image_view_create_info(
			vk::ImageViewCreateFlags(),
			texture_images.at(k),
			vk::ImageViewType::e2D,
			vk::Format::eR8G8B8A8Unorm,
			vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
			vk::ImageSubresourceRange(
				vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor),
				0,
				1,
				0,
				1
			)
		);

		try {
			texture_image_views.at(k) = device.createImageView(texture_image_view_create_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::SamplerCreateInfo texture_sampler_create_info(
			vk::SamplerCreateFlags(),
			vk::Filter::eLinear,
			vk::Filter::eLinear,
			vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			0.0f,
			VK_FALSE,
			0,
			VK_FALSE,
			vk::CompareOp::eNever,
			0.0f,
			0.0f,
			vk::BorderColor::eFloatOpaqueWhite,
			VK_FALSE
		);

		try {
			texture_samplers.at(k) = device.createSampler(texture_sampler_create_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		texture_descriptors.at(k) = vk::DescriptorImageInfo(
			texture_samplers.at(k),
			texture_image_views.at(k),
			vk::ImageLayout::eGeneral
		);
	}

	// Create descriptor set
	vk::DescriptorSetAllocateInfo descriptor_set_allocate_info(
		descriptor_pool,
		(uint32_t)descriptor_set_layouts.size(),
		descriptor_set_layouts.data()
	);

	std::vector<vk::DescriptorSet> descriptor_sets;
	try {
		descriptor_sets = device.allocateDescriptorSets(descriptor_set_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
	write_descriptor_sets.push_back( // Vertex Shader Uniform Buffer
		vk::WriteDescriptorSet(
			descriptor_sets.at(0),
			0,
			0,
			1,
			vk::DescriptorType::eUniformBuffer,
			nullptr,
			&uniform_buffer_descriptor,
			nullptr
		)
	);
	write_descriptor_sets.push_back( // Fragment Shader Texture Samplers
		vk::WriteDescriptorSet(
			descriptor_sets.at(0),
			1,
			0,
			(uint32_t)texture_descriptors.size(),
			vk::DescriptorType::eCombinedImageSampler,
			texture_descriptors.data(),
			nullptr,
			nullptr
		)
	);

	try {
		device.updateDescriptorSets(write_descriptor_sets, nullptr);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Create Vertex Buffer
	vk::BufferCreateInfo vertex_buffer_create_info(
		vk::BufferCreateFlags(),
		vk::DeviceSize(vertices.size() * sizeof(vertices.front())),
		vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer),
		vk::SharingMode::eExclusive,
		0,
		nullptr
	);

	vk::Buffer vertex_buffer;
	try {
		vertex_buffer = device.createBuffer(vertex_buffer_create_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	vk::MemoryRequirements vertex_buffer_memory_requirements;
	try {
		vertex_buffer_memory_requirements = device.getBufferMemoryRequirements(vertex_buffer);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// No staging yet, just allocate host visible memory for now.
	uint32_t vertex_buffer_memory_type_index = get_memory_type(physical_device, vertex_buffer_memory_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);

	vk::MemoryAllocateInfo vertex_buffer_allocate_info(
		vertex_buffer_memory_requirements.size,
		vertex_buffer_memory_type_index
	);

	vk::DeviceMemory vertex_buffer_device_memory;
	try {
		vertex_buffer_device_memory = device.allocateMemory(vertex_buffer_allocate_info);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	update_memory(device, vertex_buffer_device_memory, vertices.data(), vertices.size() * sizeof(vertices.at(0)));

	try {
		device.bindBufferMemory(vertex_buffer, vertex_buffer_device_memory, 0);
	}
	catch (const std::system_error& e) {
		fprintf(stderr, "Vulkan failure: %s\n", e.what());
		system("pause");
		exit(-1);
	}

	// Record command buffers
	// Prelim data
	std::vector<vk::ClearValue> clear_values;
	clear_values.push_back(
		vk::ClearValue(
			std::array<float, 4>{ 0.467f, 0.725f, 0.f, 0.f }
	)
	);
	clear_values.push_back(
		vk::ClearDepthStencilValue(
			1.0f, 0
		)
	);

	vk::ImageSubresourceRange image_subresource_range(
		vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor),
		0,
		1,
		0,
		1
	);

	// Get present queue
	vk::Queue present_queue = device.getQueue(0, 0);

	// Command Buffer Begin Info
	vk::CommandBufferBeginInfo setup_cmd_buffer_begin_info(
		vk::CommandBufferUsageFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit),
		nullptr
	);

	// Convert Depth Image from Undefined to DepthStencilAttachmentOptimal
	do {
		vk::ImageSubresourceRange depth_image_subresource_range(
			vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth),
			0,
			1,
			0,
			1
		);

		vk::ImageMemoryBarrier layout_transition_barrier(
			vk::AccessFlags(),
			vk::AccessFlags(vk::AccessFlagBits::eDepthStencilAttachmentRead |
				vk::AccessFlagBits::eDepthStencilAttachmentWrite),
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eDepthStencilAttachmentOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			depth_image,
			depth_image_subresource_range
		);

		setup_command_buffer.begin(setup_cmd_buffer_begin_info); // Start recording
		{
			setup_command_buffer.pipelineBarrier(
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTopOfPipe),
				vk::DependencyFlags(),
				0,
				0,
				layout_transition_barrier
			);
		}
		try {
			setup_command_buffer.end(); // Stop recording
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::PipelineStageFlags setup_wait_dst_stage_mask = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);

		vk::SubmitInfo setup_submit_info(
			0,
			nullptr,
			&setup_wait_dst_stage_mask,
			1,
			&setup_command_buffer,
			0,
			nullptr
		);

		vk::FenceCreateInfo setup_fence_create_info;

		vk::Fence setup_fence = device.createFence(setup_fence_create_info);
		present_queue.submit(setup_submit_info, setup_fence);

		device.waitForFences(setup_fence, VK_TRUE, UINT64_MAX);
		device.resetFences(setup_fence);
		setup_command_buffer.reset(vk::CommandBufferResetFlags());
	} while (0);

	// Convert Images from Undefined to PresentSrcKHR
	for (int k = 0; k < images.size(); k++) {
		vk::ImageMemoryBarrier layout_transition_barrier(
			vk::AccessFlags(),
			vk::AccessFlags(),
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::ePresentSrcKHR,
			0,
			0,
			images.at(k),
			image_subresource_range
		);

		setup_command_buffer.begin(setup_cmd_buffer_begin_info); // Start recording
		{
			setup_command_buffer.pipelineBarrier(
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
				vk::DependencyFlags(),
				0,
				0,
				layout_transition_barrier
			);
		}
		try {
			setup_command_buffer.end(); // Stop recording
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::PipelineStageFlags setup_wait_dst_stage_mask = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);

		vk::SubmitInfo setup_submit_info(
			0,
			nullptr,
			&setup_wait_dst_stage_mask,
			1,
			&setup_command_buffer,
			0,
			nullptr
		);

		vk::FenceCreateInfo setup_fence_create_info;

		vk::Fence setup_fence = device.createFence(setup_fence_create_info);
		present_queue.submit(setup_submit_info, setup_fence);

		device.waitForFences(setup_fence, VK_TRUE, UINT64_MAX);
		device.resetFences(setup_fence);
		setup_command_buffer.reset(vk::CommandBufferResetFlags());
	}

	// Command Buffer Begin Info
	vk::CommandBufferBeginInfo cmd_buffer_begin_info(
		vk::CommandBufferUsageFlags(),
		nullptr
	);

	// Record Command Buffer
	for (int k = 0; k < images.size(); k++) {
		vk::ImageMemoryBarrier barrier_from_present_to_clear(
			vk::AccessFlags(),
			vk::AccessFlags(vk::AccessFlagBits::eColorAttachmentWrite),
			vk::ImageLayout::ePresentSrcKHR,
			vk::ImageLayout::eColorAttachmentOptimal,
			0,
			0,
			images.at(k),
			image_subresource_range
		);

		vk::ImageMemoryBarrier barrier_from_clear_to_present(
			vk::AccessFlags(vk::AccessFlagBits::eColorAttachmentWrite),
			vk::AccessFlags(),
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			0,
			0,
			images.at(k),
			image_subresource_range
		);

		command_buffers.at(k).begin(cmd_buffer_begin_info); // Start recording
		{
			command_buffers.at(k).pipelineBarrier(
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
				vk::DependencyFlags(),
				0,
				0,
				barrier_from_present_to_clear
			);

			vk::RenderPassBeginInfo render_pass_begin_info(
				render_pass,
				framebuffers[k],
				vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(WIDTH, HEIGHT)),
				(uint32_t)clear_values.size(),
				clear_values.data()
			);

			command_buffers.at(k).beginRenderPass(&render_pass_begin_info, vk::SubpassContents::eInline);
			command_buffers.at(k).bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, descriptor_sets, nullptr);
			command_buffers.at(k).bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipelines[0]);
			command_buffers.at(k).bindVertexBuffers(0, vertex_buffer, vk::DeviceSize());
			command_buffers.at(k).draw((uint32_t)vertices.size(), 1, 0, 0);
			command_buffers.at(k).endRenderPass();
			command_buffers.at(k).pipelineBarrier(
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eTransfer),
				vk::PipelineStageFlags(vk::PipelineStageFlagBits::eBottomOfPipe),
				vk::DependencyFlags(),
				0,
				0,
				barrier_from_clear_to_present
			);
		}
		try {
			command_buffers.at(k).end(); // Stop recording
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}
	}

	// Render Loop
	quit = false;
	do {
		glm::vec3 camera_pos = glm::vec3(1.0f, 1.0f, 2.0f);
		glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 camera_direction = glm::normalize(camera_pos - camera_target);
		glm::vec3 camera_right = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), camera_direction));
		glm::vec3 camera_up = glm::cross(camera_direction, camera_right);

		// GLFW Input Handling
		glfwPollEvents();
		//TODO: Move the following functions to a GLFW key callback
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, 1);
			quit = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			uboVS.model_matrix = glm::rotate(0.0004f, glm::vec3(0.0f, 1.0f, 0.0f)) * uboVS.model_matrix;
			update_memory(device, uniform_buffer_device_memory, &uboVS, sizeof(uboVS));
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			uboVS.model_matrix = glm::rotate(-0.0004f, glm::vec3(0.0f, 1.0f, 0.0f)) * uboVS.model_matrix;
			update_memory(device, uniform_buffer_device_memory, &uboVS, sizeof(uboVS));
		}
		else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			uboVS.model_matrix = glm::rotate(-0.0004f, camera_right) * uboVS.model_matrix;
			update_memory(device, uniform_buffer_device_memory, &uboVS, sizeof(uboVS));
		}
		else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			uboVS.model_matrix = glm::rotate(0.0004f, camera_right) * uboVS.model_matrix;
			update_memory(device, uniform_buffer_device_memory, &uboVS, sizeof(uboVS));
		}

		// Get next swapchain image
		uint32_t next_image;
		try {
			next_image = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE).value;
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		vk::PipelineStageFlags wait_dst_stage_mask = vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput);

		// Submit to queue
		vk::SubmitInfo submit_info(
			1,
			&imageAvailableSemaphore,
			&wait_dst_stage_mask,
			1,
			&command_buffers[next_image],
			1,
			&renderingFinishedSemaphore
		);

		present_queue.submit(submit_info, VK_NULL_HANDLE);

		// Present image to render
		vk::PresentInfoKHR present_info(
			1,
			&renderingFinishedSemaphore,
			1,
			&swapchain,
			&next_image,
			(vk::Result *) nullptr
		);

		try {
			present_queue.presentKHR(present_info);
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}

		try {
			device.waitIdle();
		}
		catch (const std::system_error& e) {
			fprintf(stderr, "Vulkan failure: %s\n", e.what());
			system("pause");
			exit(-1);
		}
	} while (!quit);

	if (glfwWindowShouldClose(window)) {
		glfwDestroyWindow(window);
	}

	// Wait for user before closing
	system("pause");

	// Clean Up
	// TODO: FIX THIS FUNCTION
	// Destroy Callbacks (comment this out if you want to see debug info about cleanup)
#if !DEBUG_CLEANUP
	PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT =
		reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>
		(instance.getProcAddr("vkDestroyDebugReportCallbackEXT"));
	for (int k = 0; k < callbacks.size(); k++) {
		vkDestroyDebugReportCallbackEXT(instance, callbacks.at(k), nullptr);
	}
#endif

	device.destroyRenderPass(render_pass);

	device.destroyCommandPool(command_pool);

	for (int k = 0; k < images.size(); k++) {
		device.destroyFramebuffer(framebuffers.at(k));
		device.destroyImageView(imageViews.at(k));
	}

	device.destroySemaphore(imageAvailableSemaphore);
	device.destroySemaphore(renderingFinishedSemaphore);

	device.destroySwapchainKHR(swapchain);
	device.destroy();

	instance.destroySurfaceKHR(surface); //vkDestroySurfaceKHR(instance, native_surface, nullptr); // ???
	instance.destroy();

#if DEBUG_CLEANUP
	// Dummy pause to catch extra debug messages related to cleanup
	D_(system("pause"));
#endif

	return 0;
}
