#include <fstream>
#include <iostream>
#include <chrono>

#include <SDL.h>
#include <SDL_image.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VKassert(vkRes) SDL_assert(vkRes == VK_SUCCESS)

/*struct CustomVertex {
	int vData;
};*/
using CustomVertex = uint32_t;

struct PushConstants {
	glm::vec4 pos;
	glm::vec3 skyColor;
	vk::Bool32 backwall;
};

struct ImageBuffer {
	ImageBuffer(vk::Image _image, vk::ImageView _view) : image(_image), view(_view) {}

	vk::Image image;
	vk::ImageView view;
};

uint32_t getMemoryType(vk::PhysicalDeviceMemoryProperties mem_props, uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask) {
	for(uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if((typeBits & 1) == 1) {
			if((mem_props.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
				return i;
			}
		}
		typeBits >>= 1;
	}
	SDL_assert(false);
}

class SingleUseBuffer {
public:

	SingleUseBuffer(const vk::Device& device, vk::CommandPool& transientPool, vk::Queue& queue) :
		_device(device),
		_pool(transientPool),
		_queue(queue),
		_buffer(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(transientPool, vk::CommandBufferLevel::ePrimary, 1))[0]) {
		_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	}

	~SingleUseBuffer() {
		_buffer.end();

		_queue.submit(vk::SubmitInfo().setCommandBufferCount(1).setPCommandBuffers(&_buffer), vk::Fence());
		_queue.waitIdle();

		_device.freeCommandBuffers(_pool, 1, &_buffer);
	}

	vk::CommandBuffer _buffer;

private:
	const vk::Device& _device;
	vk::Queue& _queue;
	vk::CommandPool& _pool;
};

struct AllocatedBuffer {
	vk::Buffer buffer;
	vk::DeviceMemory memory;
	vk::DeviceSize size;

public:
	AllocatedBuffer(const vk::Device& device,
					vk::PhysicalDeviceMemoryProperties memProps,
					vk::DeviceSize bufSize,
					vk::BufferUsageFlags usage,
					vk::MemoryPropertyFlags properties) : size(bufSize), _device(device) {
		vk::BufferCreateInfo bufferCI(vk::BufferCreateFlags(), size, usage);
		buffer = device.createBuffer(bufferCI);

		auto bufferMemReqs = device.getBufferMemoryRequirements(buffer);
		auto bufferMemtype = getMemoryType(memProps, bufferMemReqs.memoryTypeBits, properties);
		vk::MemoryAllocateInfo bufferMemoryAI(bufferMemReqs.size, bufferMemtype);
		memory = device.allocateMemory(bufferMemoryAI);

		device.bindBufferMemory(buffer, memory, 0);
	}

	void update(vk::Device& device, const void* data, vk::DeviceSize offset = 0, vk::DeviceSize updateSize = 0) {
		if(updateSize == 0) updateSize = size;
		void* bufferMappedMemory = device.mapMemory(memory, offset, updateSize);
		memcpy(bufferMappedMemory, data, updateSize);
		device.unmapMemory(memory);
	}

	void destroy() {
		if(size == 0) return;
		_device.destroyBuffer(buffer);
		_device.freeMemory(memory);
		size = 0;
	}

private:
	const vk::Device& _device;
};

struct AllocatedImage {
	vk::Image _image;
	vk::DeviceMemory _memory;
	vk::Format _format;
	vk::ImageLayout _layout;

public:
	AllocatedImage(vk::Device device,
				   vk::PhysicalDeviceMemoryProperties memProps,
				   const vk::ImageCreateInfo& imageCI) :
		_image(device.createImage(imageCI)),
		_layout(imageCI.initialLayout),
		_format(imageCI.format) {
		auto memReqs = device.getImageMemoryRequirements(_image);
		uint32_t textureMemoryType = getMemoryType(memProps, memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		auto memoryAI = vk::MemoryAllocateInfo(memReqs.size, textureMemoryType);

		_memory = device.allocateMemory(memoryAI);

		device.bindImageMemory(_image, _memory, 0);
	}

	void transitionLayout(vk::CommandBuffer cmdBuffer, vk::ImageLayout newLayout) {
		auto memoryBarrier = vk::ImageMemoryBarrier()
			.setOldLayout(_layout).setNewLayout(newLayout)
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setImage(_image)
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		vk::PipelineStageFlags sourceStage, dstStage;
		if(_layout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
			memoryBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			dstStage = vk::PipelineStageFlagBits::eTransfer;
		} else if(_layout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
			memoryBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
			memoryBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			dstStage = vk::PipelineStageFlagBits::eFragmentShader;
		}

		cmdBuffer.pipelineBarrier(sourceStage, dstStage, vk::DependencyFlags(), nullptr, nullptr, memoryBarrier);
		_layout = newLayout;
	}

	void destroy(vk::Device& device) {
		device.destroyImage(_image);
		device.freeMemory(_memory);
	}
};

struct StagedImage {
	AllocatedBuffer _staging;
	AllocatedImage _image;
	vk::ImageView _view;

public:
	StagedImage(const vk::Device& device, vk::PhysicalDeviceMemoryProperties memProps, SDL_Surface* surface) :
		_staging(device, memProps, surface->pitch * surface->h, vk::BufferUsageFlagBits::eTransferSrc,
				 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
		_image(device, memProps,
			   vk::ImageCreateInfo(vk::ImageCreateFlags(), vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, vk::Extent3D(surface->w, surface->h, 1), 1, 1)
			   .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)),
		_view(device.createImageView(vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), _image._image, vk::ImageViewType::e2D, _image._format)
									 .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)))),
		_surface(surface) {}

	void stageImage(vk::Device device, vk::CommandPool transientPool, vk::Queue deviceQueue) {
		SingleUseBuffer sub(device, transientPool, deviceQueue);
		_image.transitionLayout(sub._buffer, vk::ImageLayout::eTransferDstOptimal);
		auto region = vk::BufferImageCopy(0, _surface->pitch / _surface->format->BytesPerPixel, _surface->h,
										  vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1))
			.setImageExtent(vk::Extent3D(_surface->w, _surface->h, 1));
		sub._buffer.copyBufferToImage(_staging.buffer, _image._image, vk::ImageLayout::eTransferDstOptimal, region);
		_image.transitionLayout(sub._buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
	}

	void destroy(vk::Device device) {
		_staging.destroy();
		device.destroyImageView(_view);
		_image.destroy(device);
	}

private:
	SDL_Surface* _surface;
};

struct StagedBuffer {
	AllocatedBuffer staging;
	AllocatedBuffer local;

public:
	StagedBuffer(const vk::Device& device, vk::PhysicalDeviceMemoryProperties memProps, vk::DeviceSize size, vk::BufferUsageFlags usage)
		: staging(device, memProps, size, vk::BufferUsageFlagBits::eTransferSrc,
				  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
		local(device, memProps, size, vk::BufferUsageFlagBits::eTransferDst | usage, vk::MemoryPropertyFlagBits::eDeviceLocal),
		_device(device) {}

	void stageBuffer(vk::CommandPool transientPool, vk::Queue deviceQueue) {
		SingleUseBuffer sub(_device, transientPool, deviceQueue);
		sub._buffer.copyBuffer(staging.buffer, local.buffer, vk::BufferCopy(0, 0, staging.size));
	}

	void lock() {
		staging.destroy();
	}

	void destroy() {
		staging.destroy();
		local.destroy();
	}

private:
	const vk::Device& _device;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objType,
	uint64_t obj,
	size_t location,
	int32_t code,
	const char* layerPrefix,
	const char* msg,
	void* userData) {

	std::cerr << "validation layer: " << msg << std::endl;

	return VK_FALSE;
}

std::vector<CustomVertex> genGrid(uint8_t w, uint8_t h) {
	std::vector<CustomVertex> res;
	res.reserve(6u * w * h);
	for(uint8_t j = 0; j < h; ++j) {
		for(uint8_t i = 0; i < w; ++i) {
			uint16_t bloc = i + (j << 8);
			res.push_back(0x0 + bloc); res.push_back(0x30000 + bloc); res.push_back(0x10000 + bloc);
			res.push_back(0x0 + bloc); res.push_back(0x20000 + bloc); res.push_back(0x30000 + bloc);
		}
	}
	return res;
}

int main(int argc, char** argv) {
	constexpr unsigned int windowW = 1280, windowH = 1024;
	vk::PresentModeKHR windowPresentMode = vk::PresentModeKHR::eFifo;

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
	IMG_Init(IMG_INIT_PNG);

	auto textureSurface = IMG_Load("textures/rock.png");

	SDL_Window* sdlWindow = SDL_CreateWindow("VulkanTest", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowW, windowH, SDL_WINDOW_VULKAN);

	unsigned int extCount;
	SDL_Vulkan_GetInstanceExtensions(sdlWindow, &extCount, NULL);

	std::vector<const char*> extensions(extCount), layers;

	SDL_Vulkan_GetInstanceExtensions(sdlWindow, &extCount, extensions.data());
#ifdef _DEBUG
	extensions.push_back("VK_EXT_debug_report");
	layers.push_back("VK_LAYER_LUNARG_standard_validation");
#endif // DEBUG

	vk::ApplicationInfo appInfo("VulkanTest", 1, "CustomEngine", 1, VK_API_VERSION_1_0);
	vk::InstanceCreateInfo instCreateInfo(vk::InstanceCreateFlags(), &appInfo,
		(uint32_t)layers.size(), layers.data(),
										  (uint32_t)extensions.size(), extensions.data());
	vk::Instance inst = vk::createInstance(instCreateInfo);

#ifdef _DEBUG
	vk::DebugReportCallbackCreateInfoEXT debugInfo(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning, debugCallback);
	auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)inst.getProcAddr("vkCreateDebugReportCallbackEXT");
	auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)inst.getProcAddr("vkDestroyDebugReportCallbackEXT");
	VkDebugReportCallbackEXT debugCB;
	VKassert(vkCreateDebugReportCallbackEXT(inst, &(VkDebugReportCallbackCreateInfoEXT)debugInfo, nullptr, &debugCB));
#endif // DEBUG

	VkSurfaceKHR cWindowKHR = nullptr;
	SDL_assert_always(SDL_Vulkan_CreateSurface(sdlWindow, inst, &cWindowKHR));
	vk::SurfaceKHR windowKHR(cWindowKHR);

	vk::PhysicalDevice gpu = inst.enumeratePhysicalDevices()[0];

	auto gpuMemProps = gpu.getMemoryProperties();

	auto queueFamilyProps = gpu.getQueueFamilyProperties();
	gpu.getSurfaceSupportKHR(0, windowKHR);

	uint32_t selectedQueueFamily = -1;
	auto queueFamilies = gpu.getQueueFamilyProperties();
	for(auto i = 0; i < queueFamilies.size(); ++i) {
		if(queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics &&
		   gpu.getSurfaceSupportKHR(i, windowKHR)) {
			selectedQueueFamily = i;
			break;
		}
	}
	SDL_assert(selectedQueueFamily != -1);

	std::vector<float> queuePriorities = {0.0f};
	vk::DeviceQueueCreateInfo queueCreateInfo(vk::DeviceQueueCreateFlags(), selectedQueueFamily, 1, queuePriorities.data());

	std::vector<const char*> deviceExtensions;
	deviceExtensions.push_back("VK_KHR_swapchain");

	auto gpuFeatures = vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true);

	auto deviceCreateInfo = vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(1)
		.setPQueueCreateInfos(&queueCreateInfo)
		.setEnabledExtensionCount((uint32_t)deviceExtensions.size())
		.setPpEnabledExtensionNames(deviceExtensions.data())
		.setPEnabledFeatures(&gpuFeatures);
	vk::Device device = gpu.createDevice(deviceCreateInfo);
	auto deviceQueue = device.getQueue(selectedQueueFamily, 0);

	vk::CommandPoolCreateInfo cmdPoolCreateInfo(vk::CommandPoolCreateFlags(), selectedQueueFamily);
	vk::CommandPool cmdPool = device.createCommandPool(cmdPoolCreateInfo);

	/* Swapchain */
	vk::Format windowFormat;
	auto windowFormats = gpu.getSurfaceFormatsKHR(windowKHR);
	if(windowFormats.size() == 1 && windowFormats[0].format == vk::Format::eUndefined) {
		windowFormat = vk::Format::eB8G8R8A8Unorm;
	} else {
		SDL_assert(windowFormats.size() >= 1);
		windowFormat = windowFormats[0].format;
	}

	auto windowSurfaceCapabilities = gpu.getSurfaceCapabilitiesKHR(windowKHR);

	vk::Extent2D swapchainExtent(windowW, windowH);
	if(windowSurfaceCapabilities.currentExtent.width != UINT32_MAX) {
		swapchainExtent = windowSurfaceCapabilities.currentExtent;
	}

	vk::SurfaceTransformFlagBitsKHR windowPreTransform = windowSurfaceCapabilities.currentTransform;
	if(windowSurfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
		windowPreTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}

	auto windowPresentModes = gpu.getSurfacePresentModesKHR(windowKHR);
	SDL_assert(std::find(windowPresentModes.begin(), windowPresentModes.end(), windowPresentMode) != windowPresentModes.end());

	auto swapchainCI = vk::SwapchainCreateInfoKHR()
		.setSurface(windowKHR)
		.setMinImageCount(2)
		.setImageFormat(windowFormat)
		.setImageExtent(swapchainExtent)
		.setImageArrayLayers(1)
		.setPresentMode(windowPresentMode)
		.setClipped(true)
		.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

	auto swapchain = device.createSwapchainKHR(swapchainCI);
	auto swapchainImages = device.getSwapchainImagesKHR(swapchain);
	std::vector<ImageBuffer> swapchainImageBuffers;
	for(const auto& swapchainImage : swapchainImages) {
		auto imageViewCI = vk::ImageViewCreateInfo()
			.setImage(swapchainImage)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(windowFormat)
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		auto imageView = device.createImageView(imageViewCI);
		swapchainImageBuffers.emplace_back(swapchainImage, imageView);
	}


	auto swapchainDepthFormat = vk::Format::eD16Unorm;
	auto swapchainDepthCI = vk::ImageCreateInfo()
		.setImageType(vk::ImageType::e2D)
		.setFormat(swapchainDepthFormat)
		.setExtent(vk::Extent3D(windowW, windowH, 1))
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment);

	auto swapchainDepthImage = device.createImage(swapchainDepthCI);

	auto swapchainDepthMemoryReqs = device.getImageMemoryRequirements(swapchainDepthImage);
	uint32_t swapchainDepthMemoryType = getMemoryType(gpuMemProps,
													  swapchainDepthMemoryReqs.memoryTypeBits,
													  vk::MemoryPropertyFlagBits::eDeviceLocal);
	auto swapchainDepthMemoryAI = vk::MemoryAllocateInfo(swapchainDepthMemoryReqs.size, swapchainDepthMemoryType);

	auto swapchainDepthMemory = device.allocateMemory(swapchainDepthMemoryAI);

	device.bindImageMemory(swapchainDepthImage, swapchainDepthMemory, 0);

	auto swapchainDepthImageViewCI = vk::ImageViewCreateInfo()
		.setImage(swapchainDepthImage)
		.setViewType(vk::ImageViewType::e2D)
		.setFormat(swapchainDepthFormat)
		.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));

	auto swapchainDepthImageView = device.createImageView(swapchainDepthImageViewCI);

	ImageBuffer swapchainDepthBuffer(swapchainDepthImage, swapchainDepthImageView);

	/* RenderPass */
	std::vector<vk::AttachmentDescription> attachements(2);
	attachements[0] = vk::AttachmentDescription()
		.setFormat(windowFormat)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	attachements[1] = vk::AttachmentDescription()
		.setFormat(swapchainDepthFormat)
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		.setStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	auto subpass = vk::SubpassDescription()
		.setColorAttachmentCount(1)
		.setPColorAttachments(&vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal))
		.setPDepthStencilAttachment(&vk::AttachmentReference(1, vk::ImageLayout::eDepthStencilAttachmentOptimal));

	auto subpassDependency = vk::SubpassDependency()
		.setSrcSubpass(VK_SUBPASS_EXTERNAL)
		.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
		.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

	vk::RenderPassCreateInfo renderPassCI(vk::RenderPassCreateFlags(), 2, attachements.data(), 1, &subpass, 1, &subpassDependency);
	auto renderPass = device.createRenderPass(renderPassCI);

	/* Framebuffers */
	std::vector<vk::ImageView> attachementViews(2);
	std::vector<vk::Framebuffer> framebuffers;

	attachementViews[1] = swapchainDepthBuffer.view;
	for(const auto& colorImage : swapchainImageBuffers) {
		attachementViews[0] = colorImage.view;
		vk::FramebufferCreateInfo framebufferCI(vk::FramebufferCreateFlags(), renderPass, 2, attachementViews.data(), windowW, windowH, 1);
		framebuffers.push_back(device.createFramebuffer(framebufferCI));
	}

	/* Samplers */
	StagedImage textureImage(device, gpuMemProps, textureSurface);
	textureImage._staging.update(device, textureSurface->pixels);
	textureImage.stageImage(device, cmdPool, deviceQueue);

	auto samplerCI = vk::SamplerCreateInfo().setMinFilter(vk::Filter::eLinear).setAnisotropyEnable(true).setMaxAnisotropy(8)
		.setBorderColor(vk::BorderColor::eIntTransparentBlack);
	auto sampler = device.createSampler(samplerCI);

	vk::DescriptorImageInfo descImageInfo(sampler, textureImage._view, textureImage._image._layout);

	/* Uniforms */
	//glm::mat4 matProjection = glm::perspective(glm::radians(40.0f), 16.0f / 9.0f, 0.1f, 100.0f);
	//glm::mat4 matView = glm::lookAt(glm::vec3(0, 0, -2), glm::vec3(0, 0, 0), glm::vec3(0, -1, 0));
	//glm::mat4 matModel = glm::mat4(1.0f);
	//glm::mat4 matMVP = matModel;
	uint32_t blocCount = 8 * 1024;
	uint32_t* blocUniform = new uint32_t[blocCount];
	for(uint32_t i = 0; i < 4096; ++i) blocUniform[i] = 0xff0000 + i % 112;
	for(uint32_t i = 4096; i < 8192; ++i) blocUniform[i] = 0x7f0000 + (3 * i) % 112;

	StagedBuffer terrainBuffer(device, gpuMemProps, 4 * blocCount, vk::BufferUsageFlagBits::eStorageBuffer);
	terrainBuffer.staging.update(device, blocUniform);
	terrainBuffer.stageBuffer(cmdPool, deviceQueue);
	vk::DescriptorBufferInfo terrainBufferInfo(terrainBuffer.local.buffer, 0, 64 * 64 * sizeof(uint32_t));

	//AllocatedBuffer uniformBuffer(device, gpuMemProps, sizeof(VertexUniform), vk::BufferUsageFlagBits::eUniformBuffer,
	//							  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	//vk::DescriptorBufferInfo descBufferInfo(uniformBuffer._buffer, 0, uniformBuffer._size);

	// Size: vec4 transform(posXbloc, posYbloc, sizeX, sizeY) + vec3 skyColor + bool backwall
	std::vector<vk::PushConstantRange> pushRanges;
	pushRanges.emplace_back(vk::ShaderStageFlagBits::eAllGraphics, 0, 32);

	std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
	layoutBindings.emplace_back(0, vk::DescriptorType::eStorageBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex);
	layoutBindings.emplace_back(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
	vk::DescriptorSetLayoutCreateInfo descSetLayoutCI(vk::DescriptorSetLayoutCreateFlags(), (uint32_t)layoutBindings.size(), layoutBindings.data());
	auto descSetLayout = device.createDescriptorSetLayout(descSetLayoutCI);

	vk::PipelineLayoutCreateInfo pipelineLayoutCI(vk::PipelineLayoutCreateFlags(), 1, &descSetLayout, (uint32_t)pushRanges.size(), pushRanges.data());
	auto pipelineLayout = device.createPipelineLayout(pipelineLayoutCI);

	std::vector<vk::DescriptorPoolSize> descPoolSizes;
	descPoolSizes.emplace_back(vk::DescriptorType::eStorageBufferDynamic, 1);
	descPoolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler, 1);

	vk::DescriptorPoolCreateInfo descPoolCI(vk::DescriptorPoolCreateFlags(), 1, (uint32_t)descPoolSizes.size(), descPoolSizes.data());
	auto descPool = device.createDescriptorPool(descPoolCI);

	vk::DescriptorSetAllocateInfo descSetAI(descPool, 1, &descSetLayout);
	auto descSet = device.allocateDescriptorSets(descSetAI)[0];

	std::vector<vk::WriteDescriptorSet> writeDescSets;
	writeDescSets.emplace_back(descSet, 0, 0, 1, vk::DescriptorType::eStorageBufferDynamic, nullptr, &terrainBufferInfo);
	writeDescSets.emplace_back(descSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descImageInfo, nullptr);

	device.updateDescriptorSets(writeDescSets, nullptr);

	/* Shaders */
	std::ifstream vertStream("shaders/terrainvert.spv", std::ios::binary);
	std::ifstream fragStream("shaders/terrainfrag.spv", std::ios::binary);
	std::vector<char> vertShaderSPV((std::istreambuf_iterator<char>(vertStream)), std::istreambuf_iterator<char>());
	std::vector<char> fragShaderSPV((std::istreambuf_iterator<char>(fragStream)), std::istreambuf_iterator<char>());
	vertStream.close();
	fragStream.close();

	std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStagesCI;
	vk::ShaderModuleCreateInfo vertShaderCI(vk::ShaderModuleCreateFlags(), vertShaderSPV.size(), (const uint32_t*)vertShaderSPV.data());
	auto vertShader = device.createShaderModule(vertShaderCI);
	pipelineShaderStagesCI.emplace_back(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vertShader, "main");

	vk::ShaderModuleCreateInfo fragShaderCI(vk::ShaderModuleCreateFlags(), fragShaderSPV.size(), (const uint32_t*)fragShaderSPV.data());
	auto fragShader = device.createShaderModule(fragShaderCI);
	pipelineShaderStagesCI.emplace_back(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fragShader, "main");

	/* Vertex buffer */
	auto vertices = genGrid(64, 64);
	//std::vector<CustomVertex> vertices{ {-1, -1,  0, 0}, {-1, 1,  0, 1}, { 1, 1,  1, 1}, {-1, -1,  0, 0}, {1,  1,  1, 1}, {1, -1,  1, 0}};
	StagedBuffer vertexBuffer(device, gpuMemProps, vertices.size() * sizeof(CustomVertex), vk::BufferUsageFlagBits::eVertexBuffer);
	vertexBuffer.staging.update(device, vertices.data());
	vertexBuffer.stageBuffer(cmdPool, deviceQueue);

	vk::VertexInputBindingDescription vertexInputBinding(0, sizeof(CustomVertex));
	std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
	vertexInputAttributes.emplace_back(0, 0, vk::Format::eR32Uint, 0);

	/* Pipeline */
	//vk::PipelineDynamicStateCreateInfo pipelineDynamicCI;
	vk::PipelineVertexInputStateCreateInfo pipelineVertexInputCI(vk::PipelineVertexInputStateCreateFlags(),
																 1, &vertexInputBinding,
																 (uint32_t)vertexInputAttributes.size(), vertexInputAttributes.data());
	vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyCI(vk::PipelineInputAssemblyStateCreateFlags(),
																	 vk::PrimitiveTopology::eTriangleList);
	auto pipelineRasterizationCI = vk::PipelineRasterizationStateCreateInfo()
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setLineWidth(1.0f);

	auto pipelineColorBlendState = vk::PipelineColorBlendAttachmentState().setColorWriteMask(vk::ColorComponentFlagBits::eR |
																							 vk::ColorComponentFlagBits::eG |
																							 vk::ColorComponentFlagBits::eB |
																							 vk::ColorComponentFlagBits::eA);
	auto pipelineColorBlendCI = vk::PipelineColorBlendStateCreateInfo()
		.setAttachmentCount(1)
		.setPAttachments(&pipelineColorBlendState)
		.setBlendConstants({{1, 1, 1, 1}});

	vk::Viewport pipelineViewport(0, 0, (float)windowW, (float)windowH, 0, 1);
	vk::Rect2D pipelineScissor(vk::Offset2D(), vk::Extent2D(windowW, windowH));
	vk::PipelineViewportStateCreateInfo pipelineViewportCI(vk::PipelineViewportStateCreateFlags(), 1, &pipelineViewport, 1, &pipelineScissor);
	vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilCI(vk::PipelineDepthStencilStateCreateFlags(), true, true, vk::CompareOp::eLessOrEqual);
	vk::PipelineMultisampleStateCreateInfo pipelineMultisampleCI;
	vk::GraphicsPipelineCreateInfo pipelineCI(vk::PipelineCreateFlags(), (uint32_t)pipelineShaderStagesCI.size(),
											  pipelineShaderStagesCI.data(),
											  &pipelineVertexInputCI,
											  &pipelineInputAssemblyCI,
											  nullptr,
											  &pipelineViewportCI,
											  &pipelineRasterizationCI,
											  &pipelineMultisampleCI,
											  &pipelineDepthStencilCI,
											  &pipelineColorBlendCI,
											  nullptr,
											  pipelineLayout,
											  renderPass, 0);
	auto pipeline = device.createGraphicsPipeline(vk::PipelineCache(), pipelineCI);

	std::vector<vk::ClearValue> clearValues;
	clearValues.emplace_back(vk::ClearColorValue(std::array<float, 4>{ 0.2f, 0.2f, 0.2f, 0.2f }));
	clearValues.emplace_back(vk::ClearDepthStencilValue(1, 0));

	auto swapchainGotImageSem = device.createSemaphore(vk::SemaphoreCreateInfo());
	auto renderPassBeginInfo = vk::RenderPassBeginInfo(renderPass,
													   vk::Framebuffer(),
													   vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(windowW, windowH)),
													   (uint32_t)clearValues.size(),
													   clearValues.data());

	vk::CommandBufferAllocateInfo cmdBufferInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
	auto cmdBuffer = device.allocateCommandBuffers(cmdBufferInfo)[0];

	auto drawFence = device.createFence(vk::FenceCreateInfo());
	vk::PipelineStageFlags queueSubmitInfoWaitDstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	auto queueSubmitInfo = vk::SubmitInfo()
		.setWaitSemaphoreCount(1)
		.setPWaitSemaphores(&swapchainGotImageSem)
		.setPWaitDstStageMask(&queueSubmitInfoWaitDstStageMask)
		.setCommandBufferCount(1);

	vk::PresentInfoKHR present(0, nullptr, 1, &swapchain, nullptr);

	SDL_Event event;
	bool done = false;
	uint32_t frameCount = 0;
	auto start = std::chrono::system_clock::now();
	while(!done) {
		while(SDL_PollEvent(&event) > 0) {
			switch(event.type) {
			case SDL_QUIT:
				done = true;
				break;
			}
		}

		uint32_t next = device.acquireNextImageKHR(swapchain, UINT64_MAX, swapchainGotImageSem, vk::Fence()).value;

		cmdBuffer.begin(vk::CommandBufferBeginInfo());
		renderPassBeginInfo.setFramebuffer(framebuffers[next]);
		cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descSet, 0u);
		cmdBuffer.bindVertexBuffers(0, vertexBuffer.local.buffer, (vk::DeviceSize)0);

		PushConstants pushConsts = {glm::vec4(-32.f, -32.f, 32.0f / windowW, 32.0f / windowH), glm::vec3(1, 1, 1), false};
		cmdBuffer.pushConstants<PushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0, pushConsts);

		cmdBuffer.draw(6 * 4096, 1, 0, 0);

		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descSet, 4 * 4096);
		pushConsts.pos.x += 64;
		cmdBuffer.pushConstants<PushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0, pushConsts);
		cmdBuffer.draw(6 * 4096, 1, 0, 0);

		cmdBuffer.endRenderPass();

		cmdBuffer.end();

		queueSubmitInfo.setPCommandBuffers(&cmdBuffer);

		//uniformBuffer.update(device, &matMVP);

		deviceQueue.submit(queueSubmitInfo, drawFence);

		present.setPImageIndices(&next);
		device.waitForFences(drawFence, true, UINT64_MAX);
		deviceQueue.presentKHR(present);

		if((std::chrono::system_clock::now() - start) > std::chrono::seconds(1)) {
			start = std::chrono::system_clock::now();
			std::cout << "FPS:" << frameCount << std::endl;
			frameCount = 0;
		}

		cmdBuffer.reset(vk::CommandBufferResetFlags());
		device.resetFences(drawFence);
		frameCount++;
	}

	device.waitIdle();

	device.destroySemaphore(swapchainGotImageSem);
	device.destroyFence(drawFence);

	device.destroyPipeline(pipeline);

	vertexBuffer.destroy();

	for(auto& framebuffer : framebuffers) device.destroyFramebuffer(framebuffer);

	device.destroyShaderModule(fragShader);
	device.destroyShaderModule(vertShader);

	device.destroyRenderPass(renderPass);

	device.destroyDescriptorPool(descPool);

	device.destroyDescriptorSetLayout(descSetLayout);
	device.destroyPipelineLayout(pipelineLayout);

	terrainBuffer.destroy();

	device.destroySampler(sampler);
	textureImage.destroy(device);

	device.destroyImageView(swapchainDepthImageView);
	device.destroyImage(swapchainDepthImage);
	device.freeMemory(swapchainDepthMemory);

	for(const auto& imageBuffer : swapchainImageBuffers) device.destroyImageView(imageBuffer.view);
	device.destroySwapchainKHR(swapchain);

	device.destroyCommandPool(cmdPool);

	device.destroy();

#ifdef _DEBUG
	vkDestroyDebugReportCallbackEXT(inst, debugCB, nullptr);
#endif // DEBUG

	inst.destroy();

	SDL_FreeSurface(textureSurface);

	SDL_DestroyWindow(sdlWindow);

	delete[] blocUniform;

	IMG_Quit();
	SDL_Quit();

	return 0;
}
