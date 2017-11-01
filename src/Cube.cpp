#include <SDL.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#define VKassert(vkRes) SDL_assert(vkRes == vk::Result::eSuccess)

struct ImageBuffer {
	ImageBuffer(vk::Image _image, vk::ImageView _view) : image(_image), view(_view) {}

	vk::Image image;
	vk::ImageView view;
};

uint32_t getMemoryType(vk::PhysicalDeviceMemoryProperties mem_props, uint32_t typeBits, vk::MemoryPropertyFlags requirements_mask) {
	// Search memtypes to find first index with those properties
	for(uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
		if((typeBits & 1) == 1) {
			// Type is available, does it match user properties?
			if((mem_props.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
				return i;
			}
		}
		typeBits >>= 1;
	}
	// No memory types matched, return failure
	SDL_assert(false);
	return -1;
}

int main(int argc, char** argv) {
	constexpr unsigned int windowW = 1280, windowH = 720;

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

	SDL_Window* sdlWindow = SDL_CreateWindow("VulkanTest", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowW, windowH, SDL_WINDOW_VULKAN);


	unsigned int extCount;
	SDL_Vulkan_GetInstanceExtensions(sdlWindow, &extCount, NULL);

	const char** extNames = new const char*[extCount];

	SDL_Vulkan_GetInstanceExtensions(sdlWindow, &extCount, extNames);

	vk::ApplicationInfo appInfo("VulkanTest", 1, "CustomEngine", 1, VK_API_VERSION_1_0);
	vk::InstanceCreateInfo instCreateInfo(vk::InstanceCreateFlags(), &appInfo, 0, nullptr, extCount, extNames);
	vk::Instance inst = vk::createInstance(instCreateInfo);

	delete[] extNames;

	VkSurfaceKHR cWindowKHR;
	SDL_assert(SDL_Vulkan_CreateSurface(sdlWindow, inst, &cWindowKHR));
	vk::SurfaceKHR windowKHR(cWindowKHR);

	vk::PhysicalDevice gpu = inst.enumeratePhysicalDevices()[0];

	auto gpuMemoryProperties = gpu.getMemoryProperties();

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

	std::vector<float> queuePriorities = { 0.0f };
	vk::DeviceQueueCreateInfo queueCreateInfo(vk::DeviceQueueCreateFlags(), selectedQueueFamily, 1, queuePriorities.data());

	auto deviceCreateInfo = vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(1)
		.setPQueueCreateInfos(&queueCreateInfo);
	vk::Device device = gpu.createDevice(deviceCreateInfo);

	vk::CommandPoolCreateInfo cmdPoolCreateInfo(vk::CommandPoolCreateFlags(), selectedQueueFamily);
	vk::CommandPool cmdPool = device.createCommandPool(cmdPoolCreateInfo);

	vk::CommandBufferAllocateInfo cmdBufferInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
	vk::CommandBuffer cmdBuffer = device.allocateCommandBuffers(cmdBufferInfo)[0];

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

	auto swapchainCI = vk::SwapchainCreateInfoKHR()
		.setSurface(windowKHR)
		.setMinImageCount(2)
		.setImageFormat(windowFormat)
		.setImageExtent(swapchainExtent)
		.setImageArrayLayers(1)
		.setPresentMode(vk::PresentModeKHR::eFifo)
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
	uint32_t swapchainDepthMemoryType = getMemoryType(gpuMemoryProperties,
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

	SDL_Event event;
	bool done = false;
	while(!done) {
		while(SDL_PollEvent(&event) > 0) {
			switch(event.type) {
			case SDL_QUIT:
				done = true;
				break;
			}
		}

	}

	device.destroyImageView(swapchainDepthImageView);
	device.destroyImage(swapchainDepthImage);
	device.freeMemory(swapchainDepthMemory);

	for(const auto& imageBuffer : swapchainImageBuffers) device.destroyImageView(imageBuffer.view);
	device.destroySwapchainKHR(swapchain);

	device.freeCommandBuffers(cmdPool, cmdBuffer);
	device.destroyCommandPool(cmdPool);

	device.destroy();
	inst.destroy();

	SDL_DestroyWindow(sdlWindow);

	return 0;
}
