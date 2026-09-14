#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <fstream>
#include <optional>
#include <cmath>
#include <chrono>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_main.h>

#define M_PI 3.14159265358979323846

// std140 layout, must match the `UniformBufferObject` block in
// shaders/vert.vert and shaders/frag.frag (3 x mat4 + 1 x vec4 = 208 bytes).
// Columns are stored first, i.e. exactly like a GLSL mat4.
struct UniformBufferObject {
    float model[16];
    float view[16];
    float proj[16];
    float params[4];   // x = time (s), y = frame index, z = width, w = height
};

// --------------------------------------------------------------- mat4 utils --
// Column-major 4x4 helpers so the matrices below are built the same way GLSL
// reads them: m[col * 4 + row].
static void mat4Identity(float* m) {
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4Mul(const float* a, const float* b, float* out) {   // out = a * b
    float tmp[16];
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a[k * 4 + row] * b[col * 4 + k];
            tmp[col * 4 + row] = sum;
        }
    }
    memcpy(out, tmp, sizeof(tmp));
}

static void mat4RotX(float angle, float* m) {
    mat4Identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[5] = c;  m[6] = s;
    m[9] = -s; m[10] = c;
}

static void mat4RotY(float angle, float* m) {
    mat4Identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0] = c;  m[2] = -s;
    m[8] = s;  m[10] = c;
}

static void mat4RotZ(float angle, float* m) {
    mat4Identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0] = c;  m[1] = s;
    m[4] = -s; m[5] = c;
}

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Right-handed look-at, GLM convention (camera looks down its own -Z).
static void mat4LookAt(const float* eye, const float* center, const float* up, float* m) {
    float f[3] = { center[0] - eye[0], center[1] - eye[1], center[2] - eye[2] };
    float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    f[0] /= fl; f[1] /= fl; f[2] /= fl;

    float s[3] = { f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0] };
    float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    s[0] /= sl; s[1] /= sl; s[2] /= sl;

    float u[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };

    m[0] = s[0]; m[1] = u[0]; m[2] = -f[0]; m[3] = 0.0f;
    m[4] = s[1]; m[5] = u[1]; m[6] = -f[1]; m[7] = 0.0f;
    m[8] = s[2]; m[9] = u[2]; m[10] = -f[2]; m[11] = 0.0f;
    m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    m[14] = (f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
    m[15] = 1.0f;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {WIDTH, HEIGHT};
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

void generateTorus(float majorRadius, float minorRadius, int majorSegments, int minorSegments, std::vector<float>& vertices, std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    for (int i = 0; i <= majorSegments; ++i) {
        float theta = 2.0f * M_PI * i / majorSegments;
        float cosTheta = cosf(theta);
        float sinTheta = sinf(theta);

        for (int j = 0; j <= minorSegments; ++j) {
            float phi = 2.0f * M_PI * j / minorSegments;
            float cosPhi = cosf(phi);
            float sinPhi = sinf(phi);

            float x = (majorRadius + minorRadius * cosPhi) * cosTheta;
            float y = minorRadius * sinPhi;
            float z = (majorRadius + minorRadius * cosPhi) * sinTheta;

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            float nx = cosPhi * cosTheta;
            float ny = sinPhi;
            float nz = cosPhi * sinTheta;
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    for (int i = 0; i < majorSegments; ++i) {
        for (int j = 0; j < minorSegments; ++j) {
            int first = i * (minorSegments + 1) + j;
            int second = first + minorSegments + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
}

class TorusApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    SDL_Window* window = nullptr;
    uint32_t applicationExtensionCount = 0;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout backgroundPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout torusPipelineLayout = VK_NULL_HANDLE;
    VkPipeline backgroundPipeline = VK_NULL_HANDLE;
    VkPipeline torusPipeline = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    VkBuffer uniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;

    // --------------------------------------------------- camera / input state --
    // Spectator "fly" camera, AAA-style: hold RMB to look, WASD to move,
    // Space/Ctrl to rise/sink, Shift to sprint, wheel to change speed.
    // Velocity is smoothed exponentially so motion feels weighted.
    float camPos[3]  = { 0.0f, 1.6f, 13.4f };
    float camYaw     = 0.0f;      // around world +Y (0 faces -Z)
    float camPitch   = -0.1409f;  // reproduces the old look-at framing
    float camVel[3]  = { 0.0f, 0.0f, 0.0f };
    float moveSpeed  = 6.0f;      // units/s, wheel-adjustable
    bool  looking    = false;     // RMB held -> relative mouse mode
    bool  draggingTorus = false;  // LMB held -> rotate the torus

    // ------------------------------------------------------ torus input state --
    float spinAngle  = 0.0f;      // showcase spin, advanced every frame
    bool  autoSpin   = true;      // T toggles
    float torusYaw   = 0.0f;      // user orientation offsets (LMB drag/arrows)
    float torusPitch = 0.0f;

    // tuning constants
    static constexpr float kLookSensitivity  = 0.0024f; // rad per pixel
    static constexpr float kTorusDragRate    = 0.0050f; // rad per pixel
    static constexpr float kTorusKeyRate     = 1.4f;    // rad per second
    static constexpr float kSpinRate         = 0.15f;   // rad per second
    static constexpr float kMoveAccel        = 10.0f;   // 1/s, velocity lerp rate
    static constexpr float kSprintMultiplier = 4.0f;
    static constexpr float kMinSpeed         = 0.5f;
    static constexpr float kMaxSpeed         = 60.0f;
    static constexpr float kMaxPitch         = 1.5533f; // ~89 degrees

    void initWindow() {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            throw std::runtime_error(std::string("Failed to initialize SDL3: ") + SDL_GetError());
        }

        window = SDL_CreateWindow(
            "Torus",
            WIDTH, HEIGHT,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
        );
        if (!window) {
            throw std::runtime_error(std::string("Failed to create SDL window: ") + SDL_GetError());
        }
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createCommandPool();
        createTorusMesh();
        createUniformBuffer();
        createDescriptorPool();
        createDescriptorSet();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Torus";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // SDL3 knows exactly which extensions the platform requires
        const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&applicationExtensionCount);
        if (!sdlExtensions) {
            throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());
        }

        std::vector<const char*> extensionNames(sdlExtensions, sdlExtensions + applicationExtensionCount);
        // Enable the API dump layer when running with validation enabled
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionNames.size());
        createInfo.ppEnabledExtensionNames = extensionNames.data();
        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create instance");
        }
    }

    void createSurface() {
        if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
            throw std::runtime_error(std::string("Failed to create window surface: ") + SDL_GetError());
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU");
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.fillModeNonSolid = VK_TRUE;
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views");
            }
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // Depth attachment: the torus pass is rasterised against it, which is
        // what gives the path tracer an exact, depth-tested primary hit.
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        // The colour attachment is written before the depth attachment is
        // first read, make that dependency explicit.
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkAttachmentDescription attachments[] = { colorAttachment, depthAttachment };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = attachments;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                       (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        throw std::runtime_error("Failed to find a supported depth/stencil format");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    void createImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                     VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = swapChainExtent.width;
        imageInfo.extent.height = swapChainExtent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate image memory");
        }
        vkBindImageMemory(device, image, memory, 0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view");
        }
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();
        createImage(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, depthImage, depthImageMemory, depthImageView);
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        // The fragment shader needs the matrices too: it unprojects the very
        // same proj*view the rasteriser uses to build its camera rays.
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }
    }

    void createGraphicsPipeline() {
        // Single vertex + fragment module pair serves both pipelines; mode comes from push constants
        auto vertShaderCode = readFile("shaders/vert.vert.spv");
        auto fragShaderCode = readFile("shaders/frag.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertStageInfo{};
        vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStageInfo.module = vertShaderModule;
        vertStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragStageInfo{};
        fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStageInfo.module = fragShaderModule;
        fragStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertStageInfo, fragStageInfo };

        // Both pipelines share the same vertex input description (background draw simply
        // does not read the attributes, but the descriptors must still be declared)
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = 6 * sizeof(float);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributeDescriptions[2]{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = 0;
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = 3 * sizeof(float);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 2;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;

        VkPipelineInputAssemblyStateCreateInfo bgInputAssembly{};
        bgInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        bgInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo bgViewportState{};
        bgViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        bgViewportState.viewportCount = 1;
        bgViewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo bgRasterizer{};
        bgRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        bgRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        bgRasterizer.lineWidth = 1.0f;
        bgRasterizer.cullMode = VK_CULL_MODE_NONE;

        VkPipelineMultisampleStateCreateInfo bgMultisampling{};
        bgMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        bgMultisampling.sampleShadingEnable = VK_FALSE;
        bgMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState bgColorBlendAttachment{};
        bgColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        bgColorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo bgColorBlending{};
        bgColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        bgColorBlending.logicOpEnable = VK_FALSE;
        bgColorBlending.attachmentCount = 1;
        bgColorBlending.pAttachments = &bgColorBlendAttachment;

        // The background pass paints the whole frame, so it must not touch the
        // depth buffer that the torus pass is about to be tested against.
        VkPipelineDepthStencilStateCreateInfo bgDepthStencil{};
        bgDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        bgDepthStencil.depthTestEnable = VK_FALSE;
        bgDepthStencil.depthWriteEnable = VK_FALSE;
        bgDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        bgDepthStencil.depthBoundsTestEnable = VK_FALSE;
        bgDepthStencil.stencilTestEnable = VK_FALSE;

        std::vector<VkDynamicState> bgDynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo bgDynamicState{};
        bgDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        bgDynamicState.dynamicStateCount = static_cast<uint32_t>(bgDynamicStates.size());
        bgDynamicState.pDynamicStates = bgDynamicStates.data();

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(int);

        VkPipelineLayoutCreateInfo bgPipelineLayoutInfo{};
        bgPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        bgPipelineLayoutInfo.setLayoutCount = 1;
        bgPipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        bgPipelineLayoutInfo.pushConstantRangeCount = 1;
        bgPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &bgPipelineLayoutInfo, nullptr, &backgroundPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create background pipeline layout");
        }

        VkGraphicsPipelineCreateInfo bgPipelineInfo{};
        bgPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        bgPipelineInfo.stageCount = 2;
        bgPipelineInfo.pStages = shaderStages;
        bgPipelineInfo.pVertexInputState = &vertexInputInfo;
        bgPipelineInfo.pInputAssemblyState = &bgInputAssembly;
        bgPipelineInfo.pViewportState = &bgViewportState;
        bgPipelineInfo.pRasterizationState = &bgRasterizer;
        bgPipelineInfo.pMultisampleState = &bgMultisampling;
        bgPipelineInfo.pColorBlendState = &bgColorBlending;
        bgPipelineInfo.pDepthStencilState = &bgDepthStencil;
        bgPipelineInfo.pDynamicState = &bgDynamicState;
        bgPipelineInfo.layout = backgroundPipelineLayout;
        bgPipelineInfo.renderPass = renderPass;
        bgPipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &bgPipelineInfo, nullptr, &backgroundPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create background pipeline");
        }

        // Create torus pipeline (reuses the same shader modules and vertex input state)
        VkPipelineInputAssemblyStateCreateInfo torusInputAssembly{};
        torusInputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        torusInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo torusViewportState{};
        torusViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        torusViewportState.viewportCount = 1;
        torusViewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo torusRasterizer{};
        torusRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        // Filled, not wireframe: this pass supplies the depth-tested primary
        // hit for the path tracer.  Culling stays off, the depth test resolves
        // visibility, so the winding order of generateTorus() cannot bite us.
        torusRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        torusRasterizer.lineWidth = 1.0f;
        torusRasterizer.cullMode = VK_CULL_MODE_NONE;
        torusRasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo torusMultisampling{};
        torusMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        torusMultisampling.sampleShadingEnable = VK_FALSE;
        torusMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState torusColorBlendAttachment{};
        torusColorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        torusColorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo torusColorBlending{};
        torusColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        torusColorBlending.logicOpEnable = VK_FALSE;
        torusColorBlending.attachmentCount = 1;
        torusColorBlending.pAttachments = &torusColorBlendAttachment;

        VkPipelineDepthStencilStateCreateInfo torusDepthStencil{};
        torusDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        torusDepthStencil.depthTestEnable = VK_TRUE;
        torusDepthStencil.depthWriteEnable = VK_TRUE;
        torusDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        torusDepthStencil.depthBoundsTestEnable = VK_FALSE;
        torusDepthStencil.stencilTestEnable = VK_FALSE;

        std::vector<VkDynamicState> torusDynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo torusDynamicState{};
        torusDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        torusDynamicState.dynamicStateCount = static_cast<uint32_t>(torusDynamicStates.size());
        torusDynamicState.pDynamicStates = torusDynamicStates.data();

        VkPipelineLayoutCreateInfo torusPipelineLayoutInfo{};
        torusPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        torusPipelineLayoutInfo.setLayoutCount = 1;
        torusPipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        torusPipelineLayoutInfo.pushConstantRangeCount = 1;
        torusPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &torusPipelineLayoutInfo, nullptr, &torusPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create torus pipeline layout");
        }

        VkGraphicsPipelineCreateInfo torusPipelineInfo{};
        torusPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        torusPipelineInfo.stageCount = 2;
        torusPipelineInfo.pStages = shaderStages;
        torusPipelineInfo.pVertexInputState = &vertexInputInfo;
        torusPipelineInfo.pInputAssemblyState = &torusInputAssembly;
        torusPipelineInfo.pViewportState = &torusViewportState;
        torusPipelineInfo.pRasterizationState = &torusRasterizer;
        torusPipelineInfo.pMultisampleState = &torusMultisampling;
        torusPipelineInfo.pColorBlendState = &torusColorBlending;
        torusPipelineInfo.pDepthStencilState = &torusDepthStencil;
        torusPipelineInfo.pDynamicState = &torusDynamicState;
        torusPipelineInfo.layout = torusPipelineLayout;
        torusPipelineInfo.renderPass = renderPass;
        torusPipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &torusPipelineInfo, nullptr, &torusPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create torus pipeline");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = { swapChainImageViews[i], depthImageView };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 2;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }
    }

    void createTorusMesh() {
        std::vector<float> vertices;
        std::vector<uint32_t> indices;
        // 128 x 64 keeps the rasterised silhouette (which the fragment shader
        // uses as its primary hit) indistinguishable from the analytic torus
        // the shader intersects for every secondary ray.
        // NOTE: major/minor radius must stay in sync with TORUS_R / TORUS_r
        //       in shaders/frag.frag.
        generateTorus(3.0f, 1.0f, 128, 64, vertices, indices);
        indexCount = static_cast<uint32_t>(indices.size());

        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(float);
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, vertexBufferSize, 0, &data);
        memcpy(data, vertices.data(), vertexBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, vertexBufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        VkDeviceSize indexBufferSize = indices.size() * sizeof(uint32_t);
        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        vkMapMemory(device, stagingBufferMemory, 0, indexBufferSize, 0, &data);
        memcpy(data, indices.data(), indexBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
        copyBuffer(stagingBuffer, indexBuffer, indexBufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffer() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffer, uniformBufferMemory);
    }

    void updateUniformBuffer() {
        static auto startTime = std::chrono::high_resolution_clock::now();
        static uint32_t frameIndex = 0;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};

        // ---- model: tip the ring towards the camera, showcase-spin it, then
        //      apply the user's manual rotations (LMB drag / arrow keys).
        //      Kept orthonormal on purpose: frag.frag uses
        //      transpose(mat3(model)) as the world->object transform for the
        //      traced/marched rays, so the model must stay a pure rotation.
        float tilt = 0.62f + 0.08f * sinf(time * 0.23f);
        float rotX[16], rotY[16];
        mat4RotY(spinAngle + torusYaw, rotY);
        mat4RotX(tilt + torusPitch, rotX);
        mat4Mul(rotY, rotX, ubo.model);
        // the torus floats above the studio floor so its penumbra detaches
        ubo.model[12] = 0.0f;
        ubo.model[13] = 1.30f;
        ubo.model[14] = 0.0f;

        // ---- view: free-fly camera (open-world AAA controls).  The shader
        //      recovers the eye position from inverse(view), so any rigid
        //      view matrix works -----------------------------------------
        float cp = cosf(camPitch), sp = sinf(camPitch);
        float cy = cosf(camYaw),   sy = sinf(camYaw);
        const float eye[3]    = { camPos[0], camPos[1], camPos[2] };
        const float center[3] = { camPos[0] + sy * cp,
                                  camPos[1] + sp,
                                  camPos[2] - cy * cp };
        const float up[3]     = { 0.0f, 1.0f, 0.0f };
        mat4LookAt(eye, center, up, ubo.view);

        // ---- projection: Vulkan conventions -------------------------------
        //      y is flipped (framebuffer y grows downwards) and depth is
        //      mapped to [0,1] instead of OpenGL's [-1,1].  frag.frag
        //      unprojects with the very same matrix, so the traced image and
        //      the rasterised torus can never drift apart.
        float aspect = swapChainExtent.width / (float)swapChainExtent.height;
        float fov = 45.0f * (float)M_PI / 180.0f;
        float zNear = 0.1f;
        float zFar = 200.0f;

        float f = 1.0f / tanf(fov / 2.0f);
        for (int i = 0; i < 16; ++i) ubo.proj[i] = 0.0f;
        ubo.proj[0]  = f / aspect;
        ubo.proj[5]  = -f;
        ubo.proj[10] = zFar / (zNear - zFar);
        ubo.proj[11] = -1.0f;
        ubo.proj[14] = (zFar * zNear) / (zNear - zFar);

        // ---- params: time, frame index, resolution ------------------------
        ubo.params[0] = time;
        ubo.params[1] = (float)frameIndex++;
        ubo.params[2] = (float)swapChainExtent.width;
        ubo.params[3] = (float)swapChainExtent.height;

        void* data;
        vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBufferMemory);
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(1);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = static_cast<uint32_t>(1);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }
    }

    void createDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }

    void createCommandBuffers() {
        commandBuffers.resize(swapChainFramebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffers");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("Failed to begin recording command buffer");
            }

            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            VkClearValue clearValues[2];
            clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            clearValues[1].depthStencil = {1.0f, 0};
            renderPassInfo.clearValueCount = 2;
            renderPassInfo.pClearValues = clearValues;

            vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            // Draw background gradient first (mode 0: full-screen triangle, shared shader modules)
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipeline);
            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            VkBuffer vertexBuffers[] = {vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(swapChainExtent.width);
            viewport.height = static_cast<float>(swapChainExtent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

            int mode = 0;
            vkCmdPushConstants(commandBuffers[i], backgroundPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
            vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

            // Draw torus on top (mode 1)
            vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, torusPipeline);
            vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, torusPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            mode = 1;
            vkCmdPushConstants(commandBuffers[i], torusPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
            vkCmdDrawIndexed(commandBuffers[i], indexCount, 1, 0, 0, 0);

            vkCmdEndRenderPass(commandBuffers[i]);
            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to record command buffer");
            }
        }
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create synchronization objects");
        }
    }

    // ------------------------------------------------- open-world AAA controls --
    void setMouseCapture(bool enable) {
        if (enable == looking) return;
        SDL_SetWindowRelativeMouseMode(window, enable);
        looking = enable;
        if (!enable) draggingTorus = false;
    }

    void resetTorus() {
        torusYaw = 0.0f;
        torusPitch = 0.0f;
        spinAngle = 0.0f;
        autoSpin = true;
    }

    void resetCamera() {
        camPos[0] = 0.0f; camPos[1] = 1.6f; camPos[2] = 13.4f;
        camYaw = 0.0f;
        camPitch = -0.1409f;
        camVel[0] = camVel[1] = camVel[2] = 0.0f;
    }

    void processEvent(const SDL_Event& event, bool& shouldClose) {
        switch (event.type) {
        case SDL_EVENT_KEY_DOWN: {
            if (event.key.repeat) break;
            switch (event.key.key) {
            case SDLK_ESCAPE:
                // ESC releases the mouse; when already free it quits
                if (looking) setMouseCapture(false);
                else         shouldClose = true;
                break;
            case SDLK_T: autoSpin = !autoSpin; break;   // showcase spin on/off
            case SDLK_R: resetTorus(); break;           // reset torus orientation
            case SDLK_F: resetCamera(); break;          // reset camera to default
            default: break;
            }
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event.button.button == SDL_BUTTON_RIGHT)     setMouseCapture(true);
            else if (event.button.button == SDL_BUTTON_LEFT) draggingTorus = true;
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.button == SDL_BUTTON_RIGHT)     setMouseCapture(false);
            else if (event.button.button == SDL_BUTTON_LEFT) draggingTorus = false;
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (looking) {
                camYaw   += event.motion.xrel * kLookSensitivity;
                camPitch -= event.motion.yrel * kLookSensitivity;
                camPitch  = clampf(camPitch, -kMaxPitch, kMaxPitch);
            }
            if (draggingTorus) {
                torusYaw   += event.motion.xrel * kTorusDragRate;
                torusPitch += event.motion.yrel * kTorusDragRate;
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            moveSpeed = clampf(moveSpeed * powf(1.15f, event.wheel.y),
                               kMinSpeed, kMaxSpeed);
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            setMouseCapture(false);
            draggingTorus = false;
            break;
        default:
            break;
        }
    }

    void mainLoop() {
        bool shouldClose = false;
        uint64_t lastFpsTime = SDL_GetTicks();
        uint32_t frames = 0;
        uint64_t lastTime = SDL_GetPerformanceCounter();
        SDL_SetWindowTitle(window, "Torus | hold RMB to look");
        while (!shouldClose) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) {
                    shouldClose = true;
                } else {
                    processEvent(event, shouldClose);
                }
            }

            // frame time for framerate-independent input integration
            uint64_t now = SDL_GetPerformanceCounter();
            float dt = (float)((double)(now - lastTime) /
                               (double)SDL_GetPerformanceFrequency());
            lastTime = now;
            if (dt > 0.1f) dt = 0.1f;   // clamp hitches (alt-tab, breakpoints)

            update(dt);
            drawFrame();
            frames++;
            uint64_t ticks = SDL_GetTicks();
            if (ticks - lastFpsTime >= 500) {
                char title[160];
                double fps = frames / ((ticks - lastFpsTime) / 1000.0);
                snprintf(title, sizeof(title),
                         "Torus | FPS: %.0f | speed %.1f u/s | %s",
                         fps, moveSpeed,
                         looking ? "RMB look - WASD fly - LMB drag torus - ESC release"
                                 : "hold RMB to look");
                SDL_SetWindowTitle(window, title);
                frames = 0;
                lastFpsTime = ticks;
            }
        }
        vkDeviceWaitIdle(device);
    }

    // Per-frame input integration, framerate independent via dt.
    void update(float dt) {
        const bool* kb = SDL_GetKeyboardState(nullptr);

        // ---- camera: build the wish velocity from held keys ---------------
        bool sprint = kb[SDL_SCANCODE_LSHIFT] || kb[SDL_SCANCODE_RSHIFT];
        float stickZ = 0.0f, stickX = 0.0f, stickY = 0.0f;
        if (kb[SDL_SCANCODE_W])     stickZ += 1.0f;   // forward
        if (kb[SDL_SCANCODE_S])     stickZ -= 1.0f;   // back
        if (kb[SDL_SCANCODE_D])     stickX += 1.0f;   // strafe right
        if (kb[SDL_SCANCODE_A])     stickX -= 1.0f;   // strafe left
        if (kb[SDL_SCANCODE_SPACE]) stickY += 1.0f;   // rise
        if (kb[SDL_SCANCODE_E])     stickY += 1.0f;   // rise (alt)
        if (kb[SDL_SCANCODE_LCTRL] || kb[SDL_SCANCODE_RCTRL] ||
            kb[SDL_SCANCODE_Q])     stickY -= 1.0f;   // sink

        float cp = cosf(camPitch), sp = sinf(camPitch);
        float cy = cosf(camYaw),   sy = sinf(camYaw);
        float fwd[3]   = { sy * cp, sp, -cy * cp };   // camera look direction
        float right[3] = { cy, 0.0f, sy };            // world-space right

        float wish[3];
        for (int i = 0; i < 3; ++i) wish[i] = fwd[i] * stickZ + right[i] * stickX;
        wish[1] += stickY;
        // normalise so diagonal/vertical+horizontal combos aren't faster
        float wl = sqrtf(wish[0] * wish[0] + wish[1] * wish[1] + wish[2] * wish[2]);
        float speed = moveSpeed * (sprint ? kSprintMultiplier : 1.0f);
        if (wl > 1e-4f) {
            for (int i = 0; i < 3; ++i) wish[i] = wish[i] / wl * speed;
        }

        // exponential smoothing towards the wish velocity (AAA feel:
        // snappy but never instant, and glides to a stop)
        float t = 1.0f - expf(-kMoveAccel * dt);
        for (int i = 0; i < 3; ++i) {
            camVel[i] += (wish[i] - camVel[i]) * t;
            camPos[i] += camVel[i] * dt;
        }

        // ---- torus: keyboard rotation + showcase spin ---------------------
        float keyYaw = 0.0f, keyPitch = 0.0f;
        if (kb[SDL_SCANCODE_LEFT])  keyYaw   -= 1.0f;
        if (kb[SDL_SCANCODE_RIGHT]) keyYaw   += 1.0f;
        if (kb[SDL_SCANCODE_UP])    keyPitch -= 1.0f;
        if (kb[SDL_SCANCODE_DOWN])  keyPitch += 1.0f;
        torusYaw   += keyYaw   * kTorusKeyRate * dt;
        torusPitch += keyPitch * kTorusKeyRate * dt;
        if (autoSpin) spinAngle += kSpinRate * dt;
    }

    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        updateUniformBuffer();

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);
    }

    void cleanup() {
        vkDeviceWaitIdle(device);

        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        vkDestroyPipeline(device, torusPipeline, nullptr);
        vkDestroyPipelineLayout(device, torusPipelineLayout, nullptr);
        vkDestroyPipeline(device, backgroundPipeline, nullptr);
        vkDestroyPipelineLayout(device, backgroundPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vkFreeMemory(device, uniformBufferMemory, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroySwapchainKHR(device, swapChain, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module");
        }

        return shaderModule;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate buffer memory");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }
};

int main(int, char**) {
    TorusApp app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
