// vk_compute.h - minimal Vulkan compute runner for executing RDNU HLSL on any Vulkan device.
// Used on Linux with Mesa lavapipe to run the DP4a network kernels and the NSS pass shaders
// against their goldens without AMD hardware. HLSL is compiled to SPIR-V with DXC using the
// binding shifts below, so HLSL register b#, t#, u#, s# map to bindings #, 16+#, 32+#, 48+#.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vkc
{

constexpr uint32_t kShiftB = 0, kShiftT = 16, kShiftU = 32, kShiftS = 48;

struct Buffer
{
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    VkDeviceSize   size   = 0;
};

struct Image
{
    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    uint32_t       width = 0, height = 0, texelBytes = 0;
};

struct Resource
{
    VkDescriptorType type    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const Buffer*    buffer  = nullptr;
    VkDeviceSize     offset  = 0;
    VkDeviceSize     range   = VK_WHOLE_SIZE;
    const Image*     image   = nullptr;
    VkSampler        sampler = VK_NULL_HANDLE;
};

struct Pipeline
{
    VkShaderModule        module = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};

class Context
{
public:
    bool Init(std::string& error);
    ~Context();

    Buffer CreateBuffer(VkDeviceSize size);
    void   Destroy(Buffer& b);
    Image  CreateImage(uint32_t w, uint32_t h, VkFormat format, uint32_t texelBytes);
    void   Destroy(Image& i);
    void   Upload(const Image& img, const void* texels);    // tightly packed rows
    void   Download(const Image& img, void* texels);
    VkSampler LinearClampSampler();
    VkSampler PointClampSampler();

    // Compiles HLSL (cached by name) and builds a pipeline for the given binding layout.
    bool CreatePipeline(const std::string& hlslPath, const std::string& entry, const std::string& profile,
                        const std::vector<std::string>& defines, const std::vector<std::string>& includeDirs,
                        const std::string& cacheName, const std::vector<std::pair<uint32_t, VkDescriptorType>>& layout,
                        Pipeline& out, std::string& error);
    void Destroy(Pipeline& p);

    // Binds resources (binding index -> resource), dispatches, waits.
    bool Dispatch(const Pipeline& p, const std::map<uint32_t, Resource>& resources, uint32_t gx, uint32_t gy, uint32_t gz,
                  std::string& error);

    std::string DeviceName() const { return deviceName_; }

private:
    uint32_t MemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const;
    VkCommandBuffer Begin();
    void            SubmitAndWait(VkCommandBuffer cb);

    VkInstance       instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_     = VK_NULL_HANDLE;
    VkDevice         device_   = VK_NULL_HANDLE;
    VkQueue          queue_    = VK_NULL_HANDLE;
    uint32_t         family_   = 0;
    VkCommandPool    pool_     = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkSampler        linear_   = VK_NULL_HANDLE;
    VkSampler        point_    = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};
    std::string      deviceName_;
};

// DXC location: $DXC, else "dxc" on PATH. Compiled SPIR-V is cached in $RDNU_SPV_CACHE or /tmp/rdnu_spv.
std::string DxcPath();
bool CompileHlsl(const std::string& hlslPath, const std::string& entry, const std::string& profile,
                 const std::vector<std::string>& defines, const std::vector<std::string>& includeDirs,
                 const std::string& cacheName, std::vector<uint32_t>& spirv, std::string& error);

}  // namespace vkc
