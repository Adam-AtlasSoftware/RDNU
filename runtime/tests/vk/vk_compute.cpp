// vk_compute.cpp - see vk_compute.h
#include "vk_compute.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace vkc
{
namespace
{
#define VKCHECK(expr)                                                                        \
    do                                                                                       \
    {                                                                                        \
        VkResult r_ = (expr);                                                                \
        if (r_ != VK_SUCCESS)                                                                \
        {                                                                                    \
            std::fprintf(stderr, "vulkan: %s failed (%d) at %s:%d\n", #expr, int(r_), __FILE__, __LINE__); \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)

bool ReadFile(const std::string& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

std::string Quote(const std::string& s) { return "'" + s + "'"; }
}  // namespace

std::string DxcPath()
{
    const char* e = std::getenv("DXC");
    return e && *e ? e : "dxc";
}

bool CompileHlsl(const std::string& hlslPath, const std::string& entry, const std::string& profile,
                 const std::vector<std::string>& defines, const std::vector<std::string>& includeDirs,
                 const std::string& cacheName, std::vector<uint32_t>& spirv, std::string& error)
{
    const char* c   = std::getenv("RDNU_SPV_CACHE");
    std::string dir = c && *c ? c : "/tmp/rdnu_spv";
    mkdir(dir.c_str(), 0755);
    std::string out = dir + "/" + cacheName + ".spv";
    std::string log = dir + "/" + cacheName + ".log";

    std::ostringstream cmd;
    cmd << Quote(DxcPath()) << " -spirv -fspv-target-env=vulkan1.3 -fvk-use-dx-layout -HV 2021 -enable-16bit-types"
        << " -fvk-b-shift " << kShiftB << " 0 -fvk-t-shift " << kShiftT << " 0 -fvk-u-shift " << kShiftU
        << " 0 -fvk-s-shift " << kShiftS << " 0"
        << " -T " << profile << " -E " << entry;
    for (const std::string& d : defines)
        cmd << " -D " << Quote(d);
    for (const std::string& i : includeDirs)
        cmd << " -I " << Quote(i);
    cmd << " " << Quote(hlslPath) << " -Fo " << Quote(out) << " > " << Quote(log) << " 2>&1";
    if (std::system(cmd.str().c_str()) != 0)
    {
        std::vector<char> l;
        ReadFile(log, l);
        error = "dxc failed for " + cacheName + ":\n" + std::string(l.begin(), l.end());
        return false;
    }
    std::vector<char> bytes;
    if (!ReadFile(out, bytes) || bytes.size() % 4)
        return error = "cannot read " + out, false;
    spirv.resize(bytes.size() / 4);
    std::memcpy(spirv.data(), bytes.data(), bytes.size());
    return true;
}

bool Context::Init(std::string& error)
{
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "rdnu-vk-tests";
    app.apiVersion       = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS)
        return error = "vkCreateInstance failed", false;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (!n)
        return error = "no Vulkan device", false;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());
    const char* want = std::getenv("RDNU_VK_DEVICE");
    phys_ = devs[0];
    for (VkPhysicalDevice d : devs)
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (want && std::strstr(p.deviceName, want))
            phys_ = d;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_, &props);
    deviceName_ = props.deviceName;
    if (props.apiVersion < VK_API_VERSION_1_3)
        return error = "device must support Vulkan 1.3", false;
    vkGetPhysicalDeviceMemoryProperties(phys_, &memProps_);

    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, fams.data());
    family_ = UINT32_MAX;
    for (uint32_t i = 0; i < n; ++i)
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            family_ = i;
            break;
        }
    if (family_ == UINT32_MAX)
        return error = "no compute queue", false;

    // Enable every feature the device supports (1.0-1.3): the shaders need 8/16-bit types,
    // 8/16-bit storage, integer dot product and format-less storage images.
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(phys_, &f2);
    if (!f12.shaderFloat16 || !f12.shaderInt8 || !f13.shaderIntegerDotProduct || !f11.storageBuffer16BitAccess)
        return error = "device lacks fp16 / int8 / integer dot product", false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    if (vkCreateDevice(phys_, &dci, nullptr, &device_) != VK_SUCCESS)
        return error = "vkCreateDevice failed", false;
    vkGetDeviceQueue(device_, family_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family_;
    VKCHECK(vkCreateCommandPool(device_, &pci, nullptr, &pool_));

    VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
                                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
                                    {VK_DESCRIPTOR_TYPE_SAMPLER, 256}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets       = 256;
    dpi.poolSizeCount = sizeof(sizes) / sizeof(sizes[0]);
    dpi.pPoolSizes    = sizes;
    VKCHECK(vkCreateDescriptorPool(device_, &dpi, nullptr, &descPool_));

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    VKCHECK(vkCreateSampler(device_, &sci, nullptr, &linear_));
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    VKCHECK(vkCreateSampler(device_, &sci, nullptr, &point_));
    return true;
}

Context::~Context()
{
    if (!device_)
    {
        if (instance_)
            vkDestroyInstance(instance_, nullptr);
        return;
    }
    vkDeviceWaitIdle(device_);
    vkDestroySampler(device_, linear_, nullptr);
    vkDestroySampler(device_, point_, nullptr);
    vkDestroyDescriptorPool(device_, descPool_, nullptr);
    vkDestroyCommandPool(device_, pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroyInstance(instance_, nullptr);
}

VkSampler Context::LinearClampSampler() { return linear_; }
VkSampler Context::PointClampSampler() { return point_; }

uint32_t Context::MemoryType(uint32_t bits, VkMemoryPropertyFlags flags) const
{
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memProps_.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    std::fprintf(stderr, "vulkan: no memory type\n");
    std::abort();
}

Buffer Context::CreateBuffer(VkDeviceSize size)
{
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKCHECK(vkCreateBuffer(device_, &bci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = MemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKCHECK(vkAllocateMemory(device_, &mai, nullptr, &b.memory));
    VKCHECK(vkBindBufferMemory(device_, b.buffer, b.memory, 0));
    VKCHECK(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    std::memset(b.mapped, 0, size);
    return b;
}

void Context::Destroy(Buffer& b)
{
    if (!b.buffer)
        return;
    vkDestroyBuffer(device_, b.buffer, nullptr);
    vkFreeMemory(device_, b.memory, nullptr);
    b = Buffer{};
}

Image Context::CreateImage(uint32_t w, uint32_t h, VkFormat format, uint32_t texelBytes)
{
    Image img;
    img.width = w;
    img.height = h;
    img.format = format;
    img.texelBytes = texelBytes;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = {w, h, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(device_, &ici, nullptr, &img.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, img.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = MemoryType(req.memoryTypeBits, 0);
    VKCHECK(vkAllocateMemory(device_, &mai, nullptr, &img.memory));
    VKCHECK(vkBindImageMemory(device_, img.image, img.memory, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = img.image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VKCHECK(vkCreateImageView(device_, &vci, nullptr, &img.view));

    VkCommandBuffer cb = Begin();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
    b.image            = img.image;
    b.subresourceRange = vci.subresourceRange;
    b.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    SubmitAndWait(cb);
    return img;
}

void Context::Destroy(Image& i)
{
    if (!i.image)
        return;
    vkDestroyImageView(device_, i.view, nullptr);
    vkDestroyImage(device_, i.image, nullptr);
    vkFreeMemory(device_, i.memory, nullptr);
    i = Image{};
}

void Context::Upload(const Image& img, const void* texels)
{
    VkDeviceSize bytes = VkDeviceSize(img.width) * img.height * img.texelBytes;
    Buffer staging     = CreateBuffer(bytes);
    std::memcpy(staging.mapped, texels, bytes);
    VkCommandBuffer cb = Begin();
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent      = {img.width, img.height, 1};
    vkCmdCopyBufferToImage(cb, staging.buffer, img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    SubmitAndWait(cb);
    Destroy(staging);
}

void Context::Download(const Image& img, void* texels)
{
    VkDeviceSize bytes = VkDeviceSize(img.width) * img.height * img.texelBytes;
    Buffer staging     = CreateBuffer(bytes);
    VkCommandBuffer cb = Begin();
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent      = {img.width, img.height, 1};
    vkCmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &r);
    SubmitAndWait(cb);
    std::memcpy(texels, staging.mapped, bytes);
    Destroy(staging);
}

VkCommandBuffer Context::Begin()
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKCHECK(vkAllocateCommandBuffers(device_, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

void Context::SubmitAndWait(VkCommandBuffer cb)
{
    VKCHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VKCHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VKCHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, pool_, 1, &cb);
}

bool Context::CreatePipeline(const std::string& hlslPath, const std::string& entry, const std::string& profile,
                             const std::vector<std::string>& defines, const std::vector<std::string>& includeDirs,
                             const std::string& cacheName, const std::vector<std::pair<uint32_t, VkDescriptorType>>& layout,
                             Pipeline& out, std::string& error)
{
    std::vector<uint32_t> spirv;
    if (!CompileHlsl(hlslPath, entry, profile, defines, includeDirs, cacheName, spirv, error))
        return false;
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * 4;
    smci.pCode    = spirv.data();
    VKCHECK(vkCreateShaderModule(device_, &smci, nullptr, &out.module));

    out.bindings.clear();
    for (auto& b : layout)
    {
        VkDescriptorSetLayoutBinding lb{};
        lb.binding         = b.first;
        lb.descriptorType  = b.second;
        lb.descriptorCount = 1;
        lb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        out.bindings.push_back(lb);
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = uint32_t(out.bindings.size());
    dl.pBindings    = out.bindings.data();
    VKCHECK(vkCreateDescriptorSetLayout(device_, &dl, nullptr, &out.setLayout));
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts    = &out.setLayout;
    VKCHECK(vkCreatePipelineLayout(device_, &pl, nullptr, &out.layout));
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = out.module;
    cp.stage.pName  = entry.c_str();
    cp.layout       = out.layout;
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp, nullptr, &out.pipeline) != VK_SUCCESS)
        return error = "vkCreateComputePipelines failed for " + cacheName, false;
    return true;
}

void Context::Destroy(Pipeline& p)
{
    if (!p.pipeline)
        return;
    vkDestroyPipeline(device_, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device_, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device_, p.setLayout, nullptr);
    vkDestroyShaderModule(device_, p.module, nullptr);
    p = Pipeline{};
}

bool Context::Dispatch(const Pipeline& p, const std::map<uint32_t, Resource>& resources, uint32_t gx, uint32_t gy,
                       uint32_t gz, std::string& error)
{
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &p.setLayout;
    VkDescriptorSet set;
    VKCHECK(vkAllocateDescriptorSets(device_, &ai, &set));

    std::vector<VkWriteDescriptorSet>   writes;
    std::vector<VkDescriptorBufferInfo> bufInfos(p.bindings.size());
    std::vector<VkDescriptorImageInfo>  imgInfos(p.bindings.size());
    for (size_t i = 0; i < p.bindings.size(); ++i)
    {
        const VkDescriptorSetLayoutBinding& lb = p.bindings[i];
        auto it = resources.find(lb.binding);
        if (it == resources.end())
            return error = "binding " + std::to_string(lb.binding) + " not provided", false;
        const Resource& r = it->second;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = set;
        w.dstBinding      = lb.binding;
        w.descriptorCount = 1;
        w.descriptorType  = lb.descriptorType;
        switch (lb.descriptorType)
        {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            bufInfos[i] = {r.buffer->buffer, r.offset, r.range};
            w.pBufferInfo = &bufInfos[i];
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            imgInfos[i] = {VK_NULL_HANDLE, r.image->view, VK_IMAGE_LAYOUT_GENERAL};
            w.pImageInfo = &imgInfos[i];
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            imgInfos[i] = {VK_NULL_HANDLE, r.image->view, VK_IMAGE_LAYOUT_GENERAL};
            w.pImageInfo = &imgInfos[i];
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            imgInfos[i] = {r.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            w.pImageInfo = &imgInfos[i];
            break;
        default:
            return error = "unsupported descriptor type", false;
        }
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(device_, uint32_t(writes.size()), writes.data(), 0, nullptr);

    VkCommandBuffer cb = Begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cb, gx, gy, gz);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
    SubmitAndWait(cb);
    vkFreeDescriptorSets(device_, descPool_, 1, &set);
    return true;
}

}  // namespace vkc
