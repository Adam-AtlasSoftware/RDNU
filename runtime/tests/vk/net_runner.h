// net_runner.h - records the production network plan (engine core + INT8 kernels) on a vkc::Context.
#pragma once

#include "../../src/engine/rdnu_engine_core.h"
#include "vk_compute.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct NetRunner
{
    vkc::Context&         vk;
    const rdnu::Manifest& manifest;
    rdnu::Plan&           plan;
    std::string           shaderDir;
    vkc::Buffer           weights, arena, input, kpn, constants;
    vkc::Image            temporal;
    std::map<std::string, vkc::Pipeline> pipelines;

    // temporal: RGBA8_SNORM, at least the plan's maximum size. The arena is cleared by the
    // same fill kernel the DX12 engine uses.
    bool Allocate(const std::vector<uint8_t>& blob, uint32_t temporalW, uint32_t temporalH, std::string& err)
    {
        weights = vk.CreateBuffer(blob.size());
        std::memcpy(weights.mapped, blob.data(), blob.size());
        arena = vk.CreateBuffer(plan.ArenaBytes());
        std::memset(arena.mapped, 0x11, plan.ArenaBytes());
        input = vk.CreateBuffer(plan.InputBytes());
        kpn   = vk.CreateBuffer(plan.KpnBytes());
        std::memset(kpn.mapped, 0x55, plan.KpnBytes());
        temporal  = vk.CreateImage(temporalW, temporalH, VK_FORMAT_R8G8B8A8_SNORM, 4);
        constants = vk.CreateBuffer(256);

        vkc::Pipeline fill;
        if (!vk.CreatePipeline(shaderDir + "/nss_fill.hlsl", "main", "cs_6_2", {}, {}, "nss_fill",
                               {{vkc::kShiftB, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {vkc::kShiftU, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}, fill, err))
            return false;
        const uint32_t quads = uint32_t(plan.ArenaBytes() / 16), groups = rdnu::DivUp(quads, 256);
        const uint32_t gx = std::min(groups, 65535u), fc[3] = {quads, 0x80808080u, gx};
        std::memcpy(constants.mapped, fc, sizeof(fc));
        bool ok = vk.Dispatch(fill, {{vkc::kShiftB, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &constants, 0, 256}},
                                     {vkc::kShiftU, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &arena}}},
                              gx, rdnu::DivUp(groups, gx), 1, err);
        vk.Destroy(fill);
        return ok;
    }

    void Release()
    {
        for (auto& kv : pipelines)
            vk.Destroy(kv.second);
        pipelines.clear();
        vk.Destroy(weights), vk.Destroy(arena), vk.Destroy(input), vk.Destroy(kpn), vk.Destroy(constants);
        vk.Destroy(temporal);
    }

    vkc::Buffer* BufferFor(rdnu::Resource r)
    {
        switch (r)
        {
        case rdnu::Resource::Input: return &input;
        case rdnu::Resource::Kpn: return &kpn;
        default: return &arena;
        }
    }

    bool Run(std::string& err)
    {
        for (const rdnu::Dispatch& d : plan.Dispatches())
        {
            const bool  emu  = d.key.kernel == rdnu::Kernel::Wmma;
            std::string name = d.key.Name() + (emu ? "_emu" : "");
            if (!pipelines.count(name))
            {
                std::vector<std::pair<uint32_t, VkDescriptorType>> layout = {
                    {vkc::kShiftB + 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                    {vkc::kShiftT + 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                    {vkc::kShiftU + 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                    {vkc::kShiftU + 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
                if (d.key.outKind == uint32_t(rdnu::OutKind::Temporal))
                    layout.push_back({vkc::kShiftU + 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
                std::vector<std::string> defines = d.key.Defines();
                if (emu)
                    defines.push_back("RDNU_WMMA_EMULATE=1");
                vkc::Pipeline p;
                if (!vk.CreatePipeline(shaderDir + "/" + d.key.Source(), "main", "cs_6_4", defines, {shaderDir}, name, layout, p, err))
                    return false;
                pipelines[name] = p;
            }
            std::memcpy(constants.mapped, &d.constants, sizeof(d.constants));
            std::map<uint32_t, vkc::Resource> res;
            res[vkc::kShiftB + 0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &constants, 0, 256};
            res[vkc::kShiftT + 0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &weights};
            res[vkc::kShiftU + 0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BufferFor(d.out)};
            res[vkc::kShiftU + 1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BufferFor(d.in)};
            vkc::Resource img;
            img.type              = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            img.image             = &temporal;
            res[vkc::kShiftU + 2] = img;
            if (!vk.Dispatch(pipelines[name], res, d.groups[0], d.groups[1], d.groups[2], err))
                return err = std::string(manifest.layers[d.layer].name) + ": " + err, false;
        }
        return true;
    }
};
