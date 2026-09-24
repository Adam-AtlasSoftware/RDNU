// rdnu_shaderc.cpp - compiles every shader the runtime needs to DXIL and writes them into one
// C++ source (rdnu::FindShader). Network kernels come from the model manifest, pass shaders
// from rdnu_pass_shaders.h. Runs on Windows and Linux with any DXC; the wave-matrix kernels
// need the AMD shader intrinsics headers (FidelityFX SDK, api/internal/dx12/AmdExtD3D).
//
//   rdnu_shaderc --dxc <dxc> --shaders <runtime/shaders> --manifest <nss.rdnm>
//                --amd-ext <AmdExtD3D dir> --out <rdnu_shaders_dxil.cpp> [--no-wmma]
#include "../src/engine/rdnu_engine_core.h"
#include "../src/engine/rdnu_pass_shaders.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace
{
std::string Quote(const std::string& s)
{
#ifdef _WIN32
    return "\"" + s + "\"";
#else
    return "'" + s + "'";
#endif
}

bool ReadAll(const std::string& path, std::vector<unsigned char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

struct Job
{
    std::string              name, file, profile;
    std::vector<std::string> defines, includes, flags;
};

bool Compile(const std::string& dxc, const std::string& tmp, const Job& j, std::vector<unsigned char>& dxil)
{
    const std::string out = tmp + "/" + j.name + ".dxil", log = tmp + "/" + j.name + ".log";
    std::ostringstream cmd;
    cmd << Quote(dxc) << " -T " << j.profile << " -E main -HV 2021 -O3 -enable-16bit-types";
    for (const std::string& f : j.flags)
        cmd << " " << f;
    for (const std::string& d : j.defines)
        cmd << " -D " << Quote(d);
    for (const std::string& i : j.includes)
        cmd << " -I " << Quote(i);
    cmd << " " << Quote(j.file) << " -Fo " << Quote(out) << " > " << Quote(log) << " 2>&1";
    std::string c = cmd.str();
#ifdef _WIN32
    c = "\"" + c + "\"";  // cmd.exe strips one level of quotes
#endif
    if (std::system(c.c_str()) != 0 || !ReadAll(out, dxil))
    {
        std::vector<unsigned char> l;
        ReadAll(log, l);
        std::fprintf(stderr, "rdnu_shaderc: %s failed\n%.*s\n", j.name.c_str(), int(l.size()), reinterpret_cast<const char*>(l.data()));
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string dxc, shaders, manifestPath, amdExt, out;
    bool        wmma = true;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&](std::string& v) { if (i + 1 < argc) v = argv[++i]; };
        if (a == "--dxc") next(dxc);
        else if (a == "--shaders") next(shaders);
        else if (a == "--manifest") next(manifestPath);
        else if (a == "--amd-ext") next(amdExt);
        else if (a == "--out") next(out);
        else if (a == "--no-wmma") wmma = false;
    }
    if (dxc.empty() || shaders.empty() || manifestPath.empty() || out.empty() || (wmma && amdExt.empty()))
        return std::fprintf(stderr, "usage: rdnu_shaderc --dxc <dxc> --shaders <dir> --manifest <nss.rdnm> --amd-ext <dir> --out <file> [--no-wmma]\n"), 1;

    std::vector<unsigned char> bytes;
    std::string                err;
    rdnu::Manifest             manifest;
    if (!ReadAll(manifestPath, bytes) || !rdnu::ParseManifest(bytes.data(), bytes.size(), manifest, err))
        return std::fprintf(stderr, "rdnu_shaderc: %s: %s\n", manifestPath.c_str(), err.c_str()), 1;
    rdnu::PlanConfig cfg;
    cfg.maxWidth = cfg.maxHeight = 64;
    rdnu::Plan plan;
    if (!plan.Build(manifest, cfg, err))
        return std::fprintf(stderr, "rdnu_shaderc: %s\n", err.c_str()), 1;

    const std::string net = shaders + "/net", nss = shaders + "/nss";
    std::vector<Job> jobs;
    jobs.push_back({"nss_fill", net + "/nss_fill.hlsl", "cs_6_2", {}, {}, {}});
    jobs.push_back({"rcas", nss + "/ffx_rcas_pass.hlsl", "cs_6_2", {}, {}, {}});
    for (const rdnu::KernelKey& k : plan.RequiredKernels(wmma))
    {
        Job j{k.Name(), net + "/" + k.Source(), "cs_6_4", k.Defines(), {net}, {}};
        if (k.kernel == rdnu::Kernel::Wmma)
            j.profile = "cs_6_6", j.includes.push_back(amdExt);
        jobs.push_back(j);
    }
    for (const rdnu::PassShader& p : rdnu::PassShaders())
        for (uint32_t q : rdnu::PassQualities())
            for (uint32_t bits = 0; bits < 8; ++bits)
            {
                if (bits & ~p.bits)
                    continue;
                jobs.push_back({rdnu::PassShaderName(p, q, bits), nss + "/ffx_nss_" + p.name + ".hlsl", "cs_6_2",
                                rdnu::PassShaderDefines(p, q, bits), {nss}, {"-Wno-ambig-lit-shift"}});
            }

    const std::string tmp = out + ".d";
#ifdef _WIN32
    const std::string mkdir = "mkdir " + Quote(tmp) + " 2> nul";
#else
    const std::string mkdir = "mkdir -p " + Quote(tmp);
#endif
    if (std::system(mkdir.c_str())) {}  // exists already, or Compile reports the failure
    std::ostringstream src;
    src << "// Generated by rdnu_shaderc. Do not edit.\n#include \"rdnu_shader_blobs.h\"\n\n#include <cstring>\n\nnamespace\n{\n";
    size_t total = 0;
    for (size_t i = 0; i < jobs.size(); ++i)
    {
        std::vector<unsigned char> dxil;
        if (!Compile(dxc, tmp, jobs[i], dxil))
            return 2;
        total += dxil.size();
        src << "const unsigned char kBlob" << i << "[] = {";
        for (size_t b = 0; b < dxil.size(); ++b)
            src << (b % 24 ? "" : "\n    ") << unsigned(dxil[b]) << ",";
        src << "\n};\n";
    }
    src << "const rdnu::ShaderBlob kShaders[] = {\n";
    for (size_t i = 0; i < jobs.size(); ++i)
        src << "    {\"" << jobs[i].name << "\", kBlob" << i << ", sizeof(kBlob" << i << ")},\n";
    src << "};\n}  // namespace\n\nconst rdnu::ShaderBlob* rdnu::FindShader(const char* name)\n{\n"
        << "    for (const ShaderBlob& s : kShaders)\n        if (!std::strcmp(s.name, name))\n            return &s;\n"
        << "    return nullptr;\n}\n";
    std::ofstream f(out, std::ios::binary);
    f << src.str();
    if (!f)
        return std::fprintf(stderr, "rdnu_shaderc: cannot write %s\n", out.c_str()), 1;
    std::printf("rdnu_shaderc: %zu shaders, %zu KB of DXIL -> %s\n", jobs.size(), total / 1024, out.c_str());
    return 0;
}
