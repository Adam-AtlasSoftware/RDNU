// rdnu_shaderc.cpp - compiles every shader the runtime needs to DXIL and writes them, with the
// model files, into one C++ source (rdnu_embedded.h). Network kernels come from the model
// manifest, pass shaders from rdnu_pass_shaders.h. Runs on Windows and Linux with any DXC; the
// wave-matrix kernels need the AMD shader intrinsics headers (FidelityFX SDK,
// api/internal/dx12/AmdExtD3D).
//
//   rdnu_shaderc --dxc <dxc> --shaders <runtime/shaders> --model <dir with nss.rdnm, nss_w8.bin>
//                --amd-ext <AmdExtD3D dir> --out <rdnu_embedded.cpp> [--no-wmma]
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

struct Binding
{
    std::string name, kind;
    unsigned    slot;
};

// Parses the "Resource Bindings" table of a DXC listing: only resources the shader uses.
bool ParseBindings(const std::string& listing, std::vector<Binding>& out)
{
    std::ifstream f(listing);
    std::string   line;
    bool          table = false;
    while (std::getline(f, line))
    {
        if (line.rfind("; Resource Bindings:", 0) == 0)
        {
            table = true;
            continue;
        }
        if (!table)
            continue;
        std::istringstream ss(line.substr(1));
        std::string        name, type, format, dim, id, bind, count;
        if (!(ss >> name >> type >> format >> dim >> id >> bind >> count))
        {
            if (line.size() <= 1 && !out.empty())
                break;
            continue;
        }
        if (name == "Name" || name[0] == '-')
            continue;
        const char* kind = nullptr;
        size_t      digits = bind.find_first_of("0123456789");
        if (type == "cbuffer")
            kind = "Cbv";
        else if (type == "sampler")
            kind = "Sampler";
        else if (type == "texture")
            kind = dim == "r/o" || dim == "buf" ? "SrvBuffer" : "SrvTexture";
        else if (type == "UAV")
            kind = dim == "r/w" || dim == "buf" ? "UavBuffer" : "UavTexture";
        if (!kind || digits == std::string::npos)
            return false;
        out.push_back({name, kind, unsigned(std::stoul(bind.substr(digits)))});
    }
    return true;
}

bool Compile(const std::string& dxc, const std::string& tmp, const Job& j, std::vector<unsigned char>& dxil,
             std::vector<Binding>& bindings)
{
    const std::string out = tmp + "/" + j.name + ".dxil", log = tmp + "/" + j.name + ".log", lst = tmp + "/" + j.name + ".lst";
    std::ostringstream cmd;
    cmd << Quote(dxc) << " -T " << j.profile << " -E main -HV 2021 -O3 -enable-16bit-types";
    for (const std::string& f : j.flags)
        cmd << " " << f;
    for (const std::string& d : j.defines)
        cmd << " -D " << Quote(d);
    for (const std::string& i : j.includes)
        cmd << " -I " << Quote(i);
    cmd << " " << Quote(j.file) << " -Fo " << Quote(out) << " -Fc " << Quote(lst) << " > " << Quote(log) << " 2>&1";
    std::string c = cmd.str();
#ifdef _WIN32
    c = "\"" + c + "\"";  // cmd.exe strips one level of quotes
#endif
    if (std::system(c.c_str()) != 0 || !ReadAll(out, dxil) || !ParseBindings(lst, bindings))
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
    std::string dxc, shaders, modelDir, amdExt, out;
    bool        wmma = true;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&](std::string& v) { if (i + 1 < argc) v = argv[++i]; };
        if (a == "--dxc") next(dxc);
        else if (a == "--shaders") next(shaders);
        else if (a == "--model") next(modelDir);
        else if (a == "--amd-ext") next(amdExt);
        else if (a == "--out") next(out);
        else if (a == "--no-wmma") wmma = false;
    }
    if (dxc.empty() || shaders.empty() || modelDir.empty() || out.empty() || (wmma && amdExt.empty()))
        return std::fprintf(stderr, "usage: rdnu_shaderc --dxc <dxc> --shaders <dir> --model <dir> --amd-ext <dir> --out <file> [--no-wmma]\n"), 1;

    const char* modelFiles[] = {"nss.rdnm", "nss_w8.bin"};
    std::vector<unsigned char> model[2];
    for (int i = 0; i < 2; ++i)
        if (!ReadAll(modelDir + "/" + modelFiles[i], model[i]))
            return std::fprintf(stderr, "rdnu_shaderc: cannot read %s/%s\n", modelDir.c_str(), modelFiles[i]), 1;
    std::string    err;
    rdnu::Manifest manifest;
    if (!rdnu::ParseManifest(model[0].data(), model[0].size(), manifest, err))
        return std::fprintf(stderr, "rdnu_shaderc: nss.rdnm: %s\n", err.c_str()), 1;
    rdnu::PlanConfig cfg;
    cfg.maxWidth = cfg.maxHeight = 64;
    rdnu::Plan plan;
    if (!plan.Build(manifest, cfg, err))
        return std::fprintf(stderr, "rdnu_shaderc: %s\n", err.c_str()), 1;

    const std::string net = shaders + "/net", nss = shaders + "/nss";
    std::vector<Job> jobs;
    jobs.push_back({"nss_fill", net + "/nss_fill.hlsl", "cs_6_2", {}, {}, {}});
    jobs.push_back({"rcas", nss + "/ffx_rcas_pass.hlsl", "cs_6_2", {}, {}, {}});
    jobs.push_back({"rdnu_exposure", nss + "/rdnu_exposure.hlsl", "cs_6_2", {}, {}, {}});
    jobs.push_back({"rdnu_exposure_patch", nss + "/rdnu_exposure.hlsl", "cs_6_2", {"RDNU_EXPOSURE_PATCH=1"}, {}, {}});
    jobs.push_back({"rdnu_motion", nss + "/rdnu_motion.hlsl", "cs_6_2", {}, {}, {}});
    for (const rdnu::KernelKey& k : plan.RequiredKernels(wmma))
    {
        Job j{k.Name(), net + "/" + k.Source(), "cs_6_4", k.Defines(), {net}, {}};
        if (k.kernel == rdnu::Kernel::Wmma)
            j.profile = "cs_6_6", j.includes.push_back(amdExt);
        jobs.push_back(j);
    }
    // the pre-process pass quantises the network input with the model's learned scale
    char inputScale[64] = "";
    for (const rdnu::TensorRecord& t : manifest.tensors)
        if (rdnu::TensorKind(t.kind) == rdnu::TensorKind::Input)
            std::snprintf(inputScale, sizeof(inputScale), "RDNU_INPUT_SCALE=%.17g", double(t.scale));
    for (const rdnu::PassShader& p : rdnu::PassShaders())
        for (uint32_t q : rdnu::PassQualities())
            for (uint32_t bits = 0; bits < 8; ++bits)
            {
                if (bits & ~p.bits)
                    continue;
                std::vector<std::string> defines = rdnu::PassShaderDefines(p, q, bits);
                defines.push_back(inputScale);
                jobs.push_back({rdnu::PassShaderName(p, q, bits), nss + "/ffx_nss_" + p.name + ".hlsl", "cs_6_2", defines, {nss},
                                {"-Wno-ambig-lit-shift"}});
            }

    const std::string tmp = out + ".d";
#ifdef _WIN32
    const std::string mkdir = "mkdir " + Quote(tmp) + " 2> nul";
#else
    const std::string mkdir = "mkdir -p " + Quote(tmp);
#endif
    if (std::system(mkdir.c_str())) {}  // exists already, or Compile reports the failure
    std::ostringstream src;
    src << "// Generated by rdnu_shaderc. Do not edit.\n#include \"rdnu_embedded.h\"\n\n#include <cstring>\n\nnamespace\n{\n";
    auto bytes = [&](const char* name, const std::vector<unsigned char>& data) {
        src << "const unsigned char " << name << "[] = {";
        for (size_t b = 0; b < data.size(); ++b)
            src << (b % 24 ? "" : "\n    ") << unsigned(data[b]) << ",";
        src << "\n};\n";
    };
    bytes("kModel0", model[0]);
    bytes("kModel1", model[1]);
    size_t              total = 0;
    std::vector<size_t> counts;
    for (size_t i = 0; i < jobs.size(); ++i)
    {
        std::vector<unsigned char> dxil;
        std::vector<Binding>       bindings;
        if (!Compile(dxc, tmp, jobs[i], dxil, bindings))
            return 2;
        total += dxil.size();
        bytes(("kBlob" + std::to_string(i)).c_str(), dxil);
        src << "const rdnu::ShaderBinding kBindings" << i << "[] = {\n";
        for (const Binding& b : bindings)
            src << "    {\"" << b.name << "\", rdnu::BindingKind::" << b.kind << ", " << b.slot << "},\n";
        src << "    {nullptr, rdnu::BindingKind::Cbv, 0}};\n";
        counts.push_back(bindings.size());
    }
    src << "const rdnu::ShaderBlob kShaders[] = {\n";
    for (size_t i = 0; i < jobs.size(); ++i)
        src << "    {\"" << jobs[i].name << "\", kBlob" << i << ", sizeof(kBlob" << i << "), kBindings" << i << ", " << counts[i] << "},\n";
    src << "};\nconst rdnu::EmbeddedFile kModel[] = {\n    {\"nss.rdnm\", kModel0, sizeof(kModel0)},\n"
        << "    {\"nss_w8.bin\", kModel1, sizeof(kModel1)},\n};\n}  // namespace\n\n"
        << "const rdnu::ShaderBlob* rdnu::FindShader(const char* name)\n{\n"
        << "    for (const ShaderBlob& s : kShaders)\n        if (!std::strcmp(s.name, name))\n            return &s;\n"
        << "    return nullptr;\n}\n\n"
        << "const rdnu::EmbeddedFile* rdnu::FindModelFile(const char* name)\n{\n"
        << "    for (const EmbeddedFile& f : kModel)\n        if (!std::strcmp(f.name, name))\n            return &f;\n"
        << "    return nullptr;\n}\n";
    std::ofstream f(out, std::ios::binary);
    f << src.str();
    if (!f)
        return std::fprintf(stderr, "rdnu_shaderc: cannot write %s\n", out.c_str()), 1;
    std::printf("rdnu_shaderc: %zu shaders, %zu KB of DXIL -> %s\n", jobs.size(), total / 1024, out.c_str());
    return 0;
}
