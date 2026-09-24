// rdnut.h - portable reader for the .rdnut tensor bundles written by runtime/tools/*.py
// ('RDNT', u32 version, u32 count, then per tensor: u32 name length, name, u32 ndim,
// u32 dims[ndim], float32 data).
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace rdnut
{

struct Tensor
{
    std::vector<uint32_t> dims;
    std::vector<float>    data;
    size_t Count() const
    {
        size_t n = 1;
        for (uint32_t d : dims)
            n *= d;
        return n;
    }
};

inline bool Load(const std::string& path, std::map<std::string, Tensor>& out, std::string& error)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return error = "cannot open " + path, false;
    char magic[4];
    uint32_t version = 0, count = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&count), 4);
    if (!f || std::string(magic, 4) != "RDNT")
        return error = path + ": not an .rdnut bundle", false;
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t len = 0, nd = 0;
        f.read(reinterpret_cast<char*>(&len), 4);
        std::string name(len, '\0');
        f.read(&name[0], len);
        f.read(reinterpret_cast<char*>(&nd), 4);
        Tensor t;
        t.dims.resize(nd);
        f.read(reinterpret_cast<char*>(t.dims.data()), std::streamsize(nd) * 4);
        t.data.resize(t.Count());
        f.read(reinterpret_cast<char*>(t.data.data()), std::streamsize(t.data.size()) * 4);
        if (!f)
            return error = path + ": truncated at " + name, false;
        out[name] = std::move(t);
    }
    return true;
}

inline bool ReadBytes(const std::string& path, std::vector<uint8_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

}  // namespace rdnut
