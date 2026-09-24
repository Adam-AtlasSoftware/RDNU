#!/usr/bin/env python3
"""Port Arm's GLSL NSS pass headers to HLSL.

Reads sdk/include/FidelityFX/gpu/nss/*.h from the neural-graphics SDK submodule and writes
runtime/shaders/nss/generated/. The algorithms are untouched: only declarations, GLSL
built-ins that have no HLSL spelling and the int8 tensor buffer accessors are rewritten.
Every targeted rewrite must match exactly once, so an upstream change fails loudly.

    python3 runtime/tools/port_nss_shaders.py [--check]
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "runtime/external/neural-graphics-sdk-for-game-engines/sdk/include/FidelityFX/gpu/nss"
DST = ROOT / "runtime/shaders/nss/generated"

FILES = {
    "ffx_nss_resources.h": "ffx_nss_resources.h",
    "ffx_nss_common_glsl.h": "ffx_nss_common_hlsl.h",
    "ffx_nss_depth_scatter.h": "ffx_nss_depth_scatter.h",
    "ffx_nss_disocclusion_mask_lq.h": "ffx_nss_disocclusion_mask_lq.h",
    "ffx_nss_generate_offset_lut.h": "ffx_nss_generate_offset_lut.h",
    "ffx_nss_preprocess.h": "ffx_nss_preprocess.h",
    "ffx_nss_postprocess.h": "ffx_nss_postprocess.h",
    "ffx_nss_debug_view.h": "ffx_nss_debug_view.h",
}

VECTOR_TYPES = [f"{b}{n}" for b in ("float", "half", "int", "uint", "bool", "vec", "ivec", "uvec",
                                    "int32_t", "uint32_t", "int16_t", "uint16_t", "int8_t")
                for n in (2, 3, 4)]

# (pattern, replacement) pairs that must match exactly once in the file they apply to.
TARGETED = {
    "ffx_nss_common_glsl.h": [
        (r'#include "ffx_nss_resources.h"', '#include "../ffx_nss_hlsl_compat.h"\n#include "ffx_nss_resources.h"'),
        (r'#include "ffx_core.h"\n', ""),
        # HLSL has rcp and saturate intrinsics for every float width.
        (r"#if FFX_HALF\n// --- RCP functions for float16 types ---.*?(?=#define MAX_FP16)", ""),
    ],
    "ffx_nss_preprocess.h": [
        # the input quantiser scale is learned per model: rdnu_shaderc passes the manifest's
        (r"// High model\n"
         r"const half2 kPreprocessQuant   = half2\(1\.0 / 0\.003912401385605335, -128\.0\);\n"
         r"const half2 kPreprocessDequant = half2\(0\.003912401385605335, -128\.0\);\n",
         "// High model; RDNU_INPUT_SCALE from the model manifest (rdnu_shaderc), else the released one\n"
         "#ifndef RDNU_INPUT_SCALE\n"
         "#define RDNU_INPUT_SCALE 0.003912401385605335\n"
         "#endif\n"
         "const half2 kPreprocessQuant   = half2(1.0 / RDNU_INPUT_SCALE, -128.0);\n"
         "const half2 kPreprocessDequant = half2(RDNU_INPUT_SCALE, -128.0);\n"),
        (r"    rw_preprocessed_tensor_buffer\.data\[base \+ 0u\] = t_vec0;\n"
         r"    rw_preprocessed_tensor_buffer\.data\[base \+ 1u\] = t_vec1;\n"
         r"    rw_preprocessed_tensor_buffer\.data\[base \+ 2u\] = t_vec2;\n",
         "    rw_preprocessed_tensor_buffer.Store3(base * 4u, uint3(PackI8x4(t_vec0), PackI8x4(t_vec1), PackI8x4(t_vec2)));\n"),
        (r"    t_vec0        = rw_preprocessed_tensor_buffer\.data\[base \+ 0u\];\n"
         r"    t_vec1        = rw_preprocessed_tensor_buffer\.data\[base \+ 1u\];\n"
         r"    t_vec2        = rw_preprocessed_tensor_buffer\.data\[base \+ 2u\];\n",
         "    uint3 words   = rw_preprocessed_tensor_buffer.Load3(base * 4u);\n"
         "    t_vec0        = UnpackI8x4(words.x);\n"
         "    t_vec1        = UnpackI8x4(words.y);\n"
         "    t_vec2        = UnpackI8x4(words.z);\n"),
    ],
    "ffx_nss_postprocess.h": [
        # the alias expands to "tex, sampler"; parentheses would make it a comma expression
        (r"#define _NearestDepthOffsetTex \(_NearestDepthCoordTex\)", "#define _NearestDepthOffsetTex _NearestDepthCoordTex"),
        (r"#define _MotionVectorTex \(_MotionTex\)", "#define _MotionVectorTex _MotionTex"),
        (r"r_coefficients_kpn_buffer\.data\[linear_idx\]",
         "UnpackI8x4(r_coefficients_kpn_buffer.Load(uint(linear_idx) * 4u))"),
    ],
}

# Declarations. GLSL binding numbers become t#/u#/b# of the same number.
DECLS = [
    (r"layout\(set = 0, binding = 1000\) uniform sampler s_PointClamp;",
     "SamplerState s_PointClamp : register(s0);"),
    (r"layout\(set = 0, binding = 1001\) uniform sampler s_LinearClamp;",
     "SamplerState s_LinearClamp : register(s1);"),
    (r"layout\(set = 0, binding = (\w+)\) uniform (?:(?:highp|mediump|lowp) )?texture2D (\w+);",
     r"Texture2D<float4> \2 : RDNU_REG_T(\1);"),
    (r"layout\(set = 0, binding = (\w+)\) uniform (?:(?:highp|mediump|lowp) )?utexture2D (\w+);",
     r"Texture2D<uint4> \2 : RDNU_REG_T(\1);"),
    (r"layout\(set = 0, binding = (\w+), r32ui\) (?:coherent )?uniform uimage2D (\w+);",
     r"RWTexture2D<uint> \2 : RDNU_REG_U(\1);"),
    (r"layout\(set = 0, binding = (\w+), rgba32ui\) uniform (?:writeonly )?uimage2D (\w+);",
     r"RWTexture2D<uint4> \2 : RDNU_REG_U(\1);"),
    (r"layout\(set = 0, binding = (\w+), \w+\) uniform (?:writeonly )?(?:(?:highp|mediump|lowp) )?image2D (\w+);",
     r"RWTexture2D<float4> \2 : RDNU_REG_U(\1);"),
    (r"layout\(set = 0, binding = (\w+), std140\) uniform (\w+)\n\{(.*?)\}\n(\w+);",
     r"struct \2\n{\3};\nConstantBuffer<\2> \4 : RDNU_REG_B(\1);"),
    # tensor buffers keep the GLSL block name: the component binds them by that name
    (r"layout\(set = 0, binding = (\w+), std430\) readonly buffer (\w+)\n\{\n\s*int8_t4 data\[\];\n\}\n(\w+);",
     r"ByteAddressBuffer \2 : RDNU_REG_T(\1);\n#define \3 \2"),
    (r"layout\(set = 0, binding = (\w+), std430\) buffer (\w+)\n\{\n\s*int8_t4 data\[\];\n\}\n(\w+);",
     r"RWByteAddressBuffer \2 : RDNU_REG_U(\1);\n#define \3 \2"),
]


def match_paren(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced parentheses at %d" % open_idx)


def split_top_level(args):
    parts, depth, cur = [], 0, ""
    for ch in args:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def splat_to_cast(text):
    """HLSL has no scalar splat constructor: T(x) with one argument becomes ((T)(x))."""
    pat = re.compile(r"\b(" + "|".join(VECTOR_TYPES) + r")\s*\(")
    out, pos = "", 0
    while True:
        m = pat.search(text, pos)
        if not m:
            return out + text[pos:]
        close = match_paren(text, m.end() - 1)
        inner = splat_to_cast(text[m.end():close])
        if len(split_top_level(inner)) == 1:
            out += text[pos:m.start()] + "((%s)(%s))" % (m.group(1), inner)
        else:
            out += text[pos:m.end()] + inner + ")"
        pos = close + 1


def array_ctors(text, structs):
    """T[N](a, b) -> {a, b}; struct constructors inside become brace initialisers."""
    pat = re.compile(r"=\s*(\w+)\[\w*\]\s*\(")
    while True:
        m = pat.search(text)
        if not m:
            return text
        close = match_paren(text, m.end() - 1)
        inner = text[m.end():close]
        for s in structs:
            inner = brace_calls(inner, s)
        text = text[:m.start()] + "= {" + inner + "}" + text[close + 1:]


def brace_calls(text, name):
    pat = re.compile(r"\b" + name + r"\(")
    while True:
        m = pat.search(text)
        if not m:
            return text
        close = match_paren(text, m.end() - 1)
        text = text[:m.start()] + "{" + text[m.end():close] + "}" + text[close + 1:]


def sub_once(pattern, repl, text, name):
    new, n = re.subn(pattern, repl, text, flags=re.S)
    if n != 1:
        sys.exit("%s: expected one match for %r, found %d" % (name, pattern[:60], n))
    return new


def port(name, text):
    for pattern, repl in TARGETED.get(name, []):
        text = sub_once(pattern, repl, text, name)
    for pattern, repl in DECLS:
        text = re.sub(pattern, repl, text, flags=re.S)
    text = re.sub(r'#include "nss/ffx_nss_common_glsl\.h"', '#include "ffx_nss_common_hlsl.h"', text)
    text = re.sub(r'#include "nss/(ffx_nss_\w+\.h)"', r'#include "\1"', text)

    # 1.HF, 0.5HF, 1e-7HF -> half literals
    text = re.sub(r"\b(\d+)\.HF\b", r"\1.0h", text)
    text = re.sub(r"\b(\d+\.\d+|\d+(?:\.\d*)?[eE][-+]?\d+)HF\b", r"\1h", text)

    # file-scope constants
    text = re.sub(r"^const ", "static const ", text, flags=re.M)

    # texture() is a reserved word in HLSL
    text = re.sub(r"\btexture\s*\(", "textureSample(", text)

    # GLSL matrix * vector sums columns; HLSL rows: mul(v, M) is the same sum
    text = re.sub(r"\b(NSS_SQ_MAT\(\w+\)|taps\d?) \* (\w+)", r"mul(\2, \1)", text)

    for mat in re.findall(r"f16mat4x4 (\w+);", text):
        if re.search(r"\b%s\s*\*" % mat, text):
            sys.exit("%s: GLSL matrix product on %s not converted" % (name, mat))

    structs = re.findall(r"^struct (\w+)", text, flags=re.M)
    text = array_ctors(text, structs)
    text = splat_to_cast(text)

    left = [l for l in re.findall(r"\blayout\s*\(set.*|\.data\[.*", text) if "tensorARM" not in l]
    if left:
        sys.exit("%s: unported GLSL left:\n  %s" % (name, "\n  ".join(left)))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = ap.parse_args()

    header = ("// Generated by runtime/tools/port_nss_shaders.py from Arm's %s (MIT).\n"
              "// Do not edit: change the tool or the compat header instead.\n")
    stale = []
    DST.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        text = header % src + port(src, (SRC / src).read_text())
        path = DST / dst
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(dst)
        else:
            path.write_text(text)
    if stale:
        sys.exit("stale: " + ", ".join(stale))


if __name__ == "__main__":
    main()
