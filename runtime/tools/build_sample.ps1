<#
.SYNOPSIS
    Build the FidelityFX FSR sample (with the RDNU upscaler integrated) using MSBuild,
    without opening the Visual Studio IDE.

.DESCRIPTION
    Locates MSBuild via vswhere and builds
    runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/FidelityFX_FSR_2022.sln.
    The RDNU backend (runtime/src/rdg_dx12_backend.cpp) and shaders/models are pulled in by that
    solution's project references and post-build copy steps.

.EXAMPLE
    pwsh runtime/tools/build_sample.ps1 -Config Release
    pwsh runtime/tools/build_sample.ps1 -Config Debug -Rebuild
#>
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')] [string] $Config = 'Release',
    [string] $Platform = 'x64',
    [switch] $Rebuild
)

$ErrorActionPreference = 'Stop'

# runtime/tools -> runtime -> repo root
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sln = Join-Path $repoRoot 'runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/FidelityFX_FSR_2022.sln'

if (-not (Test-Path $sln)) {
    throw "Solution not found: $sln`nDid you initialise submodules?  git submodule update --init --recursive"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found. Install Visual Studio 2022+ or the VS Build Tools (with the C++ workload)."
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
if (-not $vsPath) { throw "No Visual Studio installation with MSBuild was found." }

$msbuild = Join-Path $vsPath 'MSBuild\Current\Bin\MSBuild.exe'
if (-not (Test-Path $msbuild)) { throw "MSBuild.exe not found under $vsPath" }

$target = if ($Rebuild) { 'Rebuild' } else { 'Build' }

Write-Host "MSBuild : $msbuild"
Write-Host "Solution: $sln"
Write-Host "Build   : $Config | $Platform | $target"
Write-Host ''

& $msbuild $sln "/t:$target" "/p:Configuration=$Config" "/p:Platform=$Platform" /m /nologo /verbosity:minimal
$code = $LASTEXITCODE

if ($code -eq 0) {
    $exe = Join-Path $repoRoot "runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/$Platform/$Config/FidelityFX_FSR.exe"
    Write-Host ''
    Write-Host "Build succeeded. Run: $exe"
}
exit $code
