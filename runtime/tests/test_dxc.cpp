#include <windows.h>
#include <dxcapi.h>
#include <iostream>

#pragma comment(lib, "dxcompiler.lib")

int main() {
    IDxcUtils* utils;
    HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    if (SUCCEEDED(hr)) {
        std::cout << "DXC works!" << std::endl;
        utils->Release();
    }
    return 0;
}