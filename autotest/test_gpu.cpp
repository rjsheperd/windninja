#ifdef WITH_WEBGPU

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <webgpu/webgpu.h>

#include <boost/test/unit_test.hpp>

#include "gpu_context.h"
#include "gpu_kernels.h"

namespace {

// The shared context, or null when this machine has no WebGPU adapter; tests
// then print the marker that ctest's SKIP_REGULAR_EXPRESSION matches.
gpu::Context *context()
{
    static std::unique_ptr<gpu::Context> c;
    static bool tried = false;
    if(!tried)
    {
        tried = true;
        try
        {
            c.reset(new gpu::Context());
        }
        catch(const gpu::Unavailable &e)
        {
            std::cout << "SKIP: no WebGPU adapter (" << e.what() << ")\n";
        }
    }
    return c.get();
}

uint32_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// distance in units in the last place, for finite floats of the same sign
uint32_t ulps(float a, float b)
{
    uint32_t ua = bits(a), ub = bits(b);
    return ua > ub ? ua - ub : ub - ua;
}

} // namespace

BOOST_AUTO_TEST_SUITE(gpu)

BOOST_AUTO_TEST_CASE( dawn_instance )
{
    WGPUInstance instance = wgpuCreateInstance(NULL);
    BOOST_REQUIRE(instance != NULL);
    wgpuInstanceRelease(instance);
}

BOOST_AUTO_TEST_CASE( device_create )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    BOOST_CHECK(c->device() != nullptr);
    BOOST_CHECK(c->queue() != nullptr);
    BOOST_TEST_MESSAGE("adapter: " << c->adapterName());
}

BOOST_AUTO_TEST_CASE( buffer_roundtrip )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // 1M floats, including NaN payloads and denormals, must come back bit
    // for bit
    const size_t n = 1 << 20;
    std::vector<uint32_t> in(n);
    for(size_t i = 0; i < n; i++)
        in[i] = static_cast<uint32_t>(i * 2654435761u);
    gpu::Buffer buffer(*c, n * sizeof(float));
    buffer.upload(in);
    std::vector<uint32_t> out = buffer.download<uint32_t>(n);
    BOOST_CHECK(out == in);
}

BOOST_AUTO_TEST_CASE( kernel_saxpy )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    // n isn't a multiple of the workgroup size, so the bounds check matters
    const uint32_t n = 100003;
    const float a = 1.7f;
    std::vector<float> x(n), y(n);
    for(uint32_t i = 0; i < n; i++)
    {
        x[i] = std::sin(0.001f * i) * 100.0f;
        y[i] = std::cos(0.002f * i) - 0.5f;
    }
    struct { float a; uint32_t n; } params = {a, n};

    gpu::Buffer p(*c, sizeof params, gpu::Buffer::Uniform);
    gpu::Buffer bx(*c, n * sizeof(float));
    gpu::Buffer by(*c, n * sizeof(float));
    p.upload(&params, sizeof params);
    bx.upload(x);
    by.upload(y);
    gpu::Kernel saxpy(*c, gpu::kernels::saxpy);
    saxpy.dispatch({&p, &bx, &by}, (n + 63) / 64);
    std::vector<float> out = by.download<float>(n);

    // the GPU may or may not fuse a * x + y, so accept either rounding
    // within 1 ulp
    uint32_t worst = 0;
    for(uint32_t i = 0; i < n; i++)
    {
        float separate = a * x[i] + y[i];
        float fused = std::fma(a, x[i], y[i]);
        uint32_t d = std::min(ulps(out[i], separate), ulps(out[i], fused));
        worst = std::max(worst, d);
    }
    BOOST_CHECK_LE(worst, 1u);
}

BOOST_AUTO_TEST_CASE( validation_error_throws )
{
    gpu::Context *c = context();
    if(c == nullptr)
        return;
    const char *bad = "@compute @workgroup_size(1)\n"
                      "fn main() {\n"
                      "    let x : f32 = undefined_name;\n"
                      "}\n";
    try
    {
        gpu::Kernel kernel(*c, bad);
        BOOST_FAIL("bad WGSL compiled");
    }
    catch(const gpu::Error &e)
    {
        std::string message = e.what();
        BOOST_TEST_MESSAGE(message);
        BOOST_CHECK(message.find("undefined_name") != std::string::npos);
        BOOST_CHECK(message.find("line 3") != std::string::npos);
    }
    // the context is still usable afterwards
    gpu::Kernel good(*c, gpu::kernels::saxpy);
}

BOOST_AUTO_TEST_SUITE_END()

#endif
