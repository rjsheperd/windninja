/*
** Browser check for the GPU layer (WEBGPU_PLAN 3.4): runs the embedded saxpy
** kernel through gpu::Context / Buffer / Kernel, reads the result back (which
** waits on the GPU via JSPI or Asyncify), and compares it with the CPU.
** It then checks that a WGSL compile error throws (error scopes and
** compilation info also wait on the GPU).
** The last line is "RESULT PASS", "RESULT FAIL ..." or "RESULT SKIP ...".
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_context.h"
#include "gpu_kernels.h"

namespace {

uint32_t ulps(float a, float b)
{
    uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof ua);
    std::memcpy(&ub, &b, sizeof ub);
    return ua > ub ? ua - ub : ub - ua;
}

} // namespace

int main()
{
    const uint32_t n = 100003;
    const float a = 1.7f;
    std::vector<float> x(n), y(n);
    for(uint32_t i = 0; i < n; i++)
    {
        x[i] = std::sin(0.001f * i) * 100.0f;
        y[i] = std::cos(0.002f * i) - 0.5f;
    }
    struct { float a; uint32_t n; } params = {a, n};

    try
    {
        gpu::Context c;
        std::printf("adapter: %s\n", c.adapterName().c_str());
        gpu::Buffer p(c, sizeof params, gpu::Buffer::Uniform);
        gpu::Buffer bx(c, n * sizeof(float));
        gpu::Buffer by(c, n * sizeof(float));
        p.upload(&params, sizeof params);
        bx.upload(x);
        by.upload(y);
        gpu::Kernel saxpy(c, gpu::kernels::saxpy);
        saxpy.dispatch({&p, &bx, &by}, (n + 63) / 64);
        std::vector<float> out = by.download<float>(n);

        uint32_t worst = 0;
        for(uint32_t i = 0; i < n; i++)
        {
            uint32_t d = std::min(ulps(out[i], a * x[i] + y[i]),
                                  ulps(out[i], std::fma(a, x[i], y[i])));
            worst = std::max(worst, d);
        }
        std::printf("saxpy: %u elements, worst %u ulp\n", n, worst);
        if(worst > 1)
        {
            std::printf("RESULT FAIL saxpy differs from the CPU\n");
            return 1;
        }

        try
        {
            gpu::Kernel bad(c, "@compute @workgroup_size(1)\n"
                               "fn main() {\n"
                               "    let x : f32 = undefined_name;\n"
                               "}\n");
            std::printf("RESULT FAIL bad WGSL compiled\n");
            return 1;
        }
        catch(const gpu::Error &e)
        {
            std::string message = e.what();
            std::printf("bad WGSL: %s", message.c_str());
            if(message.find("line 3") == std::string::npos)
            {
                std::printf("RESULT FAIL compile error has no line number\n");
                return 1;
            }
        }
    }
    catch(const gpu::Unavailable &e)
    {
        std::printf("RESULT SKIP %s\n", e.what());
        return 0;
    }
    catch(const gpu::Error &e)
    {
        std::printf("RESULT FAIL %s\n", e.what());
        return 1;
    }
    std::printf("RESULT PASS\n");
    return 0;
}
