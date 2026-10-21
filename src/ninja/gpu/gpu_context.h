/******************************************************************************
 *
 * Project:  WindNinja
 * Purpose:  WebGPU (Dawn) device, buffer and compute-kernel layer
 *
 ******************************************************************************
 *
 * THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
 * MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT
 * IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105
 * OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT
 * PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES
 * LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER
 * PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY,
 * RELIABILITY, OR ANY OTHER CHARACTERISTIC.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>

/*
** A thin RAII layer over the C webgpu.h API (Dawn natively, emdawnwebgpu in
** a browser).  It depends on nothing else in WindNinja so the solver-only
** WebAssembly module (WEBGPU_PLAN 3.4) can reuse it.
**
** Every wait on the GPU goes through Context::wait(), which calls
** wgpuInstanceWaitAny.  Natively that blocks; in a browser build it needs
** Emscripten's JSPI or Asyncify to yield to the event loop.  Keep it the only
** blocking point.
*/
namespace gpu {

class Error : public std::runtime_error
{
public:
    explicit Error(const std::string &what) : std::runtime_error(what) {}
};

// no WebGPU adapter or device on this machine
class Unavailable : public Error
{
public:
    explicit Unavailable(const std::string &what) : Error(what) {}
};

class Context
{
public:
    // Requests a high-performance adapter, then any adapter.  Throws
    // Unavailable when there is none.
    Context();
    ~Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    WGPUDevice device() const { return device_; }
    WGPUQueue queue() const { return queue_; }
    // adapter description, e.g. "Apple M3 Pro (Metal)"
    const std::string &adapterName() const { return adapterName_; }

    // Waits for a future, then throws if the device was lost or reported an
    // uncaptured error.
    void wait(WGPUFuture future);
    // Waits until all work submitted to the queue so far has finished.
    void finish();
    // Throws (and clears) a pending uncaptured error or a device loss.
    void check();

    // Brackets operations whose validation errors are wanted synchronously:
    // pop returns the first error since push, or "" for none.
    void pushErrorScope();
    std::string popErrorScope();

private:
    void open();
    void release();
    static void onUncapturedError(WGPUDevice const *, WGPUErrorType type,
                                  WGPUStringView message, void *self, void *);
    static void onDeviceLost(WGPUDevice const *, WGPUDeviceLostReason reason,
                             WGPUStringView message, void *self, void *);

    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;
    std::string adapterName_;
    std::string uncapturedError_;
    std::string lost_;
};

class Buffer
{
public:
    enum Kind { Storage, Uniform };

    // bytes is rounded up to a multiple of 4
    Buffer(Context &context, std::size_t bytes, Kind kind = Storage);
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    WGPUBuffer handle() const { return buffer_; }
    std::size_t size() const { return size_; }

    void upload(const void *data, std::size_t bytes);
    // Copies the first bytes of the buffer back through a staging buffer.
    void download(void *data, std::size_t bytes);

    template <typename T> void upload(const std::vector<T> &v)
    {
        upload(v.data(), v.size() * sizeof(T));
    }
    template <typename T> std::vector<T> download(std::size_t count)
    {
        std::vector<T> v(count);
        download(v.data(), count * sizeof(T));
        return v;
    }

private:
    Context &context_;
    WGPUBuffer buffer_ = nullptr;
    std::size_t size_ = 0;
};

/*
** A WGSL compute pipeline with an automatic layout.  dispatch() binds the
** buffers to @group(0) @binding(0..n-1) in order.
*/
class Kernel
{
public:
    // Throws Error with the compiler messages (including line numbers) when
    // the WGSL doesn't compile.
    Kernel(Context &context, const char *wgsl, const char *entryPoint = "main");
    ~Kernel();
    Kernel(const Kernel &) = delete;
    Kernel &operator=(const Kernel &) = delete;

    void dispatch(const std::vector<const Buffer *> &bindings, uint32_t groupsX,
                  uint32_t groupsY = 1, uint32_t groupsZ = 1);

private:
    Context &context_;
    WGPUShaderModule module_ = nullptr;
    WGPUComputePipeline pipeline_ = nullptr;
};

} // namespace gpu

#endif /* GPU_CONTEXT_H */
