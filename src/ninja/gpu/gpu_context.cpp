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

#include "gpu_context.h"

#include <cstring>

namespace gpu {

namespace {

WGPUStringView view(const char *s)
{
    WGPUStringView v = WGPU_STRING_VIEW_INIT;
    v.data = s;
    v.length = WGPU_STRLEN;
    return v;
}

std::string str(WGPUStringView v)
{
    if(v.data == nullptr)
        return std::string();
    if(v.length == WGPU_STRLEN)
        return std::string(v.data);
    return std::string(v.data, v.length);
}

struct AdapterRequest
{
    WGPUAdapter adapter = nullptr;
    std::string message;
};

void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
               WGPUStringView message, void *request, void *)
{
    AdapterRequest *r = static_cast<AdapterRequest *>(request);
    if(status == WGPURequestAdapterStatus_Success)
        r->adapter = adapter;
    r->message = str(message);
}

struct DeviceRequest
{
    WGPUDevice device = nullptr;
    std::string message;
};

void onDevice(WGPURequestDeviceStatus status, WGPUDevice device,
              WGPUStringView message, void *request, void *)
{
    DeviceRequest *r = static_cast<DeviceRequest *>(request);
    if(status == WGPURequestDeviceStatus_Success)
        r->device = device;
    r->message = str(message);
}

void onPopErrorScope(WGPUPopErrorScopeStatus status, WGPUErrorType type,
                     WGPUStringView message, void *out, void *)
{
    std::string *s = static_cast<std::string *>(out);
    if(status != WGPUPopErrorScopeStatus_Success)
        *s = "error scope could not be popped: " + str(message);
    else if(type != WGPUErrorType_NoError)
        *s = str(message);
}

void onMapped(WGPUMapAsyncStatus status, WGPUStringView message, void *out,
              void *)
{
    if(status != WGPUMapAsyncStatus_Success)
        *static_cast<std::string *>(out) = str(message);
}

void onWorkDone(WGPUQueueWorkDoneStatus, WGPUStringView, void *, void *)
{
}

struct CompileMessages
{
    std::string errors;
};

void onCompilationInfo(WGPUCompilationInfoRequestStatus status,
                       WGPUCompilationInfo const *info, void *out, void *)
{
    CompileMessages *m = static_cast<CompileMessages *>(out);
    if(status != WGPUCompilationInfoRequestStatus_Success || info == nullptr)
        return;
    for(size_t i = 0; i < info->messageCount; i++)
    {
        const WGPUCompilationMessage &msg = info->messages[i];
        if(msg.type != WGPUCompilationMessageType_Error)
            continue;
        m->errors += "line " + std::to_string(msg.lineNum) + ":" +
                     std::to_string(msg.linePos) + ": " + str(msg.message) +
                     "\n";
    }
}

} // namespace

Context::Context()
{
    try
    {
        open();
    }
    catch(...)
    {
        release();
        throw;
    }
}

Context::~Context()
{
    release();
}

void Context::open()
{
    WGPUInstanceFeatureName timedWaitAny = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceDescriptor instanceDesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = &timedWaitAny;
    instance_ = wgpuCreateInstance(&instanceDesc);
    if(instance_ == nullptr)
        throw Unavailable("WebGPU: could not create an instance");

    WGPUPowerPreference preferences[] = {WGPUPowerPreference_HighPerformance,
                                         WGPUPowerPreference_Undefined};
    AdapterRequest adapterRequest;
    for(WGPUPowerPreference preference : preferences)
    {
        WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        options.powerPreference = preference;
        WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_WaitAnyOnly;
        cb.callback = onAdapter;
        cb.userdata1 = &adapterRequest;
        wait(wgpuInstanceRequestAdapter(instance_, &options, cb));
        if(adapterRequest.adapter != nullptr)
            break;
    }
    if(adapterRequest.adapter == nullptr)
        throw Unavailable("WebGPU: no adapter: " + adapterRequest.message);
    adapter_ = adapterRequest.adapter;

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if(wgpuAdapterGetInfo(adapter_, &info) == WGPUStatus_Success)
    {
        // browsers usually report only the vendor and architecture
        adapterName_ = str(info.device);
        if(adapterName_.empty())
            adapterName_ = str(info.description);
        if(adapterName_.empty())
            adapterName_ = str(info.vendor) + " " + str(info.architecture);
        wgpuAdapterInfoFreeMembers(info);
    }

    WGPUDeviceDescriptor deviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
    deviceDesc.label = view("windninja");
    deviceDesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    deviceDesc.deviceLostCallbackInfo.callback = onDeviceLost;
    deviceDesc.deviceLostCallbackInfo.userdata1 = this;
    deviceDesc.uncapturedErrorCallbackInfo.callback = onUncapturedError;
    deviceDesc.uncapturedErrorCallbackInfo.userdata1 = this;
    DeviceRequest deviceRequest;
    WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = onDevice;
    cb.userdata1 = &deviceRequest;
    wait(wgpuAdapterRequestDevice(adapter_, &deviceDesc, cb));
    if(deviceRequest.device == nullptr)
        throw Unavailable("WebGPU: no device: " + deviceRequest.message);
    device_ = deviceRequest.device;
    queue_ = wgpuDeviceGetQueue(device_);
}

void Context::release()
{
    if(queue_ != nullptr)
        wgpuQueueRelease(queue_);
    if(device_ != nullptr)
        wgpuDeviceRelease(device_);
    if(adapter_ != nullptr)
        wgpuAdapterRelease(adapter_);
    if(instance_ != nullptr)
        wgpuInstanceRelease(instance_);
    queue_ = nullptr;
    device_ = nullptr;
    adapter_ = nullptr;
    instance_ = nullptr;
}

void Context::wait(WGPUFuture future)
{
    WGPUFutureWaitInfo info = WGPU_FUTURE_WAIT_INFO_INIT;
    info.future = future;
    WGPUWaitStatus status = wgpuInstanceWaitAny(instance_, 1, &info, UINT64_MAX);
    if(status != WGPUWaitStatus_Success)
        throw Error("WebGPU: waiting on the GPU failed");
    check();
}

void Context::finish()
{
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = onWorkDone;
    wait(wgpuQueueOnSubmittedWorkDone(queue_, cb));
}

void Context::check()
{
    if(!lost_.empty())
        throw Error("WebGPU device lost: " + lost_);
    if(!uncapturedError_.empty())
    {
        std::string message = uncapturedError_;
        uncapturedError_.clear();
        throw Error("WebGPU error: " + message);
    }
}

void Context::pushErrorScope()
{
    wgpuDevicePushErrorScope(device_, WGPUErrorFilter_Validation);
}

std::string Context::popErrorScope()
{
    std::string message;
    WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = onPopErrorScope;
    cb.userdata1 = &message;
    wait(wgpuDevicePopErrorScope(device_, cb));
    return message;
}

void Context::onUncapturedError(WGPUDevice const *, WGPUErrorType,
                                WGPUStringView message, void *self, void *)
{
    Context *c = static_cast<Context *>(self);
    if(c->uncapturedError_.empty())
        c->uncapturedError_ = str(message);
}

void Context::onDeviceLost(WGPUDevice const *, WGPUDeviceLostReason reason,
                           WGPUStringView message, void *self, void *)
{
    // Destroyed / CallbackCancelled come from our own teardown
    if(reason == WGPUDeviceLostReason_Destroyed ||
       reason == WGPUDeviceLostReason_CallbackCancelled)
        return;
    static_cast<Context *>(self)->lost_ = str(message);
}

Buffer::Buffer(Context &context, std::size_t bytes, Kind kind)
    : context_(context), size_((bytes + 3) / 4 * 4)
{
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                 (kind == Uniform ? WGPUBufferUsage_Uniform
                                  : WGPUBufferUsage_Storage);
    desc.size = size_;
    context_.pushErrorScope();
    buffer_ = wgpuDeviceCreateBuffer(context_.device(), &desc);
    std::string error = context_.popErrorScope();
    if(!error.empty())
    {
        wgpuBufferRelease(buffer_);
        throw Error("WebGPU buffer of " + std::to_string(size_) +
                    " bytes: " + error);
    }
}

Buffer::~Buffer()
{
    wgpuBufferDestroy(buffer_);
    wgpuBufferRelease(buffer_);
}

void Buffer::upload(const void *data, std::size_t bytes, std::size_t offset)
{
    if(offset % 4 != 0 || offset + bytes > size_)
        throw Error("WebGPU upload outside the buffer");
    // writeBuffer needs a multiple of 4 bytes; pad the tail with zeros
    std::size_t whole = bytes / 4 * 4;
    if(whole > 0)
        wgpuQueueWriteBuffer(context_.queue(), buffer_, offset, data, whole);
    if(whole < bytes)
    {
        unsigned char tail[4] = {0, 0, 0, 0};
        std::memcpy(tail, static_cast<const unsigned char *>(data) + whole,
                    bytes - whole);
        wgpuQueueWriteBuffer(context_.queue(), buffer_, offset + whole, tail, 4);
    }
}

void Buffer::download(void *data, std::size_t bytes, std::size_t offset)
{
    if(offset % 4 != 0 || offset + bytes > size_)
        throw Error("WebGPU download outside the buffer");
    std::size_t padded = (bytes + 3) / 4 * 4;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    desc.size = padded;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(context_.device(), &desc);

    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(context_.device(), nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer_, offset, staging, 0, padded);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(context_.queue(), 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    std::string error;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = onMapped;
    cb.userdata1 = &error;
    try
    {
        context_.wait(wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, padded, cb));
    }
    catch(...)
    {
        wgpuBufferRelease(staging);
        throw;
    }
    if(error.empty())
    {
        std::memcpy(data, wgpuBufferGetConstMappedRange(staging, 0, padded), bytes);
        wgpuBufferUnmap(staging);
    }
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
    if(!error.empty())
        throw Error("WebGPU readback: " + error);
}

Kernel::Kernel(Context &context, const char *wgsl, const char *entryPoint)
    : context_(context)
{
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = view(wgsl);
    WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    moduleDesc.nextInChain = &source.chain;
    context_.pushErrorScope();
    module_ = wgpuDeviceCreateShaderModule(context_.device(), &moduleDesc);
    std::string error = context_.popErrorScope();
    if(!error.empty())
    {
        // the compilation info carries line numbers in a fixed format
        CompileMessages messages;
        WGPUCompilationInfoCallbackInfo cb =
            WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_WaitAnyOnly;
        cb.callback = onCompilationInfo;
        cb.userdata1 = &messages;
        context_.wait(wgpuShaderModuleGetCompilationInfo(module_, cb));
        wgpuShaderModuleRelease(module_);
        throw Error("WGSL compile failed:\n" +
                    (messages.errors.empty() ? error : messages.errors));
    }

    WGPUComputePipelineDescriptor pipelineDesc =
        WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipelineDesc.compute.module = module_;
    pipelineDesc.compute.entryPoint = view(entryPoint);
    context_.pushErrorScope();
    pipeline_ = wgpuDeviceCreateComputePipeline(context_.device(), &pipelineDesc);
    error = context_.popErrorScope();
    if(!error.empty())
    {
        wgpuComputePipelineRelease(pipeline_);
        wgpuShaderModuleRelease(module_);
        throw Error(std::string("WebGPU compute pipeline '") + entryPoint +
                    "': " + error);
    }
}

Kernel::~Kernel()
{
    wgpuComputePipelineRelease(pipeline_);
    wgpuShaderModuleRelease(module_);
}

void Kernel::dispatch(const std::vector<const Buffer *> &bindings,
                      uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ)
{
    std::vector<WGPUBindGroupEntry> entries(bindings.size());
    for(size_t i = 0; i < bindings.size(); i++)
    {
        entries[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[i].binding = static_cast<uint32_t>(i);
        entries[i].buffer = bindings[i]->handle();
        entries[i].size = bindings[i]->size();
    }
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline_, 0);
    WGPUBindGroupDescriptor groupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    groupDesc.layout = layout;
    groupDesc.entryCount = entries.size();
    groupDesc.entries = entries.data();
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(context_.device(), &groupDesc);
    wgpuBindGroupLayoutRelease(layout);

    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(context_.device(), nullptr);
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline_);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, groupsX, groupsY, groupsZ);
    wgpuComputePassEncoderEnd(pass);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(context_.queue(), 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(group);
}

} // namespace gpu
