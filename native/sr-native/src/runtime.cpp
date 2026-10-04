// Device enumeration and context lifetime.

#include "sr_internal.h"

#include <cstring>
#include <mutex>

#include "gpu.h"

namespace {

// ncnn reports the device class as a plain int, documented in gpu.h as
// "0 = discrete gpu, 1 = integrated gpu". Mapped explicitly rather than passed
// through, because the ABI's numbering is this project's to define and a
// reordering upstream must not silently change what the caller sees.
int32_t device_type_code(int type)
{
    switch (type)
    {
    case 0:
        return SR_DEVICE_DISCRETE;
    case 1:
        return SR_DEVICE_INTEGRATED;
    case 2:
        return SR_DEVICE_VIRTUAL;
    case 3:
        return SR_DEVICE_CPU;
    default:
        return SR_DEVICE_OTHER;
    }
}

// Vulkan packs a version into one word: 10 bits major, 10 minor, 12 patch.
void format_version(uint32_t packed, char* out, size_t size)
{
    snprintf(out,
             size,
             "%u.%u.%u",
             (packed >> 22) & 0x3FFu,
             (packed >> 12) & 0x3FFu,
             packed & 0xFFFu);
}

void copy_truncated(char* dst, size_t size, const char* src)
{
    if (size == 0)
        return;
    if (!src)
    {
        dst[0] = '\0';
        return;
    }
    std::strncpy(dst, src, size - 1);
    dst[size - 1] = '\0';
}

} // namespace

void sr_ensure_instance()
{
    static std::once_flag once;
    std::call_once(once, []() { ncnn::create_gpu_instance(); });
}

void sr_set_error(sr_context* ctx, const char* format, ...)
{
    if (!ctx)
        return;
    char buffer[512];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    ctx->last_error = buffer;
}

int sr_fail(sr_context* ctx, int code, const char* format, ...)
{
    if (ctx)
    {
        char buffer[512];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        ctx->last_error = buffer;
    }
    return code;
}

int sr_device_count(void)
{
    sr_ensure_instance();
    return ncnn::get_gpu_count();
}

int sr_device_info(int32_t index, sr_device_info_t* out)
{
    if (!out)
        return SR_ERR_INVALID_ARGUMENT;
    if (out->struct_size != sizeof(sr_device_info_t))
        return SR_ERR_INVALID_ARGUMENT;

    sr_ensure_instance();
    if (index < 0 || index >= ncnn::get_gpu_count())
        return SR_ERR_NO_DEVICE;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info(index);

    out->index = index;
    copy_truncated(out->name, sizeof(out->name), info.device_name());
    copy_truncated(out->driver_version, sizeof(out->driver_version), info.driver_name());
    format_version(info.api_version(), out->api_version, sizeof(out->api_version));
    out->vendor_id = info.vendor_id();
    out->device_id = info.device_id();
    out->device_type = device_type_code(info.type());
    out->unified_memory = (out->device_type == SR_DEVICE_INTEGRATED) ? 1 : 0;

    // The heap budget is what the driver will let *this process* allocate, which
    // is the number the planner wants. It is 0 when the driver will not say, and
    // that is a real answer rather than an error.
    uint64_t budget = 0;
    {
        ncnn::VulkanDevice* device = ncnn::get_gpu_device(index);
        if (device)
            budget = device->get_heap_budget();
    }
    out->budget_mib = budget;
    out->total_mib = budget;

    return SR_OK;
}

sr_context* sr_context_create(int32_t device_index)
{
    sr_ensure_instance();

    if (device_index < 0 || device_index >= ncnn::get_gpu_count())
        return nullptr;

    sr_context* ctx = new sr_context();
    ctx->device_index = device_index;
    return ctx;
}

void sr_context_destroy(sr_context* ctx)
{
    delete ctx;
}

const char* sr_last_error(sr_context* ctx)
{
    if (!ctx)
        return "";
    return ctx->last_error.c_str();
}
