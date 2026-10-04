// Gate 1: the runtime can see a Vulkan device and open a context on it.
//
// This is deliberately the first gate, because every later failure — an
// unreadable model, a wrong tensor shape, a driver that cannot allocate — looks
// the same from the outside ("the model produced nothing") until this one passes.

#include "sr_native.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
    if (condition)
    {
        printf("  ok    %s\n", what);
    }
    else
    {
        printf("  FAIL  %s\n", what);
        failures++;
    }
}

const char* type_name(int32_t code)
{
    switch (code)
    {
    case SR_DEVICE_DISCRETE:
        return "discrete";
    case SR_DEVICE_INTEGRATED:
        return "integrated";
    case SR_DEVICE_VIRTUAL:
        return "virtual";
    case SR_DEVICE_CPU:
        return "cpu";
    default:
        return "other";
    }
}

} // namespace

int main()
{
    printf("sr-native device_test\n");

    const int count = sr_device_count();
    printf("  devices: %d\n", count);
    check(count >= 1, "at least one Vulkan device is visible");

    if (count < 1)
    {
        printf("\nFAILED: no Vulkan device. ncnn needs a working Vulkan driver.\n");
        return 1;
    }

    for (int i = 0; i < count; i++)
    {
        sr_device_info_t info;
        memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(sr_device_info_t);

        const int rc = sr_device_info(i, &info);
        if (rc != SR_OK)
        {
            printf("  FAIL  sr_device_info(%d) returned %d\n", i, rc);
            failures++;
            continue;
        }

        printf("  [%d] %s  vendor=0x%04x device=0x%04x type=%s api=%s driver=%s budget=%llu MiB%s\n",
               info.index,
               info.name,
               info.vendor_id,
               info.device_id,
               type_name(info.device_type),
               info.api_version,
               info.driver_version,
               (unsigned long long)info.budget_mib,
               info.unified_memory ? " (unified memory)" : "");
    }

    // A device row with no name is a row a user cannot act on, and it is also
    // what a half-initialised Vulkan instance returns.
    sr_device_info_t first;
    memset(&first, 0, sizeof(first));
    first.struct_size = sizeof(sr_device_info_t);
    check(sr_device_info(0, &first) == SR_OK, "device 0 reports its properties");
    check(first.name[0] != '\0', "device 0 has a name");

    // The budget is what the planner sizes against. It may legitimately be 0 on
    // an integrated part whose memory is shared, but on a discrete card it is
    // the number the whole VRAM policy rests on, so it is asserted there.
    if (first.device_type == SR_DEVICE_DISCRETE)
        check(first.budget_mib > 0, "a discrete device reports a heap budget");

    check(sr_device_info(count, &first) == SR_ERR_NO_DEVICE, "an out-of-range index is refused");

    sr_context* ctx = sr_context_create(0);
    check(ctx != nullptr, "a context opens on device 0");
    if (ctx)
    {
        check(sr_last_error(ctx)[0] == '\0', "a fresh context has no error recorded");
        sr_context_destroy(ctx);
    }

    check(sr_context_create(-1) == nullptr, "a negative device index is refused");
    check(sr_context_create(count) == nullptr, "an out-of-range device index is refused");

    if (failures)
    {
        printf("\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASSED\n");
    return 0;
}
