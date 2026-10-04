// rife implemented with ncnn library
//
// Provenance: vendored from styler00dollar/VapourSynth-RIFE-ncnn-Vulkan,
// RIFE/rife_ops.h, blob 550610ec800f5745b9445c9156df17aac110274d (MIT).
// The pack8 pipeline member is dropped; see the note in warp.cpp.

#ifndef RIFE_OPS_H
#define RIFE_OPS_H

#include <vector>

// ncnn
#include "layer.h"
#include "pipeline.h"

class Warp : public ncnn::Layer
{
public:
    Warp();
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs, std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd, const ncnn::Option& opt) const;

private:
    ncnn::Pipeline* pipeline_warp;
    ncnn::Pipeline* pipeline_warp_pack4;
};

#endif // RIFE_OPS_H
