// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/hip/tensor/slice_impl.h"

namespace onnxruntime {
namespace hip {
#define REGISTER_VERSIONED_TYPED_SLICE(TIND)                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      1, 9,                                                             \
      TIND,                                                             \
      kHipExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<false>);

REGISTER_VERSIONED_TYPED_SLICE(int32_t)
REGISTER_VERSIONED_TYPED_SLICE(int64_t)
REGISTER_VERSIONED_TYPED_SLICE(float)

#define REGISTER_V10_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                              \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      10, 10,                                                           \
      TIND,                                                             \
      kHipExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V10_TYPED_SLICE(int32_t)
REGISTER_V10_TYPED_SLICE(int64_t)
REGISTER_V10_TYPED_SLICE(float)

#define REGISTER_V11_TYPED_SLICE(TIND)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      Slice,                                                            \
      kOnnxDomain,                                                      \
      11,                                                               \
      TIND,                                                             \
      kHipExecutionProvider,                                           \
      KernelDefBuilder()                                                \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                       \
          .InputMemoryType<OrtMemTypeCPUInput>(4)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      Slice<true>);

REGISTER_V11_TYPED_SLICE(int32_t)
REGISTER_V11_TYPED_SLICE(int64_t)
REGISTER_V11_TYPED_SLICE(float)

template <bool dynamic>
Status Slice<dynamic>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = GetSlicedOrUnslicedTensor(ctx);

  ORT_ENFORCE(nullptr != input_tensor);

  auto& input_dimensions = input_tensor->Shape().GetDims();

  // Initialize the starts & ends to the actual tensor shape
  size_t dimension_count = input_dimensions.size();
  std::vector<int64_t> starts(dimension_count, 0);
  std::vector<int64_t> steps(dimension_count, 1);
  std::vector<int64_t> output_dims(input_dimensions);
  std::vector<int64_t> flattened_output_dims;
  std::vector<int64_t>* p_flattened_output_dims = &flattened_output_dims;

  if (dynamic) {
    std::vector<int64_t> input_starts, input_ends, input_axes, input_steps;
    FillInputVectors(ctx, input_starts, input_ends, input_axes, input_steps);
    ORT_RETURN_IF_ERROR(PrepareForCompute(input_starts, input_ends, input_axes,
                                          input_steps, input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));

  } else {
    ORT_RETURN_IF_ERROR(PrepareForCompute(attr_starts_, attr_ends_, attr_axes_,
                                          input_dimensions, starts, steps, output_dims,
                                          p_flattened_output_dims));
  }

  // if we are able to flatten the output dims we updated 'starts' and 'steps' to match the smaller number of dims.
  // update dimension_count to match.
  if (p_flattened_output_dims != nullptr) {
    dimension_count = flattened_output_dims.size();
  }

  //TArray<int64_t> starts_buffer(starts);
  HipAsyncBuffer<int64_t> starts_buffer(this, starts.size());
  gsl::span<int64_t> starts_buffer_span = starts_buffer.CpuSpan();
  for (auto i = 0; i < starts.size(); ++i) {
    starts_buffer_span[i] = starts[i];
  }
  starts_buffer.CopyToGpu();

  //TArray<int64_t> steps_buffer(steps);
  HipAsyncBuffer<int64_t> steps_buffer(this, steps.size());
  gsl::span<int64_t> steps_buffer_span = steps_buffer.CpuSpan();
  for (auto i = 0; i < steps.size(); ++i) {
    steps_buffer_span[i] = steps[i];
  }
  steps_buffer.CopyToGpu();

  // TArray<int64_t> input_strides(gsl::narrow_cast<int32_t>(dimension_count));
  // const gsl::span<int64_t> input_strides_span = gsl::make_span(input_strides.data_, input_strides.size_);
  HipAsyncBuffer<int64_t> input_strides(this, dimension_count);
  const gsl::span<int64_t> input_strides_span = input_strides.CpuSpan();
  if (p_flattened_output_dims != nullptr) {
    // we were able to flatten the innermost dimensions as they're being copied in full to the output.
    // do the same flattening to the innermost input dimensions in order to calculate pitches that match
    // the flattened output dimensions.
    int64_t aggregated_last_dim = 1;
    for (size_t i = dimension_count - 1, end = input_dimensions.size(); i < end; ++i) {
      aggregated_last_dim *= input_dimensions[i];
    }

    auto flattened_input_dims(input_dimensions);
    flattened_input_dims.resize(dimension_count);
    flattened_input_dims.back() = aggregated_last_dim;
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, flattened_input_dims));
  } else {
    ORT_ENFORCE(TensorPitches::Calculate(input_strides_span, input_dimensions));
  }
  input_strides.CopyToGpu();

  TensorPitches original_output_strides(p_flattened_output_dims != nullptr ? flattened_output_dims : output_dims);
  //TArray<fast_divmod> output_strides(gsl::narrow_cast<int32_t>(original_output_strides.size()));
  HipAsyncBuffer<fast_divmod> output_strides(this, dimension_count);
  gsl::span<fast_divmod> output_strides_span = output_strides.CpuSpan();
  for (int32_t i = 0; i < static_cast<int32_t>(original_output_strides.size()); ++i) {
    output_strides_span[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
  }
  output_strides.CopyToGpu();

  size_t element_size = input_tensor->DataType()->Size();

  ORT_RETURN_IF_ERROR(CallSliceImp(element_size,
                                   gsl::narrow_cast<int32_t>(dimension_count),
                                   starts_buffer.GpuPtr(),
                                   steps_buffer.GpuPtr(),
                                   input_strides.GpuPtr(),
                                   output_strides.GpuPtr(),
                                   ctx,
                                   TensorShape(output_dims)));

  return Status::OK();
}

template <bool dynamic>
const Tensor* Slice<dynamic>::GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const {
  return ctx->Input<Tensor>(0);
}

template <bool dynamic>
void Slice<dynamic>::FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts,
                                      std::vector<int64_t>& input_ends, std::vector<int64_t>& input_axes,
                                      std::vector<int64_t>& input_steps) const {
  FillVectorsFromInput(*ctx->Input<Tensor>(1), *ctx->Input<Tensor>(2), ctx->Input<Tensor>(3),
                       ctx->Input<Tensor>(4), input_starts, input_ends, input_axes, input_steps);
}

template <bool dynamic>
Status Slice<dynamic>::CallSliceImp(size_t element_size, size_t dimension_count, const int64_t* starts_buffer,
                                    const int64_t* steps_buffer, const int64_t* input_strides,
                                    const fast_divmod* output_strides, OpKernelContext* ctx,
                                    TensorShape output_shape) const {
  auto* output_tensor = ctx->Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  return SliceImpl(element_size,
                   gsl::narrow_cast<int32_t>(dimension_count),
                   starts_buffer,
                   steps_buffer,
                   input_strides,
                   output_strides,
                   ctx->Input<Tensor>(0)->DataRaw(),
                   output_tensor->MutableDataRaw(),
                   output_shape.Size());
}

}  // namespace hip
}  // namespace onnxruntime
