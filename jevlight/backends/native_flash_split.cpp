// Call the CUDA kernels shipped in the pinned PyTorch 2.8.0 wheel.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "native_flash_split_params.h"

struct CUstream_st;
namespace cutlass { struct bfloat16_t; }
template<int D, int M, int N, int W, class T> struct Flash_kernel_traits;
template<int D, int M, int N, int W, bool Q, bool K, class T, class Base>
struct Flash_fwd_kernel_traits;
namespace pytorch_flash {
template<class Traits, bool Causal>
void run_flash_splitkv_fwd(Flash_fwd_params&, CUstream_st*);
}

at::Tensor flash_split256(const at::Tensor& q, const at::Tensor& k,
                         const at::Tensor& v, int64_t splits, double scale,
                         int64_t stream) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4,
                "Native split Flash requires BF16 CUDA [batch,tokens,heads,256]");
    TORCH_CHECK(k.device()==q.device() && v.device()==q.device() &&
                k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type() &&
                k.dim()==4 && v.sizes()==k.sizes() && q.size(0)==k.size(0) &&
                q.size(1)==k.size(1) && q.size(0)>0 && q.size(1)>0 &&
                q.size(2)>0 && k.size(2)>0 && q.size(3)==256 && k.size(3)==256 &&
                q.size(2)%k.size(2)==0 && q.stride(3)==1 && k.stride(3)==1 && v.stride(3)==1,
                "Native split Flash received incompatible complete prefix rows");
    TORCH_CHECK(splits>1 && splits<=128,"Native split count is invalid");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"Native attention scale is invalid");
    const int b=q.size(0), s=q.size(1), h=q.size(2), hk=k.size(2);
    auto out=at::empty_like(q);
    auto fp=q.options().dtype(at::kFloat);
    auto lse=at::empty({b,h,s},fp);
    auto lse_accum=at::empty({splits,b,h,s},fp);
    auto out_accum=at::empty({splits,b,h,s,256},fp);
    pytorch_flash::Flash_fwd_params p{};
    p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();
    p.q_batch_stride=q.stride(0);p.k_batch_stride=k.stride(0);p.v_batch_stride=v.stride(0);p.o_batch_stride=out.stride(0);
    p.q_row_stride=q.stride(1);p.k_row_stride=k.stride(1);p.v_row_stride=v.stride(1);p.o_row_stride=out.stride(1);
    p.q_head_stride=q.stride(2);p.k_head_stride=k.stride(2);p.v_head_stride=v.stride(2);p.o_head_stride=out.stride(2);
    p.h=h;p.h_k=hk;p.h_h_k_ratio=h/hk;
    p.b=b;p.seqlen_q=s;p.seqlen_k=s;p.d=256;p.d_rounded=256;
    p.seqlen_q_rounded=((s+127)/128)*128;p.seqlen_k_rounded=((s+127)/128)*128;
    p.scale_softmax=static_cast<float>(scale);
    p.scale_softmax_log2=static_cast<float>(scale*1.4426950408889634074);
    p.p_dropout=1.f;p.p_dropout_in_uint8_t=255;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.window_size_left=-1;p.window_size_right=0;p.is_bf16=true;p.is_causal=true;p.is_seqlens_k_cumulative=true;
    p.num_splits=splits;p.softmax_lse_ptr=lse.data_ptr();
    p.softmax_lseaccum_ptr=lse_accum.data_ptr();p.oaccum_ptr=out_accum.data_ptr();
    using Base=Flash_kernel_traits<256,64,64,4,cutlass::bfloat16_t>;
    using Traits=Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::bfloat16_t,Base>;
    pytorch_flash::run_flash_splitkv_fwd<Traits,true>(p,reinterpret_cast<CUstream_st*>(stream));
    return out;
}

TORCH_LIBRARY(jevlight_native, m) {
    m.def("flash_split256(Tensor query, Tensor key, Tensor value, int splits, float scale, int stream) -> Tensor");
}
TORCH_LIBRARY_IMPL(jevlight_native, CUDA, m) {
    m.impl("flash_split256",TORCH_FN(flash_split256));
}
