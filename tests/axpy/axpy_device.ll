; ModuleID = 'tests/axpy/axpy.cu'
source_filename = "tests/axpy/axpy.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind optsize willreturn memory(argmem: readwrite)
define dso_local void @_Z11axpy_kernelifPfS_(i32 noundef %0, float noundef %1, ptr nocapture noundef readonly %2, ptr nocapture noundef %3) local_unnamed_addr #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %7 = mul i32 %5, %6
  %8 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %9 = add i32 %7, %8
  %10 = icmp slt i32 %9, %0
  br i1 %10, label %11, label %19

11:                                               ; preds = %4
  %12 = sext i32 %9 to i64
  %13 = getelementptr inbounds float, ptr %2, i64 %12
  %14 = load float, ptr %13, align 4, !tbaa !5
  %15 = fmul contract float %14, %1
  %16 = getelementptr inbounds float, ptr %3, i64 %12
  %17 = load float, ptr %16, align 4, !tbaa !5
  %18 = fadd contract float %15, %17
  store float %18, ptr %16, align 4, !tbaa !5
  br label %19

19:                                               ; preds = %11, %4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind optsize willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1, !2, !3}
!llvm.ident = !{!4}

!0 = !{ptr @_Z11axpy_kernelifPfS_, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
