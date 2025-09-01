#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Module.h"

#include "wgsl_backend.hpp"

namespace
{

struct LLVM2WGSL : llvm::PassInfoMixin< LLVM2WGSL >
{
    llvm::PreservedAnalyses run( llvm::Module& M, llvm::ModuleAnalysisManager& )
    {
        WGSL::Backend backend;

        // Go through all Functions that have a body
        for ( llvm::Function& F : M ) {
            if ( !F.isDeclaration() )
                backend.RegisterFunction( F );
        }

        return llvm::PreservedAnalyses::all();
    }

    static bool isRequired()
    {
        return true;
    }
};

} // namespace

llvm::PassPluginLibraryInfo getLLVM2WGSLPluginInfo()
{
    llvm::PassPluginLibraryInfo passInfo;

    passInfo.APIVersion = LLVM_PLUGIN_API_VERSION;
    passInfo.PluginName = "LLVM 2 WGSL";
    passInfo.PluginVersion = LLVM_VERSION_STRING;

    passInfo.RegisterPassBuilderCallbacks = []( llvm::PassBuilder& PB ) {
        auto pipelineParsingCallback = []( llvm::StringRef Name,
                                           llvm::ModulePassManager& MPM,
                                           llvm::ArrayRef< llvm::PassBuilder::PipelineElement > ) {
            if ( Name == "llvm2wgsl" ) {
                MPM.addPass( LLVM2WGSL() );
                return true;
            }
            return false;
        };

        PB.registerPipelineParsingCallback( pipelineParsingCallback );
    };

    return passInfo;
}

extern "C" {
LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo()
{
    return getLLVM2WGSLPluginInfo();
}
}
