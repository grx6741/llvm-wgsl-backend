#pragma once

#include "llvm/IR/Function.h"
#include "llvm/Demangle/Demangle.h"

#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/type/manager.h"

namespace WGSL
{

class Backend
{
public:
    Backend();
    ~Backend();

    void RegisterFunction( const llvm::Function& F );

private:
    bool isKernelFunction( const llvm::Function& F );
    void registerAsNormalFunction( const llvm::Function& F );
    void registerAsKernelFunction( const llvm::Function& F );


    const tint::core::type::Type* getWGSLType( const llvm::Type* type );
    const std::string demangledName( const std::string& mangled_name );

private:
    tint::core::ir::Module m_Module;
    tint::core::ir::Builder m_Builder;
    tint::core::type::Manager& m_TypeManager;
    llvm::ItaniumPartialDemangler m_Demangler;
};

} // namespace WGSL
