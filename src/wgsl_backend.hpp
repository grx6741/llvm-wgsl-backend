#pragma once

#include "llvm/IR/Function.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/ADT/SmallSet.h"

#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/type/manager.h"

#include "translator.hpp"

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

    const llvm::Type* isArgUsedAsArray( const llvm::Value* val,
                                        llvm::SmallSet< llvm::Value*, 32 >& visited );

private:
    tint::core::ir::Module m_Module;
    tint::core::ir::Builder m_Builder;
    tint::SymbolTable m_SymbolTable;
    tint::core::type::Manager& m_TypeManager;
    llvm::ItaniumPartialDemangler m_Demangler;

    std::vector< Translator > m_Translators;
};

} // namespace WGSL
