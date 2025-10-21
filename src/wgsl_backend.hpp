#pragma once

#include "llvm/IR/Function.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/ADT/SmallSet.h"
#include <memory>

#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/type/manager.h"

#include "translator.hpp"
#include "globals.hpp"

namespace WGSL
{

class Backend
{
public:
    Backend();
    ~Backend();

    void RegisterFunction( const llvm::Function& F );

private:
    bool isEntryPoint( const llvm::Function& F );
    const llvm::Type* isArgUsedAsArray( const llvm::Value* val,
                                        llvm::SmallSet< llvm::Value*, 32 >& visited );

    void initializeMainFunction();

private:
    tint::core::ir::Module m_Module;
    tint::core::ir::Builder m_Builder;
    tint::SymbolTable m_SymbolTable;
    tint::core::type::Manager& m_TypeManager;
    llvm::ItaniumPartialDemangler m_Demangler;
    std::unique_ptr< Globals > m_Globals;

    tint::core::ir::Function* m_MainFunction;

    std::vector< std::unique_ptr< Translator > > m_Translators;
};

} // namespace WGSL
