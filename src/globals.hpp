#pragma once

#include "src/tint/lang/core/ir/function.h"
#include "src/tint/lang/core/ir/var.h"

#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/builder.h"
#include "src/tint/lang/core/type/manager.h"

#include <vector>

class Globals
{
public:
    Globals( tint::core::ir::Module& M, tint::core::ir::Builder& B, tint::SymbolTable& S );

    tint::core::ir::Function* GetIntrinsicAccessor( std::string_view llvm_name ) const;

    tint::core::ir::Var* GetIntrinsicsStruct() const
    {
        return m_Intrinsics;
    }

private:
    void createIntrinsicAccessors( tint::core::ir::Module& M,
                                   tint::core::ir::Builder& B,
                                   tint::SymbolTable& S );

    // component_index 0=x, 1=y, 2=z
    tint::core::ir::Function* createAccessor( tint::core::ir::Module& M,
                                              tint::core::ir::Builder& B,
                                              const std::string& name,
                                              int intrinsic_index,
                                              int component_index );

private:
    tint::core::ir::Var* m_Intrinsics;
    std::unordered_map< std::string_view, tint::core::ir::Function* > m_IntrinsicAccessors;
};
