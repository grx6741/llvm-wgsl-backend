#pragma once

#include "llvm/IR/Function.h"
#include "llvm/IR/Argument.h"
#include "llvm/Demangle/Demangle.h"
#include <llvm/ADT/SmallSet.h>

#include "src/tint/lang/core/ir/builder.h"

#include <unordered_map>
#include "globals.hpp"

namespace WGSL
{

class Translator
{
public:
    Translator( tint::core::ir::Module& M,
                tint::core::ir::Builder& B,
                tint::SymbolTable& ST,
                Globals& G,
                const llvm::Function* F,
                bool isKernel = false );

    static const tint::core::type::Type* MapLLVMtype2WGSLtype( tint::core::ir::Module& Module,
                                                               const llvm::Type* type );

    static const llvm::Type* IsArgUsedAsArray( const llvm::Value* val,
                                               llvm::SmallSet< llvm::Value*, 32 >& visited );

    void AddFunctionParam(
        const std::string_view& name,
        const llvm::Argument* llvm_param,
        const tint::core::type::Type* type,
        tint::core::BuiltinValue builtin_type = tint::core::BuiltinValue::kUndefined );

    void Translate();

    inline const bool IsEntry()
    {
        return m_IsEntry;
    }

    tint::core::ir::Function* GetWGSLFunc() const
    {
        return m_WGSLfunc;
    }

private:
    const std::string getDemangledName( const std::string& mangled_name );

    tint::core::ir::Value* getOperand( const llvm::Instruction& I, int op_idx );

    bool isArgReadOnly( const llvm::Argument* arg );
    bool isPointerWritten( const llvm::Argument* arg );

    void translateKernelFunction();
    void translateNormalFunction();
    void translateFunctionBody();

    // Intruction Visitors
    void visitFAdd( const llvm::Instruction& I );
    void visitFMul( const llvm::Instruction& I );
    void visitAdd( const llvm::Instruction& I );
    void visitMul( const llvm::Instruction& I );
    void visitRet( const llvm::Instruction& I );
    void visitCall( const llvm::Instruction& I );
    void visitICmp( const llvm::Instruction& I );
    void visitBr( const llvm::Instruction& I );

private:
    std::unordered_map< const llvm::Value*, tint::core::ir::Value* > m_ValueMap;

    const llvm::Function* m_LLVMfunc;

    tint::core::ir::Function* m_WGSLfunc;
    tint::Vector< tint::core::ir::FunctionParam*, 3 > m_ComputeBuiltinParams;

    std::unordered_map< const llvm::Value*, tint::core::type::Manager::StructMemberDesc >
        m_StructParamMembers;
    tint::Vector< tint::core::ir::FunctionParam*, 3 > m_FunctionParams;

    bool m_IsEntry;

    uint32_t m_GroupCounter;
    uint32_t m_BindingCounter;

    tint::core::ir::Module& m_Module;
    tint::core::ir::Builder& m_Builder;
    tint::SymbolTable& m_SymbolTable;
    Globals& m_Globals;

    llvm::ItaniumPartialDemangler m_Demangler;
    std::string m_DemangledName;
};

} // namespace WGSL
