#pragma once

#include "llvm/IR/Function.h"
#include "llvm/Demangle/Demangle.h"
#include <llvm/ADT/SmallSet.h>

#include "src/tint/lang/core/ir/builder.h"

namespace WGSL
{

class Translator
{
public:
    Translator( tint::core::ir::Module& M,
                tint::core::ir::Builder& B,
                tint::SymbolTable& ST,
                const llvm::Function* F,
                bool isKernel = false );

    static const tint::core::type::Type* MapLLVMtype2WGSLtype( tint::core::ir::Module& Module,
                                                               const llvm::Type* type );

    static const llvm::Type* IsArgUsedAsArray( const llvm::Value* val,
                                               llvm::SmallSet< llvm::Value*, 32 >& visited );

    void AddFunctionParam(
        const std::string_view& name,
        const llvm::Value* llvm_param,
        const tint::core::type::Type* type,
        tint::core::BuiltinValue builtin_type = tint::core::BuiltinValue::kUndefined );

    void AddFunctionBuiltinParam( const llvm::Value* llvm_param,
                                  const tint::core::BuiltinValue param );

    void Translate();

    inline const bool IsEntry()
    {
        return m_IsEntry;
    }

private:
    const std::string getDemangledName( const std::string& mangled_name );

    tint::core::ir::Value* getOperand( const llvm::Instruction& I, int op_idx );

    void translateKernelFunction();
    void translateNormalFunction();
    void translateFunctionBody();

    // Intruction Visitors
    void visitFAdd( const llvm::Instruction& I );
    void visitFMul( const llvm::Instruction& I );
    void visitRet( const llvm::Instruction& I );
    void visitAlloca( const llvm::Instruction& I );
    void visitStore( const llvm::Instruction& I );
    void visitCall( const llvm::Instruction& I );

private:
    std::unordered_map< const llvm::Value*, tint::core::ir::Value* > m_ValueMap;

    const llvm::Function* m_LLVMfunc;

    tint::core::ir::Function* m_WGSLfunc;
    tint::Vector< tint::core::ir::FunctionParam*, 3 > m_ComputeBuiltinParams;
    tint::Vector< tint::core::type::Manager::StructMemberDesc, 3 > m_StructParamMembers;
    tint::Vector< tint::core::ir::FunctionParam*, 3 > m_FunctionParams;

    bool m_IsEntry;

    uint32_t m_GroupCounter;
    uint32_t m_BindingCounter;

    tint::core::ir::Module& m_Module;
    tint::core::ir::Builder& m_Builder;
    tint::SymbolTable& m_SymbolTable;
    llvm::ItaniumPartialDemangler m_Demangler;
};

} // namespace WGSL
