#include "translator.hpp"
#include "util.hpp"

#include <cstdlib>
#include <llvm/IR/Instructions.h>

#include <cassert>

namespace WGSL
{

Translator::Translator( tint::core::ir::Module& M,
                        tint::core::ir::Builder& B,
                        tint::SymbolTable& ST,
                        const llvm::Function* F,
                        bool isEntry )
    : m_LLVMfunc{ F },
      m_Module{ M },
      m_Builder{ B },
      m_SymbolTable{ ST },
      m_IsEntry{ isEntry },
      m_GroupCounter{ 0 },
      m_BindingCounter{ 0 }
{
    if ( !m_LLVMfunc ) {
        LOG_ERROR << "Passed LLVM Function is NULL\n";
        return;
    }
    const auto demangled_name = getDemangledName( m_LLVMfunc->getName().str() );

    // m_WGSLfunc = m_Builder.Function(
    //     demangled_name, MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() ) );

    m_WGSLfunc =
        m_IsEntry
            ? m_Builder.ComputeFunction(
                  demangled_name, tint::core::i32( 1 ), tint::core::i32( 1 ), tint::core::i32( 1 ) )
            : m_Builder.Function( demangled_name,
                                  MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() ) );

    LOG_INFO << ( m_IsEntry ? "Entry" : "Normal" ) << " Function :: " << demangled_name
             << ( m_IsEntry ? ""
                            : " -> " + MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() )
                                           ->FriendlyName() )
             << LOG_END;
}

void Translator::AddFunctionBuiltinParam( const llvm::Value* llvm_param,
                                          const tint::core::BuiltinValue param )
{
    // if ( !m_WGSLfunc->IsCompute() )
    //     return;

    const tint::core::type::Type* type;
    switch ( param ) {
        case tint::core::BuiltinValue::kLocalInvocationId:
        case tint::core::BuiltinValue::kWorkgroupId:
        case tint::core::BuiltinValue::kNumWorkgroups:
            type = m_Module.Types().vec3( m_Module.Types().u32() );
            break;
        default:
            type = m_Module.Types().invalid();
            break;
    }

    const auto& name = tint::core::ToString( param );

    if ( !type->Is< tint::core::type::Invalid >() ) {
        AddFunctionParam( name, llvm_param, type, param );
    }
}

void Translator::AddFunctionParam( const std::string_view& name,
                                   const llvm::Value* llvm_param,
                                   const tint::core::type::Type* type,
                                   tint::core::BuiltinValue builtin_type )
{
    if ( m_IsEntry ) {
        if ( type->Is< tint::core::type::Array >() ) {
            LOG_INFO << "    Param< " << type->FriendlyName() << " > " << name << LOG_END;
        }
        else {
            LOG_INFO << "    Param< " << type->FriendlyName() << " > " << name << LOG_END;
            m_StructParamMembers.Push(
                tint::core::type::Manager::StructMemberDesc{ m_SymbolTable.New( name ), type } );
        }
    }
    else {
        LOG_INFO << "    Param< " << type->FriendlyName() << " > " << name << LOG_END;
        auto* param = m_Builder.FunctionParam( name, type );
        if ( builtin_type != tint::core::BuiltinValue::kUndefined )
            param->SetBuiltin( builtin_type );

        m_FunctionParams.Push( param );

        m_ValueMap[llvm_param] = param;
    }
}

void Translator::Translate()
{
    m_IsEntry ? translateKernelFunction() : translateNormalFunction();
}

// Private Methods

const std::string Translator::getDemangledName( const std::string& mangled_name )
{
    if ( m_Demangler.partialDemangle( mangled_name.c_str() ) ) {
        return mangled_name;
    }

    return m_Demangler.getFunctionBaseName( nullptr, nullptr );
}

const tint::core::type::Type* Translator::MapLLVMtype2WGSLtype( tint::core::ir::Module& M,
                                                                const llvm::Type* type )
{
    auto& types = M.Types();

    auto llvm_type_id = type->getTypeID();
    switch ( llvm_type_id ) {
        case llvm::Type::IntegerTyID:
            return types.i32();
        case llvm::Type::FloatTyID:
            return types.f32();
        case llvm::Type::VoidTyID:
            return types.void_();
        default:
            return types.invalid();
    }

    return types.invalid();
}

const llvm::Type* Translator::IsArgUsedAsArray( const llvm::Value* val,
                                                llvm::SmallSet< llvm::Value*, 32 >& visited )
{
    if ( visited.count( val ) )
        return nullptr;

    // First Check if the argument is used as an array
    // directly
    for ( const auto& U : val->users() ) {
        // If the user is a GEP instruction, then the
        // argument is used as an array
        if ( llvm::isa< llvm::GetElementPtrInst >( U ) ) {
            const auto* GEP = llvm::cast< llvm::GetElementPtrInst >( U );
            return GEP->getResultElementType();
        }

        // Or if the result of this instruction is used in a
        // GEP instruction, then the argument is used as an
        // array
        if ( const auto* type = IsArgUsedAsArray( U, visited ) )
            return type;
    }

    return nullptr;
}

tint::core::ir::Value* Translator::getOperand( const llvm::Instruction& I, int op_idx )
{
    if ( op_idx < 0 || op_idx >= I.getNumOperands() ) {
        LOG_ERROR << "Invalid Operand Index when executing " << I << LOG_END;
        return nullptr;
    }

    if ( !m_ValueMap.count( I.getOperand( op_idx ) ) ) {
        LOG_ERROR << "Operand " << op_idx << " not found when executing" << I << LOG_END;
        return nullptr;
    }

    return m_ValueMap.at( I.getOperand( op_idx ) );
}

void Translator::translateKernelFunction()
{
    if ( m_StructParamMembers.Length() > 0 ) {
        auto* struct_param_t = m_Module.Types().Struct(
            m_SymbolTable.New( m_Module.NameOf( m_WGSLfunc ).to_str() + "_struct_param_t" ),
            m_StructParamMembers );

        auto* uniform_struct_param = m_Builder.Var(
            std::string( "uniform_" ) + m_Module.NameOf( m_WGSLfunc ).to_str() + "params",
            m_Module.Types().ptr( tint::core::AddressSpace::kUniform, struct_param_t ) );

        uniform_struct_param->SetBindingPoint( m_GroupCounter, m_BindingCounter++ );

        m_Module.root_block->Append( uniform_struct_param );
    }

    // translateFunctionBody();
}

void Translator::translateNormalFunction()
{
    if ( !m_FunctionParams.IsEmpty() )
        m_WGSLfunc->SetParams( m_FunctionParams );

    translateFunctionBody();
}

void Translator::translateFunctionBody()
{
    const auto func_body = [&] {
        for ( const llvm::BasicBlock& BB : *m_LLVMfunc ) {
            for ( const llvm::Instruction& I : BB ) {
#define VISIT_INST( inst )                                                                         \
    case llvm::Instruction::inst:                                                                  \
        visit##inst( I );                                                                          \
        break;
                switch ( I.getOpcode() ) {
                    VISIT_INST( FAdd );
                    VISIT_INST( FMul );
                    VISIT_INST( Ret );
                    VISIT_INST( Alloca );
                    VISIT_INST( Store );
                    VISIT_INST( Call );
                    default:
                        LOG_WARN << I.getOpcodeName() << " instruction NOT IMPLEMENTED \n";
                }
#undef VISIT_INST
            }
        }
    };

    m_Builder.Append( m_WGSLfunc->Block(), func_body );
}

// Intruction Visitors
void Translator::visitFAdd( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Add( type, lhs, rhs )->Result();
}

void Translator::visitFMul( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Multiply( type, lhs, rhs )->Result();
}

void Translator::visitRet( const llvm::Instruction& I )
{
    if ( I.getNumOperands() == 0 ) {
        // void return
        LOG_INFO << "    Returning void\n";
        m_Builder.Return( m_WGSLfunc );
        return;
    }

    llvm::Value* ret_val = I.getOperand( 0 );

    if ( !m_ValueMap.count( ret_val ) ) {
        LOG_INFO << "    Returning no arg";
        m_Builder.Return( m_WGSLfunc );
        return;
    }

    // Look up or create the WGSL equivalent of the return value
    // auto* wgsl_val =
    //     ctx.value_map.count( ret_val )
    //         ? ctx.value_map.at( ret_val )
    //         : mapLLVMconstant2WGSLconstant( ctx, ret_val ); // You need a
    //         helper for constants

    LOG_INFO << "    Returning with arg" << ret_val << LOG_END;
    m_Builder.Return( m_WGSLfunc, getOperand( I, 0 ) );
}

void Translator::visitAlloca( const llvm::Instruction& I )
{
    const auto* inst = llvm::dyn_cast< llvm::AllocaInst >( &I );
    const auto* alloca_type = inst->getAllocatedType();

    const auto& var_name = I.getName().str();

    auto* wgsl_type = MapLLVMtype2WGSLtype( m_Module, alloca_type );

    auto* wgsl_var = m_Builder.Var(
        var_name, m_Module.Types().ptr( tint::core::AddressSpace::kFunction, wgsl_type ) );

    m_ValueMap[&I] = wgsl_var->Result();
    LOG_INFO << "    Alloca: " << var_name << " : " << wgsl_type->FriendlyName() << LOG_END;
}


void Translator::visitStore( const llvm::Instruction& I )
{}


void Translator::visitCall( const llvm::Instruction& I )
{
    const auto* inst = llvm::dyn_cast< llvm::CallInst >( &I );
    const auto* callee = inst->getCalledFunction();

    std::string_view callee_name = callee->getName();

    // TODO
}

} // namespace WGSL
