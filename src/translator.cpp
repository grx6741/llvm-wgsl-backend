#include "translator.hpp"
#include "util.hpp"

#include <cstdlib>
#include <llvm/IR/Instructions.h>

#include <cassert>

namespace WGSL
{

const std::unordered_map< std::string, int > WGSLintrinsicMap = {
    { "llvm.nvvm.read.ptx.sreg.ctaid.x", 1 },
    { "llvm.nvvm.read.ptx.sreg.ctaid.y", 1 },
    { "llvm.nvvm.read.ptx.sreg.ctaid.z", 1 } };


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
    m_DemangledName = getDemangledName( m_LLVMfunc->getName().str() );
    LOG_INFO << "LLVM function name: '" << m_LLVMfunc->getName().str() << "'" << LOG_END;
    LOG_INFO << "Demangled name: '" << m_DemangledName << "'" << LOG_END;

    m_WGSLfunc = m_Builder.Function(
        m_DemangledName, MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() ) );

    LOG_INFO << "WGSL function name after creation: '" << m_Module.NameOf( m_WGSLfunc ).to_str()
             << "'" << LOG_END;

    // m_WGSLfunc =
    //     m_IsEntry
    //         ? m_Builder.ComputeFunction(
    //               demangled_name, tint::core::i32( 1 ), tint::core::i32( 1 ), tint::core::i32( 1
    //               ) )
    //         : m_Builder.Function( demangled_name,
    //                               MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() )
    //                               );

    LOG_INFO << ( m_IsEntry ? "Entry" : "Normal" ) << " Function :: " << m_DemangledName
             << ( m_IsEntry ? ""
                            : " -> " + MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() )
                                           ->FriendlyName() )
             << LOG_END;
}

void Translator::AddFunctionParam( const std::string_view& name,
                                   const llvm::Argument* llvm_param,
                                   const tint::core::type::Type* type,
                                   tint::core::BuiltinValue builtin_type )
{
    if ( m_IsEntry ) {
        if ( type->Is< tint::core::type::Array >() ) {
            LOG_INFO << "    Param< " << type->FriendlyName() << " > " << name << LOG_END;

            auto* storage_ptr = m_Module.Types().ptr( tint::core::AddressSpace::kStorage,
                                                      type,
                                                      isArgReadOnly( llvm_param )
                                                          ? tint::core::Access::kRead
                                                          : tint::core::Access::kReadWrite );

            auto* storage_var = m_Builder.Var( name, storage_ptr );
            storage_var->SetBindingPoint( m_GroupCounter, m_BindingCounter++ );
            m_Module.root_block->Append( storage_var );

            m_ValueMap[llvm_param] = storage_var->Result();
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

void Translator::Translate( tint::core::ir::Var* globals )
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
        auto struct_name = m_SymbolTable.New( m_DemangledName + "_params_t" );
        auto* struct_type = m_Module.Types().Struct( struct_name, m_StructParamMembers );

        auto* uniform_ptr_type = m_Module.Types().ptr(
            tint::core::AddressSpace::kUniform, struct_type, tint::core::Access::kRead );

        auto var_name = m_DemangledName + "_params";
        auto* uniform_var = m_Builder.Var( var_name, uniform_ptr_type );

        uniform_var->SetBindingPoint( m_GroupCounter, m_BindingCounter++ );

        m_Module.root_block->Append( uniform_var );
    }

    translateFunctionBody();
}

void Translator::translateNormalFunction()
{
    if ( !m_FunctionParams.IsEmpty() )
        m_WGSLfunc->SetParams( m_FunctionParams );

    translateFunctionBody();
}


bool Translator::isArgReadOnly( const llvm::Argument* arg )
{
    if ( !arg->getType()->isPointerTy() ) {
        return false; // Not a pointer, irrelevant
    }

    const llvm::Function* parent = arg->getParent();
    unsigned arg_index = arg->getArgNo();

    // Check explicit ReadOnly attribute
    if ( parent->hasParamAttribute( arg_index, llvm::Attribute::ReadOnly ) ) {
        LOG_INFO << "Arg " << arg->getName() << " is ReadOnly (by attribute)" << LOG_END;
        return true;
    }

    // Check if NoAlias + ReadOnly is set (common for const pointers)
    if ( parent->hasParamAttribute( arg_index, llvm::Attribute::NoAlias ) &&
         !isPointerWritten( arg ) ) {
        LOG_INFO << "Arg " << arg->getName() << " is ReadOnly (no writes found)" << LOG_END;
        return true;
    }

    LOG_INFO << "Arg " << arg->getName() << " is NOT ReadOnly" << LOG_END;
    return false;
}

bool Translator::isPointerWritten( const llvm::Argument* arg )
{
    // Scan all uses of this argument
    for ( const llvm::User* user : arg->users() ) {
        if ( const auto* store = llvm::dyn_cast< llvm::StoreInst >( user ) ) {
            // Check if arg is the pointer operand (being written to)
            if ( store->getPointerOperand() == arg ) {
                return true;
            }
        }

        // Check GEP instructions that derive from this pointer
        if ( const auto* gep = llvm::dyn_cast< llvm::GetElementPtrInst >( user ) ) {
            // Recursively check if the GEP result is written to
            for ( const llvm::User* gep_user : gep->users() ) {
                if ( const auto* store = llvm::dyn_cast< llvm::StoreInst >( gep_user ) ) {
                    if ( store->getPointerOperand() == gep ) {
                        return true;
                    }
                }
            }
        }

        // Check for call instructions that might write through the pointer
        if ( const auto* call = llvm::dyn_cast< llvm::CallInst >( user ) ) {
            // If passed to another function without ReadOnly, assume it might be written
            const llvm::Function* callee = call->getCalledFunction();
            if ( callee ) {
                for ( unsigned i = 0; i < call->arg_size(); ++i ) {
                    if ( call->getArgOperand( i ) == arg ) {
                        if ( !callee->hasParamAttribute( i, llvm::Attribute::ReadOnly ) ) {
                            return true; // Might be written in callee
                        }
                    }
                }
            }
        }
    }

    return false;
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
    if ( !lhs )
        return;
    auto* rhs = getOperand( I, 1 );
    if ( !rhs )
        return;

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Add( type, lhs, rhs )->Result();
}

void Translator::visitFMul( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    if ( !lhs )
        return;
    auto* rhs = getOperand( I, 1 );
    if ( !rhs )
        return;

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

void Translator::visitCall( const llvm::Instruction& I )
{
    const auto* inst = llvm::dyn_cast< llvm::CallInst >( &I );
    const auto* callee = inst->getCalledFunction();

    std::string_view callee_name = callee->getName();

    // TODO
}

} // namespace WGSL
