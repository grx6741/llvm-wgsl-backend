#include "translator.hpp"
#include "util.hpp"

#include <cstdlib>

#include <cassert>

namespace WGSL
{

void Translator::RegisterWGSLFunction( const llvm::Function* llvm_func,
                                       tint::core::ir::Function* wgsl_func )
{
    s_FunctionMap[llvm_func] = wgsl_func;
}

tint::core::ir::Function* Translator::GetWGSLFunction( const llvm::Function* llvm_func )
{
    auto it = s_FunctionMap.find( llvm_func );
    return ( it != s_FunctionMap.end() ) ? it->second : nullptr;
}

std::unordered_map< const llvm::Function*, tint::core::ir::Function* > Translator::s_FunctionMap;

Translator::Translator( tint::core::ir::Module& M,
                        tint::core::ir::Builder& B,
                        tint::SymbolTable& ST,
                        Globals& G,
                        const llvm::Function* F,
                        bool isEntry )
    : m_LLVMfunc{ F },
      m_Module{ M },
      m_Builder{ B },
      m_SymbolTable{ ST },
      m_Globals{ G },
      m_IsEntry{ isEntry },
      m_GroupCounter{ 0 },
      m_BindingCounter{ 0 }
{
    if ( !m_LLVMfunc ) {
        LOG_ERROR << "Passed LLVM Function is NULL\n";
        return;
    }
    m_DemangledName = getDemangledName( m_LLVMfunc->getName().str() );
    m_WGSLfunc = m_Builder.Function(
        m_DemangledName, MapLLVMtype2WGSLtype( m_Module, m_LLVMfunc->getReturnType() ) );

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
            // m_StructParamMembers.Push(
            //     tint::core::type::Manager::StructMemberDesc{ m_SymbolTable.New( name ), type } );

            m_StructParamMembers[llvm_param] =
                tint::core::type::Manager::StructMemberDesc{ m_SymbolTable.New( name ), type };
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
    // Create uniform struct if we have scalar parameters
    tint::core::ir::Var* uniform_var = nullptr;

    if ( !m_StructParamMembers.empty() ) {
        tint::Vector< tint::core::type::Manager::StructMemberDesc, 8 > struct_params;
        std::unordered_map< const llvm::Value*, uint32_t > param_to_index;
        uint32_t index = 0;

        for ( const auto& [llvm_value, wgsl_param] : m_StructParamMembers ) {
            struct_params.Push( wgsl_param );
            param_to_index[llvm_value] = index++;
        }

        auto struct_name = m_SymbolTable.New( m_DemangledName + "_params_t" );
        auto* struct_type = m_Module.Types().Struct( struct_name, struct_params );

        auto* uniform_ptr_type = m_Module.Types().ptr(
            tint::core::AddressSpace::kUniform, struct_type, tint::core::Access::kRead );

        auto var_name = m_DemangledName + "_params";
        uniform_var = m_Builder.Var( var_name, uniform_ptr_type );

        uniform_var->SetBindingPoint( m_GroupCounter, m_BindingCounter++ );
        m_Module.root_block->Append( uniform_var );

        // Store the mapping for later use inside the function
        m_UniformVar = uniform_var;
        m_ParamToIndex = param_to_index;
    }

    // Now translate the function body
    // The parameter loading will happen at the start of translateFunctionBody
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
    if ( !m_LLVMfunc || m_LLVMfunc->empty() ) {
        return;
    }

    const llvm::BasicBlock* entry = &m_LLVMfunc->getEntryBlock();

    m_Builder.Append( m_WGSLfunc->Block(), [&] {
        // FIRST: Load kernel parameters from uniform struct
        if ( m_IsEntry && m_UniformVar ) {
            for ( const auto& [llvm_value, member_idx] : m_ParamToIndex ) {
                auto* member_type = m_StructParamMembers[llvm_value].type;

                auto* member_ptr =
                    m_Builder.Access( m_Module.Types().ptr( tint::core::AddressSpace::kUniform,
                                                            member_type,
                                                            tint::core::Access::kRead ),
                                      m_UniformVar->Result(),
                                      m_Builder.Constant( tint::core::u32( member_idx ) ) );

                // Load the value
                auto* loaded = m_Builder.Load( member_ptr );

                // Store loaded VALUE in value map
                m_ValueMap[llvm_value] = loaded->Result();

                LOG_INFO << "    Loaded parameter from uniform struct" << LOG_END;
            }
        }

        // THEN: Translate the basic blocks
        translateBasicBlock( entry, [&]() {} );
    } );
}

void Translator::translateBasicBlock( const llvm::BasicBlock* BB,
                                      std::function< void() > append_context )
{
    if ( m_TranslatedBlocks.count( BB ) ) {
        return;
    }
    m_TranslatedBlocks.insert( BB );

    LOG_INFO << "Translating BasicBlock: " << BB->getName() << LOG_END;

    for ( auto it = BB->begin(); it != BB->end(); ++it ) {
        const llvm::Instruction& I = *it;

        if ( I.isTerminator() ) {
            break;
        }

#define VISIT_INST( inst )                                                                         \
    case llvm::Instruction::inst:                                                                  \
        visit##inst( I );                                                                          \
        break;
        switch ( I.getOpcode() ) {
            VISIT_INST( FAdd );
            VISIT_INST( FMul );
            VISIT_INST( Add );
            VISIT_INST( Mul );
            VISIT_INST( Call );
            VISIT_INST( ICmp );
            VISIT_INST( Load );
            VISIT_INST( GetElementPtr );
            VISIT_INST( SExt );
            VISIT_INST( Store );
#undef VISIT_INST
            default:
                LOG_WARN << I.getOpcodeName() << " instruction NOT IMPLEMENTED" << LOG_END;
        }
    }

    const llvm::Instruction* terminator = BB->getTerminator();
    if ( !terminator ) {
        LOG_ERROR << "BasicBlock has no terminator!" << LOG_END;
        return;
    }

    if ( const auto* ret = llvm::dyn_cast< llvm::ReturnInst >( terminator ) ) {
        visitRet( *ret );
    }
    else if ( const auto* br = llvm::dyn_cast< llvm::BranchInst >( terminator ) ) {
        handleBranch( br );
    }
    else {
        LOG_WARN << "Unhandled terminator: " << terminator->getOpcodeName() << LOG_END;
    }
}

bool Translator::isJustBranchTo( const llvm::BasicBlock* BB, const llvm::BasicBlock* target )
{
    if ( !target )
        return false;

    // Count non-branch instructions
    size_t inst_count = 0;
    for ( const auto& I : *BB ) {
        if ( !I.isTerminator() ) {
            inst_count++;
        }
    }

    // If only terminator exists, check if it branches to target
    if ( inst_count == 0 ) {
        if ( const auto* br = llvm::dyn_cast< llvm::BranchInst >( BB->getTerminator() ) ) {
            if ( br->isUnconditional() && br->getSuccessor( 0 ) == target ) {
                return true;
            }
        }
    }

    return false;
}

void Translator::handleBranch( const llvm::BranchInst* br )
{
    if ( br->isUnconditional() ) {
        const llvm::BasicBlock* successor = br->getSuccessor( 0 );
        translateBasicBlock( successor, [&]() {} );
    }
    else {
        const llvm::BasicBlock* true_block = br->getSuccessor( 0 );
        const llvm::BasicBlock* false_block = br->getSuccessor( 1 );

        llvm::Value* condition = br->getCondition();
        auto* cond_value = m_ValueMap.at( condition );

        LOG_INFO << "    Creating If statement" << LOG_END;
        LOG_INFO << "      True block: " << true_block->getName() << LOG_END;
        LOG_INFO << "      False block: " << false_block->getName() << LOG_END;

        const llvm::BasicBlock* merge_block = findMergeBlock( true_block, false_block );

        if ( merge_block ) {
            LOG_INFO << "      Merge block: " << merge_block->getName() << LOG_END;
        }

        // Create If instruction
        auto* if_inst = m_Builder.If( cond_value );

        // Translate true block
        m_Builder.Append( if_inst->True(), [&] {
            if ( !isJustBranchTo( true_block, merge_block ) ) {
                translateBasicBlock( true_block, [&]() {} );
            }
        } );

        // Translate false block (if it's not the merge block)
        if ( false_block != merge_block ) {
            m_Builder.Append( if_inst->False(), [&] {
                if ( !isJustBranchTo( false_block, merge_block ) ) {
                    translateBasicBlock( false_block, [&]() {} );
                }
            } );
        }

        // Continue with merge block
        if ( merge_block ) {
            translateBasicBlock( merge_block, [&]() {} );
        }
    }
}

const llvm::BasicBlock* Translator::findMergeBlock( const llvm::BasicBlock* true_block,
                                                    const llvm::BasicBlock* false_block )
{
    if ( !true_block || !false_block )
        return nullptr;

    // Get true block's successors
    std::unordered_set< const llvm::BasicBlock* > true_successors;
    const auto* true_term = true_block->getTerminator();
    if ( const auto* br = llvm::dyn_cast< llvm::BranchInst >( true_term ) ) {
        for ( unsigned i = 0; i < br->getNumSuccessors(); ++i ) {
            true_successors.insert( br->getSuccessor( i ) );
        }
    }

    // Check if false_block itself is the merge
    if ( true_successors.count( false_block ) ) {
        return false_block;
    }

    // Check false block's successors
    const auto* false_term = false_block->getTerminator();
    if ( const auto* br = llvm::dyn_cast< llvm::BranchInst >( false_term ) ) {
        for ( unsigned i = 0; i < br->getNumSuccessors(); ++i ) {
            if ( true_successors.count( br->getSuccessor( i ) ) ) {
                return br->getSuccessor( i );
            }
        }
    }

    return nullptr;
}

// Intruction Visitors
void Translator::visitFAdd( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    if ( !lhs || !rhs ) {
        LOG_ERROR << "Failed to get operands for FAdd" << LOG_END;
        return;
    }

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Add( type, lhs, rhs )->Result();
}

void Translator::visitAdd( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    if ( !lhs || !rhs ) {
        LOG_ERROR << "Failed to get operands for Add" << LOG_END;
        return;
    }

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Add( type, lhs, rhs )->Result();
}

void Translator::visitFMul( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    if ( !lhs || !rhs ) {
        LOG_ERROR << "Failed to get operands for FMul" << LOG_END;
        return;
    }

    const auto* type = MapLLVMtype2WGSLtype( m_Module, I.getType() );

    m_ValueMap[&I] = m_Builder.Multiply( type, lhs, rhs )->Result();
}

void Translator::visitMul( const llvm::Instruction& I )
{
    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    if ( !lhs || !rhs ) {
        LOG_ERROR << "Failed to get operands for Mul" << LOG_END;
        return;
    }

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
    LOG_INFO << "    visitCall for: " << I << LOG_END;

    // Use isa instead of dyn_cast for checking
    if ( !llvm::isa< llvm::CallInst >( &I ) ) {
        LOG_ERROR << "Instruction is not a CallInst!" << LOG_END;
        LOG_ERROR << "    Opcode: " << I.getOpcodeName() << LOG_END;
        LOG_ERROR << "    Type ID: " << I.getOpcode() << LOG_END;
        return;
    }
    const auto* inst = llvm::dyn_cast< llvm::CallInst >( &I );
    if ( !inst ) {
        LOG_ERROR << "Failed to cast to CallInst" << LOG_END;
        return;
    }

    const auto* callee = inst->getCalledFunction();
    if ( !callee ) {
        LOG_WARN << "Indirect call not supported" << LOG_END;
        return;
    }

    std::string callee_name = callee->getName().str();
    LOG_INFO << "    Call to: " << callee_name << LOG_END;

    // 1. Check if it's an intrinsic
    if ( auto* accessor = m_Globals.GetIntrinsicAccessor( callee_name ) ) {
        auto* call = m_Builder.Call( accessor );
        m_ValueMap[&I] = call->Result();
        return;
    }

    // 2. Check if it's a translated function
    if ( auto* wgsl_func = GetWGSLFunction( callee ) ) {
        // Build argument list
        tint::Vector< tint::core::ir::Value*, 8 > args;
        for ( unsigned i = 0; i < inst->arg_size(); ++i ) {
            auto* arg = getOperand( I, i );
            if ( !arg ) {
                LOG_ERROR << "Failed to get argument " << i << LOG_END;
                return;
            }
            args.Push( arg );
        }

        auto* call = m_Builder.Call( wgsl_func, args );
        m_ValueMap[&I] = call->Result();

        LOG_INFO << "      Called WGSL function with " << args.Length() << " args" << LOG_END;
        return;
    }

    LOG_ERROR << "Unknown function call: " << callee_name << LOG_END;
}

void Translator::visitICmp( const llvm::Instruction& I )
{
    const auto* cmp = llvm::cast< llvm::ICmpInst >( &I );

    auto* lhs = getOperand( I, 0 );
    auto* rhs = getOperand( I, 1 );

    if ( !lhs || !rhs ) {
        LOG_ERROR << "Failed to get operands for ICmp" << LOG_END;
        return;
    }

    auto predicate = cmp->getPredicate();
    tint::core::ir::Binary* result = nullptr;

    // Map LLVM predicates to Tint IR operations
    switch ( predicate ) {
        case llvm::ICmpInst::ICMP_EQ:
            result = m_Builder.Equal( m_Module.Types().bool_(), lhs, rhs );
            break;
        case llvm::ICmpInst::ICMP_NE:
            result = m_Builder.NotEqual( m_Module.Types().bool_(), lhs, rhs );
            break;
        case llvm::ICmpInst::ICMP_SLT: // Signed less than
        case llvm::ICmpInst::ICMP_ULT: // Unsigned less than
            result = m_Builder.LessThan( m_Module.Types().bool_(), lhs, rhs );
            break;
        case llvm::ICmpInst::ICMP_SLE: // Signed less or equal
        case llvm::ICmpInst::ICMP_ULE: // Unsigned less or equal
            result = m_Builder.LessThanEqual( m_Module.Types().bool_(), lhs, rhs );
            break;
        case llvm::ICmpInst::ICMP_SGT: // Signed greater than
        case llvm::ICmpInst::ICMP_UGT: // Unsigned greater than
            result = m_Builder.GreaterThan( m_Module.Types().bool_(), lhs, rhs );
            break;
        case llvm::ICmpInst::ICMP_SGE: // Signed greater or equal
        case llvm::ICmpInst::ICMP_UGE: // Unsigned greater or equal
            result = m_Builder.GreaterThanEqual( m_Module.Types().bool_(), lhs, rhs );
            break;
        default:
            LOG_ERROR << "Unsupported ICmp predicate: " << cmp->getPredicateName( predicate )
                      << LOG_END;
            return;
    }

    if ( result ) {
        m_ValueMap[&I] = result->Result();
        LOG_INFO << "    ICmp: " << cmp->getPredicateName( predicate ) << LOG_END;
    }
}

void Translator::visitBr( const llvm::Instruction& I )
{}

void Translator::visitLoad( const llvm::Instruction& I )
{
    auto* ptr = getOperand( I, 0 );
    if ( !ptr )
        return;

    auto* load = m_Builder.Load( ptr );
    m_ValueMap[&I] = load->Result();

    LOG_INFO << "    Load from " << I.getOperand( 0 )->getName() << LOG_END;
}

void Translator::visitGetElementPtr( const llvm::Instruction& I )
{
    const auto* gep = llvm::cast< llvm::GetElementPtrInst >( &I );

    auto* base = getOperand( I, gep->getPointerOperandIndex() );
    if ( !base ) {
        LOG_ERROR << "Failed to get base pointer for GEP" << LOG_END;
        return;
    }

    LOG_INFO << "    GEP: " << gep->getNumIndices() << " indices" << LOG_END;

    if ( gep->getNumIndices() == 1 ) {
        // Get the index
        auto* index = getOperand( I, gep->getNumOperands() - 1 );
        if ( !index ) {
            LOG_ERROR << "Failed to get index for GEP" << LOG_END;
            return;
        }

        // Determine address space from base pointer
        auto* base_ptr_type = base->Type()->As< tint::core::type::Pointer >();
        auto address_space =
            base_ptr_type ? base_ptr_type->AddressSpace() : tint::core::AddressSpace::kStorage;

        // Determine access mode from base pointer
        auto access = base_ptr_type ? base_ptr_type->Access() : tint::core::Access::kReadWrite;

        auto* result_elem_type = MapLLVMtype2WGSLtype( m_Module, gep->getResultElementType() );

        auto* access_inst = m_Builder.Access(
            m_Module.Types().ptr( address_space, result_elem_type, access ), base, index );

        m_ValueMap[&I] = access_inst->Result();
        LOG_INFO << "      Created array access" << LOG_END;
    }
    else {
        LOG_ERROR << "Multi-index GEP not supported" << LOG_END;
    }
}

void Translator::visitSExt( const llvm::Instruction& I )
{
    // Sign extension: i32 -> i64 (but WGSL doesn't have i64, so just pass through)
    auto* operand = getOperand( I, 0 );
    if ( !operand )
        return;

    // For now, just map directly (WGSL will use i32 or u32)
    m_ValueMap[&I] = operand;

    LOG_INFO << "    SExt (pass-through)" << LOG_END;
}

void Translator::visitStore( const llvm::Instruction& I )
{
    auto* value = getOperand( I, 0 ); // Value to store
    auto* ptr = getOperand( I, 1 );   // Destination pointer

    if ( !value || !ptr ) {
        LOG_ERROR << "Failed to get Store operands" << LOG_END;
        return;
    }

    m_Builder.Store( ptr, value );
    LOG_INFO << "    Store" << LOG_END;
}

} // namespace WGSL
