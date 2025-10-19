#include "src/tint/lang/core/number.h"
#include "util.hpp"
#include "wgsl_backend.hpp"

#include "src/tint/lang/wgsl/writer/ir_to_program/program_options.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include <llvm-19/llvm/IR/Instruction.h>

#include "cassert"

namespace WGSL
{

Backend::Backend()
    : m_Builder( m_Module ),
      m_TypeManager( m_Module.Types() ),
      m_SymbolTable( tint::GenerationID::New() )
{
    m_Module.functions.Clear();

    LOG_INFO << "<========= :: Starting Parsing :: =========>\n";
}

Backend::~Backend()
{
    tint::Vector< tint::core::type::Manager::StructMemberDesc, 4 > members;
    members.Push(
        { m_SymbolTable.New( "global_id_" ), m_Module.Types().vec3( m_Module.Types().u32() ) } );

    auto* globals_struct = m_Module.Types().Struct( m_SymbolTable.New( "globals_t" ), members );

    auto* globals_var = m_Builder.Var( "globals",
                                       m_Module.Types().ptr( tint::core::AddressSpace::kPrivate,
                                                             globals_struct,
                                                             tint::core::Access::kRead ) );

    m_Module.root_block->Append( globals_var );

    auto* main_func = m_Builder.ComputeFunction(
        "wgsl_main", tint::core::i32( 1 ), tint::core::i32( 1 ), tint::core::i32( 1 ) );

    auto* global_id_param =
        m_Builder.FunctionParam( "global_id", m_Module.Types().vec3( m_Module.Types().u32() ) );
    global_id_param->SetBuiltin( tint::core::BuiltinValue::kGlobalInvocationId );
    main_func->SetParams( { global_id_param } );

    m_Builder.Append( main_func->Block(), [&] {
        auto* global_id = m_Builder.Access( m_TypeManager.vec3( m_TypeManager.u32() ),
                                            globals_var,
                                            m_Builder.Constant( tint::core::u32( 0 ) ) );

        m_Builder.Store( global_id, global_id_param );

        for ( auto& translator : m_Translators ) {
            if ( translator->IsEntry() )
                m_Builder.Call( translator->GetWGSLFunc() );
        }

        m_Builder.Return( main_func );
    } );

    for ( const auto& translator : m_Translators ) {
        translator->Translate( globals_var );
    }

    LOG_INFO << "<========= :: Finished Parsing :: =========>\n";

    tint::wgsl::writer::ProgramOptions program_options;
    program_options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
    auto result = tint::wgsl::writer::WgslFromIR( m_Module, {} );

    if ( result != tint::Success ) {
        LOG_ERROR << "Error generating WGSL: " << result.Failure().reason << "\n";

        return;
    }

    llvm::outs() << "\n" << result->wgsl << "\n";
}

void Backend::RegisterFunction( const llvm::Function& F )
{
    auto translator =
        std::make_unique< Translator >( m_Module, m_Builder, m_SymbolTable, &F, isEntryPoint( F ) );

    const tint::core::type::Type* arg_type;

    for ( const auto& arg : F.args() ) {
        if ( arg.getType()->getTypeID() == llvm::Type::PointerTyID ) {

            llvm::SmallSet< llvm::Value*, 32 > visited;
            // Get what data type its pointing to
            if ( const auto* llvm_pointer_type = Translator::IsArgUsedAsArray( &arg, visited ) ) {
                const auto* wgsl_array =
                    Translator::MapLLVMtype2WGSLtype( m_Module, llvm_pointer_type );
                arg_type = m_Module.Types().runtime_array( wgsl_array );
            }
            else {
                // TODO : Not Implemented
            }
        }
        else {
            arg_type = Translator::MapLLVMtype2WGSLtype( m_Module, arg.getType() );
        }

        if ( arg_type )
            translator->AddFunctionParam( arg.getName().str(), &arg, arg_type );
    }

    m_Translators.push_back( std::move( translator ) );
}

// Private Methods

bool Backend::isEntryPoint( const llvm::Function& F )
{
    const llvm::Module* M = F.getParent();
    if ( !M )
        return false;

    const llvm::NamedMDNode* Annonations = M->getNamedMetadata( "nvvm.annotations" );
    if ( !Annonations )
        return false;

    for ( const llvm::MDNode* Op : Annonations->operands() ) {
        if ( Op->getNumOperands() < 3 )
            continue;

        const auto* FuncMD = llvm::dyn_cast< llvm::ValueAsMetadata >( Op->getOperand( 0 ) );
        if ( !FuncMD )
            continue;

        if ( FuncMD->getValue() != &F )
            continue;

        const auto* KeyMD = llvm::dyn_cast< llvm::MDString >( Op->getOperand( 1 ) );
        if ( !KeyMD )
            continue;

        if ( KeyMD->getString() == "kernel" )
            return true;
    }

    return false;
}

} // namespace WGSL
