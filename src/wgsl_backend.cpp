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
      m_SymbolTable( tint::GenerationID::New() ),
      m_Globals( std::make_unique< Globals >( m_Module, m_Builder, m_SymbolTable ) )
{
    // m_Module.functions.Clear();

    LOG_INFO << "<========= :: Starting Parsing :: =========>\n";
}

Backend::~Backend()
{
    initializeMainFunction();

    for ( const auto& translator : m_Translators ) {
        translator->Translate();
    }

    LOG_INFO << "Functions in module: " << m_Module.functions.Length() << LOG_END;
    for ( auto func : m_Module.functions ) {
        auto name = m_Module.NameOf( func );
        LOG_INFO << "  - " << name.Name() << LOG_END;
    }

    LOG_INFO << "Variables in root block: " << m_Module.root_block->Length() << LOG_END;
    for ( auto* inst : *m_Module.root_block ) {
        if ( auto* var = inst->As< tint::core::ir::Var >() ) {
            auto name = m_Module.NameOf( var->Result() );
            LOG_INFO << "  - " << name.Name() << LOG_END;
        }
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
    auto translator = std::make_unique< Translator >(
        m_Module, m_Builder, m_SymbolTable, *m_Globals, &F, isEntryPoint( F ) );

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

    Translator::RegisterWGSLFunction( &F, translator->GetWGSLFunc() );

    m_Translators.push_back( std::move( translator ) );
}

// Private Methods

void Backend::initializeMainFunction()
{
    m_MainFunction = m_Builder.ComputeFunction(
        "wgsl_main", tint::core::i32( 1 ), tint::core::i32( 1 ), tint::core::i32( 1 ) );

    m_Builder.Append( m_MainFunction->Block(), [&] {
        auto* local_id_param =
            m_Builder.FunctionParam( "local_id", m_Module.Types().vec3( m_Module.Types().u32() ) );
        local_id_param->SetBuiltin( tint::core::BuiltinValue::kLocalInvocationId );

        auto* workgroup_id_param =
            m_Builder.FunctionParam( "local_id", m_Module.Types().vec3( m_Module.Types().u32() ) );
        workgroup_id_param->SetBuiltin( tint::core::BuiltinValue::kWorkgroupId );

        auto* num_workgroups_param =
            m_Builder.FunctionParam( "local_id", m_Module.Types().vec3( m_Module.Types().u32() ) );
        num_workgroups_param->SetBuiltin( tint::core::BuiltinValue::kNumWorkgroups );

        auto* global_id_param =
            m_Builder.FunctionParam( "global_id", m_Module.Types().vec3( m_Module.Types().u32() ) );
        global_id_param->SetBuiltin( tint::core::BuiltinValue::kGlobalInvocationId );

        m_MainFunction->SetParams(
            { local_id_param, workgroup_id_param, num_workgroups_param, global_id_param } );

        auto* local_id = m_Builder.Access( m_Module.Types().vec3( m_Module.Types().u32() ),
                                           m_Globals->GetIntrinsicsStruct(),
                                           m_Builder.Constant( tint::core::u32( 0 ) ) );

        m_Builder.Store( local_id, local_id_param );

        auto* workgroup_id = m_Builder.Access( m_Module.Types().vec3( m_Module.Types().u32() ),
                                               m_Globals->GetIntrinsicsStruct(),
                                               m_Builder.Constant( tint::core::u32( 1 ) ) );

        m_Builder.Store( workgroup_id, workgroup_id_param );

        auto* num_workgroups = m_Builder.Access( m_Module.Types().vec3( m_Module.Types().u32() ),
                                                 m_Globals->GetIntrinsicsStruct(),
                                                 m_Builder.Constant( tint::core::u32( 2 ) ) );

        m_Builder.Store( num_workgroups, num_workgroups_param );

        auto* global_id = m_Builder.Access( m_Module.Types().vec3( m_Module.Types().u32() ),
                                            m_Globals->GetIntrinsicsStruct(),
                                            m_Builder.Constant( tint::core::u32( 3 ) ) );

        m_Builder.Store( global_id, global_id_param );

        auto* workgroup_size = m_Builder.Access( m_Module.Types().vec3( m_Module.Types().u32() ),
                                                 m_Globals->GetIntrinsicsStruct(),
                                                 m_Builder.Constant( tint::core::u32( 4 ) ) );

        auto* workgroup_size_const =
            m_Builder.Composite( m_Module.Types().vec3( m_Module.Types().u32() ),
                                 m_Builder.Constant( tint::core::u32( 256 ) ),
                                 m_Builder.Constant( tint::core::u32( 1 ) ),
                                 m_Builder.Constant( tint::core::u32( 1 ) ) );

        // How to put a constant vector here
        m_Builder.Store( workgroup_size, workgroup_size_const );

        for ( auto& translator : m_Translators ) {
            if ( translator->IsEntry() )
                m_Builder.Call( translator->GetWGSLFunc() );
        }

        m_Builder.Return( m_MainFunction );
    } );
}

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
