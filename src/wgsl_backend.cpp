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
    m_Translators.emplace_back( m_Module, m_Builder, m_SymbolTable, &F, isKernelFunction( F ) );
    auto& translator = m_Translators.back();

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
            translator.AddFunctionParam( arg.getName().str(), &arg, arg_type );
    }

    translator.TranslateBody();
}

// Private Methods

bool Backend::isKernelFunction( const llvm::Function& F )
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
