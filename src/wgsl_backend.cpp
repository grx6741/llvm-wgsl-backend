#include "wgsl_backend.hpp"

#include "src/tint/lang/wgsl/writer/ir_to_program/program_options.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include "llvm/IR/Module.h"

namespace WGSL
{

Backend::Backend() : m_Builder( m_Module ), m_TypeManager( m_Module.Types() )
{
    m_Module.functions.Clear();
}

Backend::~Backend()
{
    tint::wgsl::writer::ProgramOptions program_options;
    program_options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
    auto result = tint::wgsl::writer::WgslFromIR( m_Module, {} );
    if ( result != tint::Success ) {
        llvm::errs() << "Error generating WGSL: " << result.Failure().reason << "\n";

        return;
    }

    llvm::outs() << result->wgsl << "\n";
}

void Backend::RegisterFunction( const llvm::Function& F )
{
    if ( isKernelFunction( F ) )
        registerAsKernelFunction( F );
    else
        registerAsNormalFunction( F );
}

const tint::core::type::Type* Backend::getWGSLType( const llvm::Type* type )
{
    auto llvm_type_id = type->getTypeID();
    switch ( llvm_type_id ) {
        case llvm::Type::IntegerTyID:
            return m_TypeManager.i32();
        case llvm::Type::FloatTyID:
            return m_TypeManager.f32();
        case llvm::Type::VoidTyID:
            return m_TypeManager.void_();
        // case llvm::Type::PointerTyID: return m_TypeManager.ptr();
        default:
            return m_TypeManager.invalid();
    }

    return m_TypeManager.invalid();
}

const std::string Backend::demangledName( const std::string& mangled_name )
{
    if ( m_Demangler.partialDemangle( mangled_name.c_str() ) ) {
        return mangled_name;
    }

    return m_Demangler.getFunctionBaseName( nullptr, nullptr );
}

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

void Backend::registerAsNormalFunction( const llvm::Function& F )
{}

void Backend::registerAsKernelFunction( const llvm::Function& F )
{
    const auto demangled_name = demangledName( F.getName().str() );

    auto* func = m_Builder.ComputeFunction(
        demangled_name, tint::core::i32( 1 ), tint::core::i32( 1 ), tint::core::i32( 1 ) );

    if ( !func ) {
        llvm::errs() << "Failed to create WGSL function for " << F.getName().str() << "\n";
        return;
    }
}

} // namespace WGSL
