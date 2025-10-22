#include "globals.hpp"

Globals::Globals( tint::core::ir::Module& M, tint::core::ir::Builder& B, tint::SymbolTable& S )
{
    tint::Vector< tint::core::type::Manager::StructMemberDesc, 4 > members;

    members.Push( { S.New( "local_id" ), M.Types().vec3( M.Types().u32() ) } );
    members.Push( { S.New( "workgroup_id" ), M.Types().vec3( M.Types().u32() ) } );
    members.Push( { S.New( "num_workgroups" ), M.Types().vec3( M.Types().u32() ) } );
    members.Push( { S.New( "global_id" ), M.Types().vec3( M.Types().u32() ) } );
    members.Push( { S.New( "workgroup_size" ), M.Types().vec3( M.Types().u32() ) } );

    auto* globals_struct = M.Types().Struct( S.New( "globals_t" ), members );

    m_Intrinsics = B.Var( "globals",
                          M.Types().ptr( tint::core::AddressSpace::kPrivate,
                                         globals_struct,
                                         tint::core::Access::kReadWrite ) );

    M.root_block->Append( m_Intrinsics );
    createIntrinsicAccessors( M, B, S );
}

void Globals::createIntrinsicAccessors( tint::core::ir::Module& M,
                                        tint::core::ir::Builder& B,
                                        tint::SymbolTable& S )
{
    // Member indices in globals_t:
    // 0: local_id
    // 1: workgroup_id
    // 2: num_workgroups
    // 3: global_id
    // 4: workgroup_size

    // workgroup_id (blockIdx)
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.x"] =
        createAccessor( M, B, "get_workgroup_id_x", 1, 0 ); // member 1, component x
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.y"] =
        createAccessor( M, B, "get_workgroup_id_y", 1, 1 ); // member 1, component y
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.z"] =
        createAccessor( M, B, "get_workgroup_id_z", 1, 2 ); // member 1, component z

    // local_id (threadIdx)
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.x"] =
        createAccessor( M, B, "get_local_id_x", 0, 0 ); // member 0, component x
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.y"] =
        createAccessor( M, B, "get_local_id_y", 0, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.z"] =
        createAccessor( M, B, "get_local_id_z", 0, 2 );

    // num_workgroups (gridDim)
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.x"] =
        createAccessor( M, B, "get_num_workgroups_x", 2, 0 ); // member 2
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.y"] =
        createAccessor( M, B, "get_num_workgroups_y", 2, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.z"] =
        createAccessor( M, B, "get_num_workgroups_z", 2, 2 );

    // workgroup_size (blockDim)
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.x"] =
        createAccessor( M, B, "get_workgroup_size_x", 4, 0 ); // member 4
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.y"] =
        createAccessor( M, B, "get_workgroups_size_y", 4, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.z"] =
        createAccessor( M, B, "get_workgroups_size_z", 4, 2 );
}

tint::core::ir::Function*
Globals::createAccessor( tint::core::ir::Module& M,
                         tint::core::ir::Builder& B,
                         const std::string& name,
                         int member_index,     // ✅ Which struct member (0-4)
                         int component_index ) // ✅ Which component (0=x, 1=y, 2=z)
{
    auto* func = B.Function( name, M.Types().i32() );

    B.Append( func->Block(), [&] {
        // Access the correct struct member AND component in one chain
        auto* component_ptr =
            B.Access( M.Types().ptr( tint::core::AddressSpace::kPrivate,
                                     M.Types().u32(),
                                     tint::core::Access::kReadWrite ),
                      m_Intrinsics->Result(),
                      B.Constant( tint::core::u32( member_index ) ),   // Struct member
                      B.Constant( tint::core::u32( component_index ) ) // Vector component
            );

        auto* loaded = B.Load( component_ptr );
        auto* casted = B.Convert( M.Types().i32(), loaded->Result() );

        B.Return( func, casted->Result() );
    } );

    return func;
}

tint::core::ir::Function* Globals::GetIntrinsicAccessor( std::string_view llvm_name ) const
{
    auto it = m_IntrinsicAccessors.find( llvm_name );
    if ( it != m_IntrinsicAccessors.end() ) {
        return it->second;
    }
    return nullptr;
}
