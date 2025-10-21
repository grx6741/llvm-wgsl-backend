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
    // --- blockIdx.* (ctaid) ---
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.x"] =
        createAccessor( M, B, "get_workgroup_id_x", 1, 0 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.y"] =
        createAccessor( M, B, "get_workgroup_id_y", 1, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ctaid.z"] =
        createAccessor( M, B, "get_workgroup_id_z", 1, 2 );

    // --- threadIdx.* (tid) ---
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.x"] =
        createAccessor( M, B, "get_local_id_x", 0, 0 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.y"] =
        createAccessor( M, B, "get_local_id_y", 0, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.tid.z"] =
        createAccessor( M, B, "get_local_id_z", 0, 2 );

    // --- gridDim.* (nctaid) ---
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.x"] =
        createAccessor( M, B, "get_num_workgroups_x", 2, 0 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.y"] =
        createAccessor( M, B, "get_num_workgroups_y", 2, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.nctaid.z"] =
        createAccessor( M, B, "get_num_workgroups_z", 2, 2 );

    // --- global_id.* (synthetic CUDA global thread id) ---
    m_IntrinsicAccessors["cuda.global.id.x"] = createAccessor( M, B, "get_global_id_x", 3, 0 );
    m_IntrinsicAccessors["cuda.global.id.y"] = createAccessor( M, B, "get_global_id_y", 3, 1 );
    m_IntrinsicAccessors["cuda.global.id.z"] = createAccessor( M, B, "get_global_id_z", 3, 2 );

    // --- gridDim.* (nctaid) ---
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.x"] =
        createAccessor( M, B, "get_workgroup_size_x", 4, 0 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.y"] =
        createAccessor( M, B, "get_workgroups_size_y", 4, 1 );
    m_IntrinsicAccessors["llvm.nvvm.read.ptx.sreg.ntid.z"] =
        createAccessor( M, B, "get_workgroups_size_z", 4, 2 );
}

tint::core::ir::Function* Globals::createAccessor( tint::core::ir::Module& M,
                                                   tint::core::ir::Builder& B,
                                                   const std::string& name,
                                                   int intrinsic_index,
                                                   int component_index )
{
    // Create function: fn llvm_nvvm_read_ptx_sreg_ctaid_x() -> u32
    auto* func = B.Function( name, M.Types().u32() );

    // Build function body
    B.Append( func->Block(), [&] {
        // Access globals.global_id_
        auto* globals_ptr_type = M.Types().ptr( tint::core::AddressSpace::kPrivate,
                                                M.Types().vec3( M.Types().u32() ),
                                                tint::core::Access::kReadWrite );

        // Access the global_id_ member (index 0 in the struct)
        auto* global_id_access =
            B.Access( globals_ptr_type,
                      m_Intrinsics->Result(),
                      B.Constant( tint::core::u32( intrinsic_index ) ) // Index of global_id_ member
            );

        // Load the vec3<u32>
        auto* global_id_vec = B.Load( global_id_access );

        // Extract the component (x=0, y=1, z=2)
        auto* component = B.Access( M.Types().u32(),
                                    global_id_vec->Result(),
                                    B.Constant( tint::core::u32( component_index ) ) );

        // Return the component
        B.Return( func, component->Result() );
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
