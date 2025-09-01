#pragma once

#define NOT_IMPLEMENTED( x ) assert( 0 && x " is Not Implemented" )

#define LOG_INFO llvm::outs() << "[LOG]\t"
#define LOG_WARN llvm::outs() << "[WARN]\t"
#define LOG_ERROR llvm::outs() << "[ERROR]\t"
#define LOG_END "\n"
