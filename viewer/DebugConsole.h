//=============================================================================
//=============================================================================
// DebugConsole.h
//
// Utility to popup & utilize console window in GUI application.
//
// taken from:
// http://dslweb.nwnexus.com/~ast/dload/guicon.htm
//=============================================================================
//=============================================================================
#ifndef _DEBUG_CONSOLE_H_
#define _DEBUG_CONSOLE_H_

// defines
#define MAX_CONSOLE_BUFFER_LINES    300     // console line count
#define MAX_CONSOLE_BUFFER_COLUMNS  80      // console line count
#define MAX_CONSOLE_WINDOW_LINES    24      // console line count
#define MAX_CONSOLE_WINDOW_COLUMNS  80      // console line count

///////////////////////////////////////////////////////////////////////////////
// SetupDebugConsole
//
// Function to allocate and display conosle window & to re-route all standard
// input, ouput & error streams to it.
//
// Note buffer and window Width/Height is in characters NOT pixels.
///////////////////////////////////////////////////////////////////////////////
void SetupDebugConsole(
    short bufferWidth  = MAX_CONSOLE_BUFFER_COLUMNS,
    short bufferHeight = MAX_CONSOLE_BUFFER_LINES,
    short windowWidth  = MAX_CONSOLE_WINDOW_COLUMNS,
    short windowHeight = MAX_CONSOLE_WINDOW_LINES );

#endif
