//=============================================================================
//=============================================================================
// DebugConsole.cpp
//
// Utility to popup & utilize console window in GUI application.
//
// taken from:
// http://dslweb.nwnexus.com/~ast/dload/guicon.htm
//=============================================================================
//=============================================================================

// local
#include "DebugConsole.h"
// ansi
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>
// microsoft
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#include <windows.h>

// for io
using namespace std;


///////////////////////////////////////////////////////////////////////////////
// SetupDebugConsole
//
// Function to allocate and display conosle window & to re-route all standard
// input, ouput & error streams to it.
///////////////////////////////////////////////////////////////////////////////
void SetupDebugConsole( short bufferWidth, short bufferHeight, short windowWidth, short windowHeight )
{
    // locals
    CONSOLE_SCREEN_BUFFER_INFO  coninfo;
    FILE                       *pFile;
    int                         conHandle;
    HANDLE                      stdHandle;
    SMALL_RECT                  window = {0,};

    // allocate console
    AllocConsole();

    // reset console properties
    GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ), &coninfo );
    coninfo.dwSize.Y = bufferHeight;
    coninfo.dwSize.X = bufferWidth;
    window.Left      = 0;
    window.Top       = 0;
    window.Right     = windowWidth-1;
    window.Bottom    = windowHeight-1;
    SetConsoleScreenBufferSize( GetStdHandle( STD_OUTPUT_HANDLE ), coninfo.dwSize );
    SetConsoleWindowInfo( GetStdHandle( STD_OUTPUT_HANDLE ), true, &window );

    // redirect STDOUT to console
    stdHandle = GetStdHandle( STD_OUTPUT_HANDLE );
    conHandle = _open_osfhandle( (intptr_t)stdHandle, _O_TEXT );
    pFile = _fdopen( conHandle, "w" );
    *stdout = *pFile;
    setvbuf( stdout, NULL, _IONBF, 0 ); // unbuffered

    // redirect STDIN to console
    stdHandle = GetStdHandle( STD_INPUT_HANDLE );
    conHandle = _open_osfhandle( (intptr_t)stdHandle, _O_TEXT );
    pFile = _fdopen( conHandle, "r" );
    *stdin = *pFile;
    setvbuf( stdin, NULL, _IONBF, 0 ); // unbuffered

    // redirect STDERR to console
    stdHandle = GetStdHandle( STD_ERROR_HANDLE );
    conHandle = _open_osfhandle( (intptr_t)stdHandle, _O_TEXT );
    pFile = _fdopen( conHandle, "w" );
    *stderr = *pFile;
    setvbuf( stderr, NULL, _IONBF, 0 ); // unbuffered

    // route cout, wcout, cin, wcin, wcerr, cerr, wclog & clog as well
    ios::sync_with_stdio();
}