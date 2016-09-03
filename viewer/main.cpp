//----------------------------------------------------------------------------
// main.cpp
//
//
//----------------------------------------------------------------------------

// local
#include "Timer.h"
#include "DebugConsole.h"
#include "BMPLoader.h"
extern "C"
#include "los.h"
// microsoft
#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN
#endif
#include <windows.h>
#include <iostream>
// dx
#ifdef _DEBUG
 #define D3D_DEBUG_INFO
#endif
#include <d3d9.h>
#include <d3dx9.h>
#define  DIRECTINPUT_VERSION  0x0800
#include <dinput.h>


//=====================================
// CONSTANTS, MACROS, DEFINTIONS
//=====================================

#define SCREEN_WIDTH    1024
#define SCREEN_HEIGHT   768
#define D3DFVF_CUSTOM   ( D3DFVF_XYZ | D3DFVF_DIFFUSE )
#define CAMERA_PAN_SPEED    ( 8.0f )
#define CAMERA_ROT_SPEED    ( 0.004f )   

//#define HEIGHT_MAP      "0016.bmp"
//#define HEIGHT_MAP      "0128.bmp"
#define HEIGHT_MAP      "0512.bmp"
//#define HEIGHT_MAP      "1024.bmp"
//#define HEIGHT_MAP      "2048.bmp"
//#define HEIGHT_MAP      "4096.bmp"
#define DO_GPU          1       // calc gpu version
#define DO_CPU          0       // calc cpu version
#define PRINT_MATRIX    0

#define KEYDOWN( buffer, key )  ( buffer[ key ] & 0x80 )

typedef struct _CUSTOMVERT
{
    D3DXVECTOR3 pos;        // position ( local )
    D3DCOLOR    color;      // vertex colors
    char        pad[16];
} CUSTOMVERT;

enum MoveDirection
{
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_FORWARD,
    MOVE_BACK,
    MOVE_UP,
    MOVE_DOWN,

    ROTATE_LEFT,
    ROTATE_RIGHT,
    ROTATE_UP,
    ROTATE_DOWN
};

enum CameraView
{
    VIEW_PERSPECTIVE=1,
    VIEW_TOP,
    VIEW_LOS
};


//=====================================
// PROTOTYPES
//=====================================

LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wparam, LPARAM lparam );
bool InitWindow( void );
bool InitDX( void );
bool InitHeightmap( void );
bool InitCursor( void );
bool InitCamera( void );
bool ProcessInput( void );
void ViewCamera( CameraView view );
void MoveCamera( MoveDirection move, float scale );
void RotateCamera( MoveDirection move, float scale );
void MoveCursor( MoveDirection move );
bool ResetDX( void );
bool Render( void );
void KillResources( void );
void KillDX( void );

void LOSPointWholeMap();
bool LOSBuildGeometry( unsigned char *los );


//=====================================
// GLOBALS
//=====================================

// windows
HINSTANCE               g_hApplication;
HWND                    g_hWindow;
// dx
IDirect3D9             *g_pD3DObject;
IDirect3DDevice9       *g_pD3DDevice;
D3DPRESENT_PARAMETERS   g_D3Dpp;
IDirect3DVertexBuffer9 *g_pHMVertices=0;
IDirect3DIndexBuffer9  *g_pHMIndices=0;
IDirect3DVertexBuffer9 *g_pLOSVertices=0;
IDirect3DIndexBuffer9  *g_pLOSIndices=0;
IDirect3DVertexBuffer9 *g_pCURSORVertices=0;
IDirect3DIndexBuffer9  *g_pCURSORIndices=0;
IDirectInput8          *g_pDIObject;
IDirectInputDevice8    *g_pDIDKeyboard;
IDirectInputDevice8    *g_pDIDMouse;
DIMOUSESTATE            g_MouseState;
// stuff
Timer                   g_Timer;
CameraView              g_CamView;
D3DXVECTOR3             g_CamPos;
D3DXVECTOR3             g_CamDir;
bool                    bWindowed   = true;
bool                    bWireframe  = false;
bool                    bFlatShaded = false;
unsigned int            nVertices = 0;
unsigned int            nIndices = 0;
unsigned int            nPolygons = 0;
// los
float                  *g_HeightMap = 0;
unsigned int            g_HMWidth = 0;
unsigned int            g_HMHeight = 0;
unsigned int            g_HMSize = 0;
unsigned int            g_LOSTimer = 0;
unsigned int            g_LOSIndex = 0;


//=====================================
// LOCAL FUNCTIONS
//=====================================

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow )
{
    // cache app instance
    g_hApplication = hInstance;    

    // setup
    SetupDebugConsole( 256, 512, 80, 64 );
    if ( !InitWindow() )    return 1;
    if ( !InitDX() )        return 1;    
    if ( !InitHeightmap() ) return 1;
    if ( !InitCursor() )    return 1;
    if ( !InitCamera() )    return 1;
    g_Timer.Reset();
    timerCreate(&g_LOSTimer);

    // set default viewpoint
    g_LOSIndex = (g_HMHeight>>1) * g_HMWidth + (g_HMWidth>>1);
    //g_LOSIndex = 100208;

    // for message loop
    MSG msg;
    ZeroMemory( &msg, sizeof( msg ) );

    // MAIN GAME LOOP!!
    while ( msg.message != WM_QUIT )
    {
        // check for messages
        if ( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }

        //LOSPointWholeMap();

        // stuff
        g_Timer.Update();
        ProcessInput();
        Render();

        // display fps
        char buffer[32];
        sprintf( buffer, "FPS: %d", g_Timer.GetFPS() );
        SetWindowText( g_hWindow, buffer );
    }

    // destroy
    KillDX();
    UnregisterClass( "D3D Application", g_hApplication );

    return 0;
}

LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wparam, LPARAM lparam )
{
    switch ( message )
    {
        case ( WM_DESTROY ):
        {
            PostQuitMessage( 0 );
        }
        break;

        case ( WM_KEYDOWN ):
        {
            // quit
            if ( GetAsyncKeyState( VK_ESCAPE ) )
            {
                PostQuitMessage( 0 );
                break;
            }

            // toggle camera view
            else if ( GetAsyncKeyState( '1' ) )
            {
                ViewCamera( VIEW_PERSPECTIVE );
            }
            else if ( GetAsyncKeyState( '2' ) )
            {
                ViewCamera( VIEW_TOP );
            }
            else if ( GetAsyncKeyState( '3' ) )
            {
                ViewCamera( VIEW_LOS );
            }

            // toggle shading mode
            else if ( GetAsyncKeyState( VK_F1 ) )
            {
                bFlatShaded = !bFlatShaded;
                if ( bFlatShaded )  g_pD3DDevice->SetRenderState( D3DRS_SHADEMODE, D3DSHADE_FLAT );
                else                g_pD3DDevice->SetRenderState( D3DRS_SHADEMODE, D3DSHADE_GOURAUD );
            }

            // toggle fill mode
            else if ( GetAsyncKeyState( VK_F2 ) )
            {
                if ( bWireframe )
                {
                    bWireframe = false;
                    g_pD3DDevice->SetRenderState( D3DRS_FILLMODE, D3DFILL_SOLID );
                    //g_pD3DDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE );
                    //g_pD3DDevice->SetRenderState( D3DRS_LIGHTING, false );
                }
                else
                {
                    bWireframe = true;
                    g_pD3DDevice->SetRenderState( D3DRS_FILLMODE, D3DFILL_WIREFRAME );
                    //g_pD3DDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE );
                    //g_pD3DDevice->SetRenderState( D3DRS_LIGHTING, true );
                }
            }

            // show LOS
            else if ( GetAsyncKeyState( VK_F3 ) )
            {
                LOSPointWholeMap();   
            }            

            // toggle fullscreen
            else if ( GetAsyncKeyState( VK_F5 ) )
            {
                bWindowed = !bWindowed;

                KillResources();
                ResetDX();
                InitHeightmap();
                InitCursor();
                InitCamera();

                SetWindowLong( g_hWindow, GWL_STYLE,
                    ( bWindowed ? WS_OVERLAPPEDWINDOW : ( WS_POPUP | WS_VISIBLE ) ) );

                // take a guess
                ShowWindow( g_hWindow, SW_SHOW );
                UpdateWindow( g_hWindow );
            }
        }
        break;

        default: break;
    }

    // always pass for further processing
    return DefWindowProc( hWnd, message, wparam, lparam );
}

bool InitWindow( void )
{
    // locals
    WNDCLASSEX  wndClass;

    // fill in window class and register
    ZeroMemory( &wndClass, sizeof( wndClass ) );
    wndClass.hInstance          = g_hApplication;
    wndClass.lpszClassName      = "D3D Application";
    wndClass.lpfnWndProc        = WndProc;
    wndClass.cbSize             = sizeof( WNDCLASSEX );
    wndClass.hCursor            = LoadCursor( NULL, IDC_ARROW );
    RegisterClassEx( &wndClass );

    // create our window
    g_hWindow = CreateWindow(
        "D3D Application",                                              // class name
        "LOS Viewer",                                                   // window title
        bWindowed ? WS_OVERLAPPEDWINDOW : ( WS_POPUP | WS_VISIBLE ),    // window style
        200,                                                            // x position
        100,                                                            // y position
        SCREEN_WIDTH,                                                   // width
        SCREEN_HEIGHT,                                                  // height
        NULL,                                                           // parent
        NULL,                                                           // menu
        g_hApplication,                                                 // instance
        NULL );                                                         // reserved

    // error check
    if ( !g_hWindow ) return false;

    // take a guess
    ShowWindow( g_hWindow, SW_SHOW );
    UpdateWindow( g_hWindow );

    // yay!
    return true;
}

bool InitDX( void )
{
    // local
    HRESULT result;

    // init
    g_pD3DObject = NULL;
    g_pD3DDevice = NULL;

    // create directx object
    g_pD3DObject = Direct3DCreate9( D3D_SDK_VERSION );

    // error check
    if ( !g_pD3DObject ) return false;

    // presentation parameters
    ZeroMemory( &g_D3Dpp, sizeof( g_D3Dpp ) );
    g_D3Dpp.hDeviceWindow           = g_hWindow;
    g_D3Dpp.Windowed                = bWindowed;
    g_D3Dpp.BackBufferFormat        = bWindowed ? D3DFMT_UNKNOWN : D3DFMT_X8R8G8B8;
    g_D3Dpp.BackBufferWidth         = SCREEN_WIDTH;
    g_D3Dpp.BackBufferHeight        = SCREEN_HEIGHT;
    g_D3Dpp.BackBufferCount         = 1;
    g_D3Dpp.EnableAutoDepthStencil  = true;
    g_D3Dpp.AutoDepthStencilFormat  = D3DFMT_D16;
    g_D3Dpp.SwapEffect              = D3DSWAPEFFECT_DISCARD;
    g_D3Dpp.PresentationInterval    = D3DPRESENT_INTERVAL_IMMEDIATE;

    // create our dx device
    result = g_pD3DObject->CreateDevice(
        D3DADAPTER_DEFAULT,
        D3DDEVTYPE_HAL,
        g_hWindow,
        D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_PUREDEVICE,
        &g_D3Dpp,
        &g_pD3DDevice );
    if ( FAILED( result ) ) return false;

    // default render states
    g_pD3DDevice->SetRenderState( D3DRS_FILLMODE, D3DFILL_SOLID );
    g_pD3DDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_CW );
    g_pD3DDevice->SetRenderState( D3DRS_SHADEMODE, D3DSHADE_GOURAUD );
    g_pD3DDevice->SetRenderState( D3DRS_LIGHTING, false );
    //g_pD3DDevice->SetRenderState( D3DRS_SPECULARENABLE, true );
    g_pD3DDevice->SetRenderState( D3DRS_AMBIENT,   0xff808080 );
    g_pD3DDevice->SetRenderState( D3DRS_ZENABLE,  true );
    g_pD3DDevice->SetRenderState( D3DRS_SRCBLEND, D3DBLEND_SRCALPHA );
    g_pD3DDevice->SetRenderState( D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA );
    g_pD3DDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, true );

    // init dinput
    DirectInput8Create(
        g_hApplication,
        DIRECTINPUT_VERSION,
        IID_IDirectInput8,
        (void**)&g_pDIObject,
        NULL);

    // setup keyboard
    g_pDIObject->CreateDevice( GUID_SysKeyboard, &g_pDIDKeyboard, NULL);
    g_pDIDKeyboard->SetDataFormat( &c_dfDIKeyboard );
    g_pDIDKeyboard->SetCooperativeLevel( g_hWindow, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE );

    // setup mouse
    g_pDIObject->CreateDevice( GUID_SysMouse, &g_pDIDMouse, NULL );
    g_pDIDMouse->SetDataFormat( &c_dfDIMouse );
    g_pDIDMouse->SetCooperativeLevel( g_hWindow, DISCL_FOREGROUND | DISCL_EXCLUSIVE );

    // setup CUDA
    initDevice();

    return true;
}

bool ResetDX( void )
{
    // presentation parameters
    ZeroMemory( &g_D3Dpp, sizeof( g_D3Dpp ) );
    g_D3Dpp.Windowed                = bWindowed;
    g_D3Dpp.BackBufferFormat        = bWindowed ? D3DFMT_UNKNOWN : D3DFMT_X8R8G8B8;
    g_D3Dpp.BackBufferWidth         = SCREEN_WIDTH;
    g_D3Dpp.BackBufferHeight        = SCREEN_HEIGHT;
    g_D3Dpp.EnableAutoDepthStencil  = true;
    g_D3Dpp.AutoDepthStencilFormat  = D3DFMT_D16;
    g_D3Dpp.SwapEffect              = D3DSWAPEFFECT_DISCARD;
    g_D3Dpp.PresentationInterval    = D3DPRESENT_INTERVAL_IMMEDIATE;

    // reset device
    g_pD3DDevice->Reset( &g_D3Dpp );

    // reset render states
    g_pD3DDevice->SetRenderState( D3DRS_FILLMODE, D3DFILL_SOLID );
    g_pD3DDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE );
    g_pD3DDevice->SetRenderState( D3DRS_SHADEMODE, D3DSHADE_GOURAUD );
    g_pD3DDevice->SetRenderState( D3DRS_LIGHTING, false );
    //g_pD3DDevice->SetRenderState( D3DRS_SPECULARENABLE, true );
    g_pD3DDevice->SetRenderState( D3DRS_AMBIENT,   0xff808080 );
    g_pD3DDevice->SetRenderState( D3DRS_ZENABLE,  true );
    g_pD3DDevice->SetRenderState( D3DRS_SRCBLEND, D3DBLEND_SRCALPHA );
    g_pD3DDevice->SetRenderState( D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA );
    g_pD3DDevice->SetRenderState( D3DRS_ALPHABLENDENABLE, true );

    // init dinput
    DirectInput8Create(
        g_hApplication,
        DIRECTINPUT_VERSION,
        IID_IDirectInput8,
        (void**)&g_pDIObject,
        NULL);

    // setup keyboard
    g_pDIObject->CreateDevice( GUID_SysKeyboard, &g_pDIDKeyboard, NULL);
    g_pDIDKeyboard->SetDataFormat( &c_dfDIKeyboard );
    g_pDIDKeyboard->SetCooperativeLevel( g_hWindow, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE );

    // setup mouse
    g_pDIObject->CreateDevice( GUID_SysMouse, &g_pDIDMouse, NULL );
    g_pDIDMouse->SetDataFormat( &c_dfDIMouse );
    g_pDIDMouse->SetCooperativeLevel( g_hWindow, DISCL_FOREGROUND | DISCL_EXCLUSIVE );

    return true;
}

bool InitHeightmap( void )
{
    HRESULT         result;
    CUSTOMVERT     *vertices;
    DWORD          *indices;
    VOID           *pBufferData;
    unsigned int    memverts, memidxs;
    unsigned int    idx;
    Pixel          *pixels;    
    BMPLoader       loader;
    
    // load    
    loader.Load( HEIGHT_MAP );
    pixels = loader.GetPixels();    

    // init heightmap stuff
    g_HMWidth   = loader.GetWidth();
    g_HMHeight  = loader.GetHeight();
    g_HMSize    = loader.GetNumPixels();    
    g_HeightMap = new float[g_HMSize];     
    for (unsigned int i=0; i<g_HMSize; i++) g_HeightMap[i] = pixels[i].r;

    // init geometry stuff
    nVertices = g_HMSize;
    nPolygons = ((g_HMWidth-1)*(g_HMHeight-1))<<1;
    nIndices  = nPolygons * 3;
    memverts  = nVertices * sizeof(CUSTOMVERT);
    memidxs   = nIndices  * sizeof( DWORD );
    vertices  = new CUSTOMVERT[nVertices]; 
    indices   = new DWORD[nIndices];
    
    // set vertices from height map pixels
    idx = 0;
    for ( unsigned int j=0; j<g_HMHeight; j++ )
    {
        for ( unsigned int i=0; i<g_HMWidth; i++ )
        {
            vertices[idx].pos.x = (float)i - (g_HMWidth>>1);
            vertices[idx].pos.y = pixels[idx].r;
            vertices[idx].pos.z = (float)j - (g_HMHeight>>1);
            vertices[idx].color = D3DCOLOR_RGBA( pixels[idx].r, pixels[idx].g, pixels[idx].b, 255 );
            idx++;
        }
    }

    //// set normals by average surrounding vertices
    //D3DXVECTOR3 neighbors[6], average, tmpvec;
    //float brightness;
    //unsigned int val;
    //idx = 0;
    //for ( unsigned int j=0; j<g_HMHeight; j++ )
    //{
    //    for ( unsigned int i=0; i<g_HMWidth; i++ )
    //    {
    //        idx = j*g_HMWidth+i;

    //        // gather neighbor vectors
    //        unsigned int idx2, n=0;
    //        for ( unsigned int ny=-1; ny<2; ny++ )
    //        {
    //            if (j+ny<0||j+ny>=g_HMHeight) continue;
    //            for ( unsigned int nx=-1; nx<2; nx++ )
    //            {
    //                if (nx==ny) continue;
    //                if (i+nx<0||i+nx>=g_HMWidth) continue;

    //                idx2 = (j+ny)*g_HMWidth+(i+nx);                    
    //                neighbors[n++] = vertices[idx2].pos - vertices[idx].pos;
    //            }
    //        }

    //        average = D3DXVECTOR3(0,1,0);
    //        D3DXVec3Cross( &tmpvec, &neighbors[1], &neighbors[0] ); average += tmpvec;
    //        D3DXVec3Cross( &tmpvec, &neighbors[0], &neighbors[2] ); average += tmpvec;
    //        D3DXVec3Cross( &tmpvec, &neighbors[2], &neighbors[4] ); average += tmpvec;
    //        D3DXVec3Cross( &tmpvec, &neighbors[4], &neighbors[5] ); average += tmpvec;
    //        D3DXVec3Cross( &tmpvec, &neighbors[5], &neighbors[3] ); average += tmpvec;
    //        D3DXVec3Cross( &tmpvec, &neighbors[3], &neighbors[1] ); average += tmpvec;
    //        
    //        //vertices[idx].normal = D3DXVECTOR3(0,1,0);
    //        //vertices[idx].normal = average;            
    //        //brightness = D3DXVec3Dot( &average, &D3DXVECTOR3(0,1,0) );            
    //        //val = (vertices[idx].color & 0x000000ff) * brightness;
    //        //vertices[idx].color = D3DCOLOR_RGBA( val,val,val,255 );            
    //    }
    //}

    // set indices
    idx = 0;
    for ( unsigned int j=0; j<g_HMHeight-1; j++ )
    {
        for ( unsigned int i=0; i<g_HMWidth-1; i++ )
        {
            // poly 1
            indices[idx++] = ((j  )*g_HMWidth) + (i  );
            indices[idx++] = ((j+1)*g_HMWidth) + (i  );
            indices[idx++] = ((j  )*g_HMWidth) + (i+1);

            // poly 2
            indices[idx++] = ((j+1)*g_HMWidth) + (i  );
            indices[idx++] = ((j+1)*g_HMWidth) + (i+1);
            indices[idx++] = ((j  )*g_HMWidth) + (i+1);
        }
    }

    // create vertex buffer
    result = g_pD3DDevice->CreateVertexBuffer(
        memverts,                           // byte size of buffer
        0,                                  // flags
        D3DFVF_CUSTOM,                      // FVF format
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pHMVertices,                     // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // create index buffer
    result = g_pD3DDevice->CreateIndexBuffer(
        memidxs,                            // bytes size of buffer
        0,                                  // flags
        D3DFMT_INDEX32,                     // format of each index
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pHMIndices,                      // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // copy model verts into our vertex buffer
    g_pHMVertices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, vertices, memverts );
    g_pHMVertices->Unlock();

    // copy model indices into our index buffer
    g_pHMIndices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, indices, memidxs );
    g_pHMIndices->Unlock();

    // cleanup
    delete [] vertices;
    delete [] indices;    

    return true;
}

bool InitCursor( void )
{
    HRESULT     result;
    CUSTOMVERT  vertices[5];
    WORD        indices[18];
    VOID       *pBufferData;
    int         idx=0;    

    // verts
    vertices[0].pos = D3DXVECTOR3( 0,0,0 );          vertices[0].color = D3DCOLOR_RGBA( 0, 0, 255, 200 );
    vertices[1].pos = D3DXVECTOR3( -2.5,10,-2.5 );   vertices[1].color = D3DCOLOR_RGBA( 255, 0, 0, 200 );
    vertices[2].pos = D3DXVECTOR3(  2.5,10,-2.5 );   vertices[2].color = D3DCOLOR_RGBA( 255, 0, 0, 200 );
    vertices[3].pos = D3DXVECTOR3( -2.5,10, 2.5 );   vertices[3].color = D3DCOLOR_RGBA( 255, 0, 0, 200 );
    vertices[4].pos = D3DXVECTOR3(  2.5,10, 2.5 );   vertices[4].color = D3DCOLOR_RGBA( 255, 0, 0, 200 );

    // indices
    indices[idx++] = 4; indices[idx++] = 1; indices[idx++] = 3;
    indices[idx++] = 4; indices[idx++] = 2; indices[idx++] = 1;
    indices[idx++] = 0; indices[idx++] = 4; indices[idx++] = 3;
    indices[idx++] = 0; indices[idx++] = 2; indices[idx++] = 4;
    indices[idx++] = 0; indices[idx++] = 1; indices[idx++] = 2;
    indices[idx++] = 0; indices[idx++] = 3; indices[idx++] = 1;

    // create vertex buffer
    result = g_pD3DDevice->CreateVertexBuffer(
        5 * sizeof(CUSTOMVERT),                           // byte size of buffer
        0,                                  // flags
        D3DFVF_CUSTOM,                      // FVF format
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pCURSORVertices,                     // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // create index buffer
    result = g_pD3DDevice->CreateIndexBuffer(
        18 * sizeof(WORD),                            // bytes size of buffer
        0,                                  // flags
        D3DFMT_INDEX16,                     // format of each index
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pCURSORIndices,                      // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // copy model verts into our vertex buffer
    g_pCURSORVertices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, vertices, 5 * sizeof(CUSTOMVERT) );
    g_pCURSORVertices->Unlock();

    // copy model indices into our index buffer
    g_pCURSORIndices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, indices, 18 * sizeof(WORD) );
    g_pCURSORIndices->Unlock(); 

    return true;
}

bool InitCamera( void )
{
    // camera variables
    D3DXMATRIX  matView;    // the view matrix
    D3DXMATRIX  matProj;    // the projection matrix

    // init camera pos & viewing direction
    g_CamPos = D3DXVECTOR3( -21.0f, 514.0f, 410.0f );
    D3DXVec3Subtract( &g_CamDir, &D3DXVECTOR3( 0,0,-128 ), &g_CamPos );
    D3DXVec3Normalize( &g_CamDir, &g_CamDir );
    //g_CamDir = D3DXVECTOR3( 0.0f, 0.0f, -1.0f );

    // build view matrix
	D3DXMatrixLookAtRH(
        &matView,
        &g_CamPos,                           // cam position
        &D3DXVECTOR3( 0,0,-128 ),     // look at point
        &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );   // up direction

    // build projection matrix
    D3DXMatrixPerspectiveFovRH(
        &matProj,                                   // stored matrix
        D3DX_PI/3,                                  // fov
        (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT,   // aspect ratio
        0.1f,                                       // near plane
        2000.0f );                                  // far plane

    // set the view matrix
	g_pD3DDevice->SetTransform( D3DTS_VIEW, &matView );

    // set projection matrix
    g_pD3DDevice->SetTransform( D3DTS_PROJECTION, &matProj );

    g_CamView = VIEW_PERSPECTIVE;

    return true;
}

bool ProcessInput( void )
{
    // error check
    if ( !g_pDIDKeyboard )  return false;
    if ( !g_pDIDMouse )     return false;

    // poll 60 times a second
    static float timePrev = 0;
    if ( g_Timer.GetTime() - timePrev > .016f )
    {
        // locals
        char buffer[ 256 ] = { 0 };

        // read keyboard
        g_pDIDKeyboard->Acquire();
        g_pDIDKeyboard->GetDeviceState( sizeof( buffer ), buffer );

        // read mouse
        memset( &g_MouseState, 0, sizeof( DIMOUSESTATE ) );
        g_pDIDMouse->Acquire();
        g_pDIDMouse->GetDeviceState( sizeof( DIMOUSESTATE ), &g_MouseState );

        // move cursor
        if ( g_CamView == VIEW_TOP )
        {
            if ( KEYDOWN( buffer, DIK_A ) )     MoveCursor( MOVE_LEFT );
            if ( KEYDOWN( buffer, DIK_D ) )     MoveCursor( MOVE_RIGHT );
            if ( KEYDOWN( buffer, DIK_W ) )     MoveCursor( MOVE_UP );
            if ( KEYDOWN( buffer, DIK_S ) )     MoveCursor( MOVE_DOWN );
            if ( KEYDOWN( buffer, DIK_SPACE ) ) MoveCursor( MOVE_FORWARD );
        }
        
        else if ( g_CamView == VIEW_PERSPECTIVE || g_CamView == VIEW_LOS )
        {
            // pan camera
            if ( KEYDOWN( buffer, DIK_A ) ) MoveCamera( MOVE_LEFT,      CAMERA_PAN_SPEED );
            if ( KEYDOWN( buffer, DIK_S ) ) MoveCamera( MOVE_BACK,      CAMERA_PAN_SPEED );
            if ( KEYDOWN( buffer, DIK_D ) ) MoveCamera( MOVE_RIGHT,     CAMERA_PAN_SPEED );
            if ( KEYDOWN( buffer, DIK_W ) ) MoveCamera( MOVE_FORWARD,   CAMERA_PAN_SPEED );
            if ( KEYDOWN( buffer, DIK_Q ) ) MoveCamera( MOVE_DOWN,      CAMERA_PAN_SPEED );
            if ( KEYDOWN( buffer, DIK_E ) ) MoveCamera( MOVE_UP,        CAMERA_PAN_SPEED );            
       
            // rotate camera
            if ( g_MouseState.lX < 0 )  RotateCamera( ROTATE_LEFT,  ((float)-g_MouseState.lX) * CAMERA_ROT_SPEED );
            if ( g_MouseState.lX > 0 )  RotateCamera( ROTATE_RIGHT, ((float)g_MouseState.lX) * CAMERA_ROT_SPEED  );
            if ( g_MouseState.lY > 0 )  RotateCamera( ROTATE_DOWN,  ((float)g_MouseState.lY) * CAMERA_ROT_SPEED  );
            if ( g_MouseState.lY < 0 )  RotateCamera( ROTATE_UP,    ((float)-g_MouseState.lY) * CAMERA_ROT_SPEED  );

            // as soon as the camera moves, we are back in perspective
            if ( g_CamView == VIEW_LOS ) g_CamView = VIEW_PERSPECTIVE;
        }

        //if ( KEYDOWN( buffer, DIK_LEFT ) )  RotateCamera( ROTATE_LEFT, 0.0001f );
        //if ( KEYDOWN( buffer, DIK_RIGHT ) ) RotateCamera( ROTATE_RIGHT, 0.0001f );
        //if ( KEYDOWN( buffer, DIK_UP ) )    RotateCamera( ROTATE_UP, 0.0001f );
        //if ( KEYDOWN( buffer, DIK_DOWN ) )  RotateCamera( ROTATE_DOWN, 0.0001f );

        //g_CursorPos.x += g_MouseState.lX;
        //g_CursorPos.y += g_MouseState.lY;

        //// alter rotation rate
        //if ( g_MouseState.rgbButtons[ 0 ] & 0x80 )          g_ObjVelocity.y += .005f;
        //else if ( g_MouseState.rgbButtons[ 1 ] & 0x80 )     g_ObjVelocity.y += .005f;
        //else if ( g_MouseState.rgbButtons[ 2 ] & 0x80 )     g_ObjVelocity.y = 0;

        timePrev = g_Timer.GetTime();
    }

    return true;
}

void ViewCamera( CameraView view )
{
    D3DXMATRIX  matView, matProj;
    D3DXVECTOR3 lookAtPoint;

    // view is already set
    if ( g_CamView == view ) return;
    
    // handle type
    if ( view == VIEW_PERSPECTIVE )
    {
        g_CamPos = D3DXVECTOR3( 0.0f, 400.0f, 400.0f );
        D3DXVec3Subtract( &g_CamDir, &D3DXVECTOR3( 0,0,-256 ), &g_CamPos );
        D3DXVec3Normalize( &g_CamDir, &g_CamDir );
        D3DXVec3Add( &lookAtPoint, &g_CamPos, &g_CamDir );

        D3DXMatrixLookAtRH( &matView, &g_CamPos, &lookAtPoint, &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );
        g_pD3DDevice->SetTransform( D3DTS_VIEW, &matView );
        D3DXMatrixPerspectiveFovRH( &matProj, D3DX_PI/3, (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT, 0.1f, 2000.0f );
        g_pD3DDevice->SetTransform( D3DTS_PROJECTION, &matProj );
    }
    else if ( view == VIEW_TOP )
    {
        g_CamPos = D3DXVECTOR3( 0.0f, 600.0f, 0.0f );
        D3DXVec3Subtract( &g_CamDir, &D3DXVECTOR3( 0,10,-10 ), &g_CamPos );
        D3DXVec3Normalize( &g_CamDir, &g_CamDir );   
        D3DXVec3Add( &lookAtPoint, &g_CamPos, &g_CamDir );

        D3DXMatrixLookAtRH( &matView, &g_CamPos, &lookAtPoint, &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );
        g_pD3DDevice->SetTransform( D3DTS_VIEW, &matView );        
        D3DXMatrixOrthoRH(&matProj, SCREEN_WIDTH, SCREEN_HEIGHT, 0.0f, 2000.0f);
        g_pD3DDevice->SetTransform(D3DTS_PROJECTION, &matProj);        

    }
    else if ( view == VIEW_LOS )
    {
        g_CamPos.x = fmod((float)g_LOSIndex, (float)g_HMWidth) - (float)(g_HMWidth>>1);
        g_CamPos.y = g_HeightMap[g_LOSIndex];
        g_CamPos.z = ((float)g_LOSIndex / (float)g_HMWidth) - (float)(g_HMWidth>>1);
        D3DXVec3Subtract( &g_CamDir, &D3DXVECTOR3( g_CamPos.x,g_HeightMap[g_LOSIndex],g_CamPos.z+100.0f ), &g_CamPos );
        D3DXVec3Normalize( &g_CamDir, &g_CamDir );
        D3DXVec3Add( &lookAtPoint, &g_CamPos, &g_CamDir );

        D3DXMatrixLookAtRH( &matView, &g_CamPos, &lookAtPoint, &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );
        g_pD3DDevice->SetTransform( D3DTS_VIEW, &matView );
        D3DXMatrixPerspectiveFovRH( &matProj, D3DX_PI/3, (float)SCREEN_WIDTH/(float)SCREEN_HEIGHT, 0.1f, 2000.0f );
        g_pD3DDevice->SetTransform( D3DTS_PROJECTION, &matProj );
    }

    // set new view
    g_CamView = view;
}

void MoveCamera( MoveDirection move, float scale )
{
    // locals
    D3DXMATRIX  viewMat;
    D3DXVECTOR3 lookAtPoint;
    D3DXVECTOR3 camRight, camUp, camForward;

    // compute camera local system
    D3DXVec3Cross( &camRight, &g_CamDir, &D3DXVECTOR3( 0.0f, 1.0f, 0.0f ) );
    D3DXVec3Cross( &camUp, &camRight, &g_CamDir );

    // scale movement
    D3DXVec3Scale( &camForward, &g_CamDir, scale );
    D3DXVec3Scale( &camRight,   &camRight, scale );
    D3DXVec3Scale( &camUp,      &camUp,    scale );

    // apply correct camera move
    switch ( move )
    {
    case ( MOVE_FORWARD ):  D3DXVec3Add( &g_CamPos, &g_CamPos, &camForward ); break;
    case ( MOVE_BACK ):     D3DXVec3Add( &g_CamPos, &g_CamPos, &(-camForward) ); break;
    case ( MOVE_LEFT ):     D3DXVec3Add( &g_CamPos, &g_CamPos, &(-camRight) ); break;
    case ( MOVE_RIGHT ):    D3DXVec3Add( &g_CamPos, &g_CamPos, &camRight ); break;
    case ( MOVE_UP ):       D3DXVec3Add( &g_CamPos, &g_CamPos, &camUp ); break;
    case ( MOVE_DOWN ):     D3DXVec3Add( &g_CamPos, &g_CamPos, &(-camUp) ); break;
    };

    // new look at point
    D3DXVec3Add( &lookAtPoint, &g_CamPos, &g_CamDir );

    // build view matrix
    D3DXMatrixLookAtRH(
        &viewMat,
        &g_CamPos,                          // cam position
        &lookAtPoint,                       // look at point
        &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );   // up direction

    // get camera matrix
    g_pD3DDevice->SetTransform( D3DTS_VIEW, &viewMat );
}

void RotateCamera( MoveDirection move, float scale )
{
    // locals
    D3DXMATRIX  viewMat, camMat, rotMat;
    D3DXVECTOR3 camRight, camUp, lookAtPoint;
    D3DXVECTOR4 newDir;

    // compute camera local system
    D3DXVec3Cross( &camRight, &g_CamDir, &D3DXVECTOR3( 0.0f, 1.0f, 0.0f ) );
    D3DXVec3Cross( &camUp, &camRight, &g_CamDir );

    // setup camera current matrix
    D3DXMatrixIdentity( &camMat );
    camMat._11 = camRight.x;
    camMat._12 = camRight.y;
    camMat._13 = camRight.z;
    camMat._21 = camUp.x;
    camMat._22 = camUp.y;
    camMat._23 = camUp.z;
    camMat._31 = -g_CamDir.x;
    camMat._32 = -g_CamDir.y;
    camMat._33 = -g_CamDir.z;

    // setup rotation matrix
    D3DXMatrixIdentity( &rotMat );
    switch ( move )
    {
    case ( ROTATE_LEFT ):   D3DXMatrixRotationY( &rotMat,  scale ); break;
    case ( ROTATE_RIGHT ):  D3DXMatrixRotationY( &rotMat, -scale ); break;
    case ( ROTATE_UP ):     D3DXMatrixRotationX( &rotMat,  scale ); break;
    case ( ROTATE_DOWN ):   D3DXMatrixRotationX( &rotMat, -scale ); break;
    };

    // rotate camera relative to view direction
    D3DXMatrixMultiply( &camMat, &rotMat, &camMat );
    g_CamDir.x = -camMat._31;
    g_CamDir.y = -camMat._32;
    g_CamDir.z = -camMat._33;
    D3DXVec3Normalize( &g_CamDir, &g_CamDir );

    // new look at point
    D3DXVec3Add( &lookAtPoint, &g_CamPos, &g_CamDir );

    // build view matrix
    D3DXMatrixLookAtRH(
        &viewMat,
        &g_CamPos,                          // cam position
        &lookAtPoint,                       // look at point
        &D3DXVECTOR3(0.0f, 1.0f, 0.0f) );   // up direction

    // get camera matrix
    g_pD3DDevice->SetTransform( D3DTS_VIEW, &viewMat );
}

void MoveCursor( MoveDirection move )
{
    int x,y;

    // release previous line-of-sight mesh
    if ( g_pLOSVertices )
    {
        g_pLOSVertices->Release();
        g_pLOSVertices = NULL;
    }
    if ( g_pLOSIndices )
    {
        g_pLOSIndices->Release();
        g_pLOSIndices = NULL;
    }    

    // extract current viewpoint
    x = g_LOSIndex % g_HMWidth;
    y = g_LOSIndex / g_HMWidth;    

    switch ( move )
    {
    case ( MOVE_LEFT ):     x = max(0,x-2);             break;
    case ( MOVE_RIGHT ):    x = min(g_HMWidth-1,x+2);   break;
    case ( MOVE_UP ):       y = max(0,y-2);             break;
    case ( MOVE_DOWN ):     y = min(g_HMHeight-1,y+2);  break;
    case ( MOVE_FORWARD ):
    {        
        //x = (g_HMWidth>>1);
        //y = (g_HMHeight>>1);
        x = 100208 % g_HMWidth; // hack for my 512x512 map
        y = 100208 / g_HMWidth;
    }
    break;
    };

    // set new viewpoint
    g_LOSIndex = y * g_HMWidth + x;
}

bool Render( void )
{
    D3DXMATRIX tmpMat, finalMat;

    // bad
    if ( !g_pD3DDevice ) return false;

    // clear the screen
    g_pD3DDevice->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB( 155, 205, 255 ), 1.0f, 0 );

    // start 3d scene
    if ( SUCCEEDED( g_pD3DDevice->BeginScene() ) )
    {
        // terrain
        g_pD3DDevice->SetStreamSource( 0, g_pHMVertices, 0, sizeof(CUSTOMVERT) );
        g_pD3DDevice->SetIndices( g_pHMIndices );
        g_pD3DDevice->SetFVF( D3DFVF_CUSTOM );
        D3DXMatrixIdentity( &finalMat );
        g_pD3DDevice->SetTransform( D3DTS_WORLD, &finalMat );
        g_pD3DDevice->DrawIndexedPrimitive( D3DPT_TRIANGLELIST, 0, 0, nVertices, 0, nPolygons );

        // line-of-sight
        if (g_pLOSVertices && g_pLOSIndices)
        {
            g_pD3DDevice->SetStreamSource( 0, g_pLOSVertices, 0, sizeof(CUSTOMVERT) );
            g_pD3DDevice->SetIndices( g_pLOSIndices );
            g_pD3DDevice->SetFVF( D3DFVF_CUSTOM );
            D3DXMatrixIdentity( &finalMat );
            g_pD3DDevice->SetTransform( D3DTS_WORLD, &finalMat );
            g_pD3DDevice->DrawIndexedPrimitive( D3DPT_TRIANGLELIST, 0, 0, nVertices, 0, nPolygons );
        }

        // cursor
        g_pD3DDevice->SetStreamSource( 0, g_pCURSORVertices, 0, sizeof(CUSTOMVERT) );
        g_pD3DDevice->SetIndices( g_pCURSORIndices );
        g_pD3DDevice->SetFVF( D3DFVF_CUSTOM );        
        D3DXMatrixIdentity( &finalMat );
        D3DXMatrixRotationY( &tmpMat, g_Timer.GetTime() );
        D3DXMatrixMultiply( &finalMat, &finalMat, &tmpMat );
        finalMat._41 = fmod((float)g_LOSIndex, (float)g_HMWidth) - (float)(g_HMWidth>>1);
        finalMat._42 = g_HeightMap[g_LOSIndex] + 5.0f + (2.0f*sin(6.0f*g_Timer.GetTime()));
        finalMat._43 = ((float)g_LOSIndex / (float)g_HMWidth) - (float)(g_HMWidth>>1);
        g_pD3DDevice->SetTransform( D3DTS_WORLD, &finalMat );
        g_pD3DDevice->DrawIndexedPrimitive( D3DPT_TRIANGLELIST, 0, 0, 5, 0, 6 );

        // end 3d scene
        g_pD3DDevice->EndScene();
    }

    // show back buffer
    g_pD3DDevice->Present( NULL, NULL, NULL, NULL );

    return true;
}

void KillResources( void )
{
    // release dinput keyboard
    if ( g_pDIDKeyboard )
    {
        g_pDIDKeyboard->Unacquire();
        g_pDIDKeyboard->Release();
        g_pDIDKeyboard = NULL;
    }

    // release dinput mouse
    if ( g_pDIDMouse )
    {
        g_pDIDMouse->Unacquire();
        g_pDIDMouse->Release();
        g_pDIDMouse = NULL;
    }

    // release dinput object
    if ( g_pDIObject )
    {
        g_pDIObject->Release();
        g_pDIObject = NULL;
    }

    // release line-of-sight mesh
    if ( g_pLOSVertices )
    {
        g_pLOSVertices->Release();
        g_pLOSVertices = NULL;
    }
    if ( g_pLOSIndices )
    {
        g_pLOSIndices->Release();
        g_pLOSIndices = NULL;
    }

    // release cursor
    if ( g_pCURSORVertices )
    {
        g_pCURSORVertices->Release();
        g_pCURSORVertices = NULL;
    }
    if ( g_pCURSORIndices )
    {
        g_pCURSORIndices->Release();
        g_pCURSORIndices = NULL;
    }

    // release terrain
    if ( g_pHMVertices )
    {
        g_pHMVertices->Release();
        g_pHMVertices = NULL;
    }
    if ( g_pHMIndices )
    {
        g_pHMIndices->Release();
        g_pHMIndices = NULL;
    }

    // release heightmap heights
    if ( g_HeightMap )
    {
        delete [] g_HeightMap;
        g_HeightMap = 0;
    }
}

void KillDX( void )
{
    // release assets
    KillResources();

    // release device
    if ( g_pD3DDevice )
    {
        g_pD3DDevice->Release();
        g_pD3DDevice = NULL;
    }

    // release dx object
    if ( g_pD3DObject )
    {
        g_pD3DObject->Release();
        g_pD3DObject = NULL;
    }
}



//=====================================
// LOS FUNCTIONS
//=====================================

void LOSPointWholeMap()
{        
    float tgpu=0, tcpu=0;
    double linesteps = sqrt((double)((g_HMWidth*g_HMWidth)+(g_HMHeight*g_HMHeight)))/2.0;
    double gflops    = ((((double)g_HMSize) * linesteps * 24.0) + (((double)g_HMSize) * (5.0+10.0)))/((double)(1024 * 1024 * 1024));
    unsigned char *los = new unsigned char[g_HMSize];

    // line-of-sight
    memset(los,LOS_VISIBLE,g_HMSize);
    timerStart(g_LOSTimer);
    losPointRectDEVICE(los, g_HeightMap, g_HMWidth,g_HMHeight, g_LOSIndex);
    tgpu = timerStop(g_LOSTimer);
    printf("GPU: %lfs (%3.5lf)\n", tgpu, gflops/tgpu);

    // setup geometry
    //static bool initd = false;
    //if (!initd) { LOSBuildGeometry(los); initd=true; }
    LOSBuildGeometry(los);

    // cleanup
    if (los)
    {
        delete [] los;
        los = 0;
    }
}

bool LOSBuildGeometry( unsigned char *los )
{
    HRESULT         result;
    CUSTOMVERT     *vertices;
    DWORD          *indices;
    VOID           *pBufferData;
    unsigned int    memverts, memidxs;
    unsigned int    idx;    

    // init geometry stuff
    nVertices = g_HMSize;
    nPolygons = ((g_HMWidth-1)*(g_HMHeight-1))<<1;
    nIndices  = nPolygons * 3;
    memverts  = nVertices * sizeof(CUSTOMVERT);
    memidxs   = nIndices  * sizeof(DWORD);
    vertices  = new CUSTOMVERT[nVertices]; 
    indices   = new DWORD[nIndices];

    // set vertices from los heights
    idx = 0;
    for ( unsigned int j=0; j<g_HMHeight; j++ )
    {
        for ( unsigned int i=0; i<g_HMWidth; i++ )
        {
            vertices[idx].pos.x = (float)i - (g_HMWidth>>1);
            vertices[idx].pos.y = g_HeightMap[idx] + 0.5f;
            vertices[idx].pos.z = (float)j - (g_HMHeight>>1);

            if (los[idx] == LOS_VISIBLE) vertices[idx].color = D3DCOLOR_RGBA(255, 255, 0, 200);
            else                         vertices[idx].color = D3DCOLOR_RGBA(255, 255, 0, 0);

            idx++;
        }
    }  

    // set indices
    idx = 0;
    for ( unsigned int j=0; j<g_HMHeight-1; j++ )
    {
        for ( unsigned int i=0; i<g_HMWidth-1; i++ )
        {
            // poly 1
            indices[idx++] = ((j  )*g_HMWidth) + (i  );
            indices[idx++] = ((j+1)*g_HMWidth) + (i  );
            indices[idx++] = ((j  )*g_HMWidth) + (i+1);

            // poly 2
            indices[idx++] = ((j+1)*g_HMWidth) + (i  );
            indices[idx++] = ((j+1)*g_HMWidth) + (i+1);
            indices[idx++] = ((j  )*g_HMWidth) + (i+1);
        }
    }

    // release previous line-of-sight mesh
    if ( g_pLOSVertices )
    {
        g_pLOSVertices->Release();
        g_pLOSVertices = NULL;
    }
    if ( g_pLOSIndices )
    {
        g_pLOSIndices->Release();
        g_pLOSIndices = NULL;
    }

    // create vertex buffer
    result = g_pD3DDevice->CreateVertexBuffer(
        memverts,                           // byte size of buffer
        0,                                  // flags
        D3DFVF_CUSTOM,                      // FVF format
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pLOSVertices,                     // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // create index buffer
    result = g_pD3DDevice->CreateIndexBuffer(
        memidxs,                            // bytes size of buffer
        0,                                  // flags
        D3DFMT_INDEX32,                     // format of each index
        D3DPOOL_DEFAULT,                    // memory pool
        &g_pLOSIndices,                      // stored buffer
        NULL );                             // reserved
    if ( FAILED( result ) ) return false;

    // copy model verts into our vertex buffer
    g_pLOSVertices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, vertices, memverts );
    g_pLOSVertices->Unlock();

    // copy model indices into our index buffer
    g_pLOSIndices->Lock( 0, 0, &pBufferData, D3DLOCK_READONLY );
    memcpy( pBufferData, indices, memidxs );
    g_pLOSIndices->Unlock();

    // cleanup
    delete [] vertices;
    delete [] indices;

    return true;
}
