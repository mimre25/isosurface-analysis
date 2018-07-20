#include "isosurfaces_cuda/hdr/MarchingCube.h"
#include "defines.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <utils/hdr/Report.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include "isosurfaces_cuda/hdr/thrustWrapper.h"

//a2fVertexOffset lists the positions, relative to vertex0, of each of the 8 vertices of a cube
static const int a2fVertexOffset[8][3] =
		{
				{0, 0, 0},
				{1, 0, 0},
				{1, 1, 0},
				{0, 1, 0},
				{0, 0, 1},
				{1, 0, 1},
				{1, 1, 1},
				{0, 1, 1}
		};

//a2iEdgeConnection lists the index of the endpoint vertices for each of the 12 edges of the cube
static const int a2iEdgeConnection[12][2] =
		{
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 0},
				{4, 5},
				{5, 6},
				{6, 7},
				{7, 4},
				{0, 4},
				{1, 5},
				{2, 6},
				{3, 7}
		};

//a2fEdgeDirection lists the direction vector (vertex1-vertex0) for each edge in the cube
static const int a2fEdgeDirection[12][3] =
		{
				{1,  0,  0},
				{0,  1,  0},
				{-1, 0,  0},
				{0,  -1, 0},
				{1,  0,  0},
				{0,  1,  0},
				{-1, 0,  0},
				{0,  -1, 0},
				{0,  0,  1},
				{0,  0,  1},
				{0,  0,  1},
				{0,  0,  1}
		};

//a2iTetrahedronEdgeConnection lists the index of the endpoint vertices for each of the 6 edges of the tetrahedron
static const int a2iTetrahedronEdgeConnection[6][2] =
		{
				{0, 1},
				{1, 2},
				{2, 0},
				{0, 3},
				{1, 3},
				{2, 3}
		};

//a2iTetrahedronEdgeConnection lists the index of verticies from a cube
// that made up each of the six tetrahedrons within the cube
static const int a2iTetrahedronsInACube[6][4] =
		{
				{0, 5, 1, 6},
				{0, 1, 2, 6},
				{0, 2, 3, 6},
				{0, 3, 7, 6},
				{0, 7, 4, 6},
				{0, 4, 5, 6},
		};

static const int aiCubeEdgeFlags[256] =
		{
				0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
				0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
				0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
				0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
				0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
				0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
				0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
				0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
				0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
				0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
				0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
				0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
				0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
				0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
				0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
				0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
		};

static const int aiTetrahedronEdgeFlags[16] =
		{
				0x00, 0x0d, 0x13, 0x1e, 0x26, 0x2b, 0x35, 0x38, 0x38, 0x35, 0x2b, 0x26, 0x1e, 0x13, 0x0d, 0x00,
		};

static const int a2iTetrahedronTriangles[16][7] =
		{
				{-1, -1, -1, -1, -1, -1, -1},
				{0,  3,  2,  -1, -1, -1, -1},
				{0,  1,  4,  -1, -1, -1, -1},
				{1,  4,  2,  2,  4,  3,  -1},

				{1,  2,  5,  -1, -1, -1, -1},
				{0,  3,  5,  0,  5,  1,  -1},
				{0,  2,  5,  0,  5,  4,  -1},
				{5,  4,  3,  -1, -1, -1, -1},

				{3,  4,  5,  -1, -1, -1, -1},
				{4,  5,  0,  5,  2,  0,  -1},
				{1,  5,  0,  5,  3,  0,  -1},
				{5,  2,  1,  -1, -1, -1, -1},

				{3,  4,  2,  2,  4,  1,  -1},
				{4,  1,  0,  -1, -1, -1, -1},
				{2,  3,  0,  -1, -1, -1, -1},
				{-1, -1, -1, -1, -1, -1, -1},
		};


static const int a2iTriangleConnectionTable[256][16] =
		{
				{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  1,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  8,  3,  9,  8,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{2,  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
				{3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  11, 2,  8,  11, 0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  9,  0,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1, -1, -1, -1},
				{3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, -1},
				{3,  9,  0,  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, -1},
				{9,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  4,  7,  3,  0,  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1},
				{9,  2,  10, 9,  0,  2,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
				{2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
				{8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
				{9,  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
				{4,  7,  11, 9,  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, -1},
				{3,  10, 1,  3,  11, 10, 7,  8,  4,  -1, -1, -1, -1, -1, -1, -1},
				{1,  11, 10, 1,  4,  11, 1,  0,  4,  7,  11, 4,  -1, -1, -1, -1},
				{4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
				{4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, -1},
				{9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  5,  4,  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  5,  4,  1,  5,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{8,  5,  4,  8,  3,  5,  3,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
				{5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
				{2,  10, 5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, -1},
				{9,  5,  4,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  11, 2,  0,  8,  11, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
				{0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
				{2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1, -1, -1},
				{10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1},
				{4,  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, -1},
				{5,  4,  0,  5,  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
				{5,  4,  8,  5,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1},
				{9,  7,  8,  5,  7,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
				{0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
				{1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  7,  8,  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, -1},
				{10, 1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3,  -1, -1, -1, -1},
				{8,  0,  2,  8,  2,  5,  8,  5,  7,  10, 5,  2,  -1, -1, -1, -1},
				{2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
				{7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1},
				{9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, -1},
				{2,  3,  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, -1},
				{11, 2,  1,  11, 1,  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
				{9,  5,  8,  8,  5,  7,  10, 1,  3,  10, 3,  11, -1, -1, -1, -1},
				{5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,  10, 11, 10, 0,  -1},
				{11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,  0,  -1},
				{11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  0,  1,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  8,  3,  1,  9,  8,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
				{1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, -1},
				{9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
				{5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, -1},
				{2,  3,  11, 10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{11, 0,  8,  11, 2,  0,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
				{0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
				{5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1, -1},
				{6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
				{3,  11, 6,  0,  3,  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, -1},
				{6,  5,  9,  6,  9,  11, 11, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
				{5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1, -1, -1},
				{1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
				{10, 6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
				{6,  1,  2,  6,  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7,  -1, -1, -1, -1},
				{8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6,  -1, -1, -1, -1},
				{7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,  -1},
				{3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
				{5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, -1},
				{0,  1,  9,  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1},
				{9,  2,  1,  9,  11, 2,  9,  4,  11, 7,  11, 4,  5,  10, 6,  -1},
				{8,  4,  7,  3,  11, 5,  3,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
				{5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,  0,  4,  11, -1},
				{0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,  -1},
				{6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, -1},
				{10, 4,  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  10, 6,  4,  9,  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, -1},
				{10, 0,  1,  10, 6,  0,  6,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
				{8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,  10, -1, -1, -1, -1},
				{1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
				{3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, -1},
				{0,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{8,  3,  2,  8,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
				{10, 4,  9,  10, 6,  4,  11, 2,  3,  -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  -1, -1, -1, -1},
				{3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1, -1, -1, -1},
				{6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  -1},
				{9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, -1},
				{8,  11, 1,  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  -1},
				{3,  11, 6,  3,  6,  0,  0,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
				{6,  4,  8,  11, 6,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1, -1, -1, -1, -1},
				{0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1, -1},
				{10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, -1},
				{10, 6,  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, -1},
				{2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9,  -1},
				{7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
				{7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, -1},
				{2,  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  -1},
				{1,  8,  0,  1,  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, -1},
				{11, 2,  1,  11, 1,  7,  10, 6,  1,  6,  7,  1,  -1, -1, -1, -1},
				{8,  9,  6,  8,  6,  7,  9,  1,  6,  11, 6,  3,  1,  3,  6,  -1},
				{0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, -1},
				{7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  0,  8,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  1,  9,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
				{10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
				{2,  9,  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
				{6,  11, 7,  2,  10, 3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, -1},
				{7,  2,  3,  6,  2,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
				{2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1, -1, -1},
				{1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, -1},
				{10, 7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
				{10, 7,  6,  1,  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, -1},
				{0,  3,  7,  0,  7,  10, 0,  10, 9,  6,  10, 7,  -1, -1, -1, -1},
				{7,  6,  10, 7,  10, 8,  8,  10, 9,  -1, -1, -1, -1, -1, -1, -1},
				{6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
				{8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, -1},
				{9,  4,  6,  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, -1},
				{6,  8,  4,  6,  11, 8,  2,  10, 1,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, 3,  0,  11, 0,  6,  11, 0,  4,  6,  -1, -1, -1, -1},
				{4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,  -1, -1, -1, -1},
				{10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,  -1},
				{8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
				{0,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, -1},
				{1,  9,  4,  1,  4,  2,  2,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
				{8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10, 1,  -1, -1, -1, -1},
				{10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
				{4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  -1},
				{10, 9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  9,  5,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  4,  9,  5,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
				{5,  0,  1,  5,  4,  0,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
				{11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1, -1, -1, -1},
				{9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
				{6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, -1},
				{7,  6,  11, 5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, -1},
				{3,  4,  8,  3,  5,  4,  3,  2,  5,  10, 5,  2,  11, 7,  6,  -1},
				{7,  2,  3,  7,  6,  2,  5,  4,  9,  -1, -1, -1, -1, -1, -1, -1},
				{9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,  -1, -1, -1, -1},
				{3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1, -1},
				{6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  -1},
				{9,  5,  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, -1},
				{1,  6,  10, 1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  -1},
				{4,  0,  10, 4,  10, 5,  0,  3,  10, 6,  10, 7,  3,  7,  10, -1},
				{7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,  10, -1, -1, -1, -1},
				{6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
				{3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, -1},
				{0,  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, -1},
				{6,  11, 3,  6,  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  10, 9,  5,  11, 9,  11, 8,  11, 5,  6,  -1, -1, -1, -1},
				{0,  11, 3,  0,  6,  11, 0,  9,  6,  5,  6,  9,  1,  2,  10, -1},
				{11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,  2,  5,  -1},
				{6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, -1},
				{5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, -1},
				{9,  5,  6,  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
				{1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8,  -1},
				{1,  5,  6,  2,  1,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,  8,  9,  6,  -1},
				{10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1, -1},
				{0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{10, 5,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{11, 5,  10, 7,  5,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{11, 5,  10, 11, 7,  5,  8,  3,  0,  -1, -1, -1, -1, -1, -1, -1},
				{5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1, -1, -1, -1, -1, -1},
				{10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1, -1, -1},
				{11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, -1},
				{9,  7,  5,  9,  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, -1},
				{7,  5,  2,  7,  2,  11, 5,  9,  2,  3,  2,  8,  9,  8,  2,  -1},
				{2,  5,  10, 2,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
				{8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1, -1, -1, -1},
				{9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, -1},
				{9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  -1},
				{1,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  7,  0,  7,  1,  1,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
				{9,  0,  3,  9,  3,  5,  5,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
				{9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1},
				{5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, -1},
				{0,  1,  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, -1},
				{10, 11, 4,  10, 4,  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  -1},
				{2,  5,  1,  2,  8,  5,  2,  11, 8,  4,  5,  8,  -1, -1, -1, -1},
				{0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11, 1,  5,  1,  11, -1},
				{0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,  5,  -1},
				{9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{2,  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, -1},
				{5,  10, 2,  5,  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
				{3,  10, 2,  3,  5,  10, 3,  8,  5,  4,  5,  8,  0,  1,  9,  -1},
				{5,  10, 2,  5,  2,  4,  1,  9,  2,  9,  4,  2,  -1, -1, -1, -1},
				{8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
				{0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, -1},
				{9,  4,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  11, 7,  4,  9,  11, 9,  10, 11, -1, -1, -1, -1, -1, -1, -1},
				{0,  8,  3,  4,  9,  7,  9,  11, 7,  9,  10, 11, -1, -1, -1, -1},
				{1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11, -1, -1, -1, -1},
				{3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,  -1},
				{4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, -1},
				{9,  7,  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  -1},
				{11, 7,  4,  11, 4,  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
				{11, 7,  4,  11, 4,  2,  8,  3,  4,  3,  2,  4,  -1, -1, -1, -1},
				{2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,  9,  -1, -1, -1, -1},
				{9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,  7,  -1},
				{3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, -1},
				{1,  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  9,  1,  4,  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
				{4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1,  -1, -1, -1, -1},
				{4,  0,  3,  7,  4,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, -1},
				{0,  1,  10, 0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, -1},
				{3,  1,  10, 11, 3,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  2,  11, 1,  11, 9,  9,  11, 8,  -1, -1, -1, -1, -1, -1, -1},
				{3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,  -1, -1, -1, -1},
				{0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{2,  3,  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
				{9,  10, 2,  0,  9,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{2,  3,  8,  2,  8,  10, 0,  1,  8,  1,  10, 8,  -1, -1, -1, -1},
				{1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{0,  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
				{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
		};

float fGetOffset(const float &fValue1, const float &fValue2, const float &fValueDesired)
{
	float fDelta = fValue2 - fValue1;
	if (fDelta == 0.0)
	{ return 0.5; }
	return (fValueDesired - fValue1) / fDelta;
}

//marchingCube performs the Marching Cubes algorithm on a single cube
void marching_cube_at_point(const vec3i& p, const float& iso_val, float*** data, std::vector<MCSortVertex>& ret_points)
{
	int iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
	float fOffset;
	float afCubeValue[8];
	vec3f asEdgeVertex[12];
	vec3f fP = makeVec3f(p.x, p.y, p.z);

	//Make a local copy of the values at the cube's corners
	for (iVertex = 0;
			 iVertex < 8;
			 iVertex++)
	{
		afCubeValue[iVertex] = data[p.z+a2fVertexOffset[iVertex][2]][p.y+a2fVertexOffset[iVertex][1]][p.x+a2fVertexOffset[iVertex][0]];
	}

	//Find which vertices are inside of the surface and which are outside
	iFlagIndex = 0;
	for (iVertexTest = 0; iVertexTest < 8; iVertexTest++)
	{
		if (afCubeValue[iVertexTest] <= iso_val)
		{ iFlagIndex |= 1 << iVertexTest; }
	}

	//Find which edges are intersected by the surface
	iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

	//If the cube is entirely inside or outside of the surface, then there will be no intersections
	if (iEdgeFlags == 0)
	{
		return;
	}

	//Find the point of intersection of the surface with each edge
	//Then find the normal to the surface at those points
	for (iEdge = 0; iEdge < 12; iEdge++)
	{
		//if there is an intersection on this edge
		if (iEdgeFlags & (1 << iEdge))
		{
			fOffset = fGetOffset(afCubeValue[a2iEdgeConnection[iEdge][0]], afCubeValue[a2iEdgeConnection[iEdge][1]], iso_val);

			asEdgeVertex[iEdge].x = fP.x + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][0] +
																			fOffset * a2fEdgeDirection[iEdge][0]);
			asEdgeVertex[iEdge].y = fP.y + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][1] +
																			fOffset * a2fEdgeDirection[iEdge][1]);
			asEdgeVertex[iEdge].z = fP.z + (a2fVertexOffset[a2iEdgeConnection[iEdge][0]][2] +
																			fOffset * a2fEdgeDirection[iEdge][2]);
		}
	}


	//Draw the triangles that were found.  There can be up to five per cube

	MCSortVertex sort_vertex;
	for (iTriangle = 0; iTriangle < 5; iTriangle++)
	{
		if (a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
		{ break; }

		for (iCorner = 0; iCorner < 3; iCorner++)
		{
			iVertex = a2iTriangleConnectionTable[iFlagIndex][3 * iTriangle + iCorner];
			sort_vertex.v = asEdgeVertex[iVertex];
			sort_vertex.vid = ret_points.size();
			ret_points.push_back(sort_vertex);
		}
	}
}

inline bool operator< (const MCSortVertex& a, const MCSortVertex& b){
	return (a.v<b.v);
}

void marching_cube(const float& iso_val, const vec3i& dim, float*** data, std::vector<MCSortVertex>& ret_points){
	vec3i p;
	for (p.z=0; p.z<dim.z-1; ++p.z){
		for (p.y=0; p.y<dim.y-1; ++p.y){
			for (p.x=0; p.x<dim.x-1; ++p.x){
				marching_cube_at_point(p, iso_val, data, ret_points);
			}
		}
	}
}

void remove_duplicate(std::vector<vec4f>& ret, std::vector<int>& indices, std::vector<MCSortVertex>& points){
	ret.clear();
	indices.resize(points.size());
	std::sort(points.begin(), points.end());

	vec3f prev = makeVec3f(-1e30,-1e30,-1e30), curr;
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		curr = points[i].v;
		if (curr != prev)
		{
			prev = curr;
			ret.push_back(makeVec4f(curr.x, curr.y, curr.z, 1.0f));
		}
		indices[points[i].vid] = (int)(ret.size())-1;
	}
}

void remove_duplicate(std::vector<vec4f>& points){
	std::sort(points.begin(), points.end());
	std::vector<vec4f> tmp_points(points.begin(), points.end());
	points.clear();
	vec4f prev = makeVec4f(-1e30,-1e30,-1e30,-1e30), curr;
	for (uint i=0; i<tmp_points.size(); ++i) {
		curr = tmp_points[i];
		if (curr!=prev) {
			prev = curr;
			points.push_back(curr);
		}
	}
}

extern "C" void
launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, float *volume,
										 uint3 gridSize, float isoValue);

extern "C" void
launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied,
										 uint *voxelOccupiedScan, uint numVoxels);

extern "C" void
launch_generateTriangles(dim3 grid, dim3 threads,
												 float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, float *volume,
												 uint3 gridSize, float isoValue, uint activeVoxels, uint maxVerts);

extern "C" void allocateTextures(uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable);
extern "C" void bindVolumeTexture(float *d_volume);
extern "C" void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements);

uint *d_edgeTable = 0;
uint *d_triTable = 0;
uint *d_numVertsTable = 0;
int allocateMemSize = 0;

float *d_volume = 0;
uint *d_voxelVerts = 0;
uint *d_voxelVertsScan = 0;
uint *d_voxelOccupied = 0;
uint *d_voxelOccupiedScan = 0;
uint *d_compVoxelArray;

void cudaMarchingCubeInit(const vec3i& dim){
	// allocate textures
	if (d_edgeTable==0) {
		allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);
	}

	// allocate device memory
	unsigned int memSize = dim.x*dim.y*dim.z;
	if (allocateMemSize<memSize) {
		if (allocateMemSize!=0) {
			checkCudaErrors(cudaFree(d_voxelVerts));
			checkCudaErrors(cudaFree(d_voxelVertsScan));
			checkCudaErrors(cudaFree(d_voxelOccupied));
			checkCudaErrors(cudaFree(d_voxelOccupiedScan));
			checkCudaErrors(cudaFree(d_compVoxelArray));
			checkCudaErrors(cudaFree(d_volume));
		}
		checkCudaErrors(cudaMalloc((void **) &d_voxelVerts,         sizeof(uint)*memSize));
		checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScan,     sizeof(uint)*memSize));
		checkCudaErrors(cudaMalloc((void **) &d_voxelOccupied,      sizeof(uint)*memSize));
		checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScan,  sizeof(uint)*memSize));
		checkCudaErrors(cudaMalloc((void **) &d_compVoxelArray,		sizeof(uint)*memSize));
		checkCudaErrors(cudaMalloc((void **) &d_volume,				sizeof(float)*memSize));
		allocateMemSize = memSize;
	}
}

void cudaMarchingCubeFree(){
	if (d_edgeTable!=0) {
		checkCudaErrors(cudaFree(d_edgeTable));
		checkCudaErrors(cudaFree(d_triTable));
		checkCudaErrors(cudaFree(d_numVertsTable));
	}

	if (allocateMemSize!=0) {
		checkCudaErrors(cudaFree(d_voxelVerts));
		checkCudaErrors(cudaFree(d_voxelVertsScan));
		checkCudaErrors(cudaFree(d_voxelOccupied));
		checkCudaErrors(cudaFree(d_voxelOccupiedScan));
		checkCudaErrors(cudaFree(d_compVoxelArray));
		checkCudaErrors(cudaFree(d_volume));
	}
}

void cudaMarchingCube(const float& iso_val, const vec3i& dim, float* data,
											const int& maxVerts,
											std::vector<vec4f>& ret_points)
{
	int threads = 128;
	int numVoxels = dim.x*dim.y*dim.z;

	cudaMarchingCubeInit(dim);
	checkCudaErrors(cudaMemcpy(d_volume, data, sizeof(float)*numVoxels, cudaMemcpyHostToDevice));
	bindVolumeTexture(d_volume);

	dim3 grid(numVoxels / threads, 1, 1);
	uint3 gridSize = make_uint3(dim.x, dim.y, dim.z);
	uint activeVoxels=0, totalVerts=0;

	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535)
	{
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(grid, threads,
											 d_voxelVerts, d_voxelOccupied, d_volume,
											 gridSize, iso_val);

	uint* voxelOccupied = new uint[numVoxels];
	checkCudaErrors(cudaMemcpy(voxelOccupied, d_voxelOccupied, sizeof(uint)*numVoxels, cudaMemcpyDeviceToHost));

#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelOccupiedScan:\n");
	dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *) &lastElement,
															 (void *)(d_voxelOccupied + numVoxels-1),
															 sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
															 (void *)(d_voxelOccupiedScan + numVoxels-1),
															 sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels==0)
	{
		// return if there are no full voxels
		totalVerts = 0;
		return;
	}

	// compact voxel index array
	launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
	getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS

	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *) &lastElement,
															 (void *)(d_voxelVerts + numVoxels-1),
															 sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
															 (void *)(d_voxelVertsScan + numVoxels-1),
															 sizeof(uint), cudaMemcpyDeviceToHost));
		totalVerts = lastElement + lastScanElement;
	}

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);
#else
	dim3 grid2((int) ceil(numVoxels / (float) NTHREADS), 1, 1);
#endif

	while (grid2.x > 65535)
	{
		grid2.x/=2;
		grid2.y*=2;
	}

	int num_vertices = (maxVerts<totalVerts)?maxVerts:totalVerts;

	float4 *d_pos, *d_normal;
	checkCudaErrors(cudaMalloc((void **) &d_pos, sizeof(float4)*maxVerts));
	checkCudaErrors(cudaMalloc((void **) &d_normal, sizeof(float4)*maxVerts));
	launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal,
													 d_compVoxelArray,
													 d_voxelVertsScan, d_volume,
													 gridSize, iso_val, activeVoxels,
													 maxVerts);
	checkCudaErrors(cudaFree(d_normal));


	float4 *d_output;
	checkCudaErrors(cudaMalloc((void **) &d_output, sizeof(float4)*num_vertices));
  Report::begin("remove duplicates");
  size_t length = removeDuplicates(reinterpret_cast<vec4f *>(d_pos), num_vertices, reinterpret_cast<vec4f *>(d_output));
  Report::end("remove duplicates");
	ret_points.resize(length);

	checkCudaErrors(cudaMemcpy(&ret_points[0], d_output, sizeof(float4)*length, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_pos));
	checkCudaErrors(cudaFree(d_output));
}