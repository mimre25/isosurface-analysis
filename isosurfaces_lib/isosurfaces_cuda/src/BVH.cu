#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <stdio.h>
#include <vector_functions.h>
#include "isosurfaces_cuda/hdr/common.h"
#include "isosurfaces_cuda/hdr/thrustWrapper.h"
#include "isosurfaces_cuda/hdr/BVH.h"
#include "runtime/hdr/globals.h"
#include <string>
#include "isosurfaces_cuda/src/helper_cuda.h"

using namespace std;
inline void printPoint(float3 &p)
{
  printf("%f, %f, %f\n", p.x, p.y, p.z);
}

////////////// BVH BUILD

////////////// BVH BUILDER HELPER

inline __device__ int calcId_d(int idx, bool internal)
{
  return internal ? idx : -idx - 1;
}

inline __device__ float3 min_d(float3 a, float3 b)
{
  return make_float3(a.x < b.x ? a.x : b.x,
                     a.y < b.y ? a.y : b.y,
                     a.z < b.z ? a.z : b.z);
}

inline __device__ float3 max_d(float3 a, float3 b)
{
  return make_float3(a.x > b.x ? a.x : b.x,
                     a.y > b.y ? a.y : b.y,
                     a.z > b.z ? a.z : b.z);
}

inline __device__ bool operator<(const float3 &a, const float3 &b)
{
  return a.x < b.x && a.y < b.y && a.z < b.z;
}

inline __device__ bool operator>(const float3 &a, const float3 &b)
{
  return a.x > b.x && a.y > b.y && a.z > b.z;
}


inline __device__ BoundingBox createBoundingBox_d(float3 min, float3 max)
{
  BoundingBox b = BoundingBox();

  b.minCorner = min;
  b.maxCorner = max;
  return b;
}

inline __device__ BoundingBox createBoundingBox_d(float3 p)
{
  return createBoundingBox_d(p, p);
}

inline __device__ BoundingBox createBoundingBox_d(BoundingBox b1, BoundingBox b2)
{
  float3 minCorner = min_d(b1.minCorner, b2.minCorner);
  float3 maxCorner = max_d(b1.maxCorner, b2.maxCorner);
  return createBoundingBox_d(minCorner, maxCorner);
}

inline __device__ int sign_d(int num)
{
  return num == 0 ? 0 : num < 0 ? -1 : +1;
}

////////////// END BVH BUILD HELPER


// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.

__device__ unsigned int expandBits_d(unsigned int v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D_d(float x, float y, float z)
{
  x = min(max(x * 1024.0f, 0.0f), 1023.0f);
  y = min(max(y * 1024.0f, 0.0f), 1023.0f);
  z = min(max(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expandBits_d((unsigned int) x);
  unsigned int yy = expandBits_d((unsigned int) y);
  unsigned int zz = expandBits_d((unsigned int) z);
  return xx * 4 + yy * 2 + zz;
}

__global__ void calculateMortonCodes_g(float3 *points, const int numPoints, unsigned int *mortonCodes)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numPoints)
  {
    float3 p = points[idx];
    unsigned int mc = morton3D_d(p.x, p.y, p.z);
    mortonCodes[idx] = mc;
  }
}



//https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
__device__ int findSplit_d(unsigned int *sortedMortonCodes,
                           int first,
                           int last)
{
  // Identical Morton codes => split the range in the middle.

  unsigned int firstCode = sortedMortonCodes[first];
  unsigned int lastCode = sortedMortonCodes[last];

  if (firstCode == lastCode)
  {
    return first;
//    return (first + last) >> 1;
  }

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.

  int commonPrefix = __clz(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.

  int split = first; // initial guess
  int step = last - first;

  do
  {
    step = (step + 1) >> 1; // exponential decrease
    int newSplit = split + step; // proposed new position

    if (newSplit < last)
    {
      unsigned int splitCode = sortedMortonCodes[newSplit];
      int splitPrefix = __clz(firstCode ^ splitCode);
      if (splitPrefix > commonPrefix)
      {
        split = newSplit;
      } // accept proposal
    }
  } while (step > 1);

  return split;
}



__device__ int delta_d(int idx, int neighbor, int numObjects, unsigned int *sortedMortonCodes)
{
  if (neighbor >= numObjects || neighbor < 0)
  {
    return -1;
  } else
  {
    int leadingZeros = 0;
    int exOr;

    if (sortedMortonCodes[idx] == sortedMortonCodes[neighbor])
    {
      exOr = (idx ^ neighbor);
      leadingZeros = 32;//not sure about this
    } else
    {
      exOr = sortedMortonCodes[idx] ^ sortedMortonCodes[neighbor];
    }

    leadingZeros += __clz(exOr);
    return leadingZeros;
  }

}

__device__ int2 determineRange_d(unsigned int *sortedMortonCodes, const int numObjects, const int idx)
{
  if (idx == 0)
  {
    return make_int2(0, numObjects - 1);
  } else
  {
    if ((idx != (numObjects-1) && sortedMortonCodes[idx] == sortedMortonCodes[idx+1] && sortedMortonCodes[idx] == sortedMortonCodes[idx -1]))
    {
      int index = idx;
      while (idx > 0 && idx < numObjects-1)
      {
        index += 1;
        if (index >= numObjects - 1)
          //we hit the left end of our list
          break;

        if (sortedMortonCodes[index] != sortedMortonCodes[index + 1])
          //there is a diffrence
          break;
      }
      return make_int2(idx,index);
    }
    else
    {
      int direction = sign_d((delta_d(idx, idx + 1, numObjects, sortedMortonCodes) -
                              delta_d(idx, idx - 1, numObjects, sortedMortonCodes)));
      int deltaMin = delta_d(idx, idx - direction, numObjects, sortedMortonCodes);
//      printf("id: %d, dir %d, dmin %d\n", idx, direction, deltaMin);
      int lMax = 2;
      while (delta_d(idx, idx + lMax * direction, numObjects, sortedMortonCodes) > deltaMin)
      {
        lMax *= 2;
      }
      int l = 0;
      int t = lMax / 2;
      while (t >= 1)
      {
        if (delta_d(idx, idx + (l + t) * direction, numObjects, sortedMortonCodes) > deltaMin)
        {
          l += t;
        }
        t /= 2;
      }
      int j = idx + l * direction;

      return direction < 0 ? make_int2(j, idx) : make_int2(idx, j);
    }
  }
}


__global__ void
initLeaves_g(LeafNode *leafNodes, float3 *sortedPoints, unsigned int *mortonCodes, const int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {
    leafNodes[idx].point = sortedPoints[idx];
    leafNodes[idx].mortonCode = mortonCodes[idx];
    leafNodes[idx].id = idx;
    leafNodes[idx].parentId = -1;
  }
}

__global__ void initNodes_g(InternalNode *internalNodes, const int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {
    internalNodes[idx] = InternalNode();
    internalNodes[idx].internal = true;
    internalNodes[idx].id = idx;
    internalNodes[idx].parentId = -1;
  }
}


__global__ void
generateInternalNodes_g(LeafNode *leafNodes, InternalNode *internalNodes, unsigned int *sortedMortonCodes,
                        const int numObjects, const int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < size)
  {
    int2 range = determineRange_d(sortedMortonCodes, numObjects, idx);
    int first = range.x;
    int last = range.y;
    // Determine where to split the range.

    int split = findSplit_d(sortedMortonCodes, first, last);
//    printf("id: %d, range: %d, %d, split: %d\n", idx, first, last, split);

    // Select childA.

    Node *childA;
    bool internalA = split != first;
    if (internalA)
    {
      childA = &internalNodes[split];
    } else
    {
      childA = &leafNodes[split];
    }
    childA->internal = internalA;

    // Select childB.

    Node *childB;
    bool internalB = (split + 1) != last;
    if (internalB)
    {
      childB = &internalNodes[split + 1];

    } else
    {
      childB = &leafNodes[split + 1];
    }

    childB->internal = internalB;

    // Record parent-child relationships.
    internalNodes[idx].childA = childA;
    internalNodes[idx].childB = childB;
    internalNodes[idx].leftChildId = calcId_d(split, internalA);
    internalNodes[idx].rightChildId = calcId_d(split + 1, internalB);
    internalNodes[idx].first = first;
    internalNodes[idx].last = last;
    internalNodes[idx].split = split;
//    printf("idx: %d,  lc %d, rc %d, f %d, l %d, s %d \n",
//           idx,
//           internalNodes[idx].leftChildId,
//           internalNodes[idx].rightChildId,
//           internalNodes[idx].first,
//           internalNodes[idx].last,
//           internalNodes[idx].split
//    );
    childA->parent = &internalNodes[idx];
    childA->parentId = idx;
    childB->parent = &internalNodes[idx];
    childB->parentId = idx;
  }
}


__device__ void createVolumes_d(LeafNode *leaves, InternalNode *nodes, int *blockades, int idx)
{
  LeafNode *leaf = &leaves[idx];
  leaf->boundingBox = createBoundingBox_d(leaf->point);


  int parent = leaf->parentId;
  bool run = atomicExch(&blockades[parent], 1) == 1;
  while (run)
  {
    InternalNode *n = &nodes[parent];
    BoundingBox left =
        n->leftChildId < 0 ? leaves[calcId_d(n->leftChildId, false)].boundingBox : nodes[n->leftChildId].boundingBox;
    BoundingBox right = n->rightChildId < 0 ? leaves[calcId_d(n->rightChildId, false)].boundingBox
                                            : nodes[n->rightChildId].boundingBox;
    n->boundingBox = createBoundingBox_d(left, right);
    if (parent == 0)
    {
      run = false;
    } else
    {
      parent = n->parentId;
      run = atomicExch(&blockades[parent], 1) == 1;
    }

  }
}

__global__ void createBoundingVolumes_g(LeafNode *leaves, InternalNode *nodes, int *blockades, const int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {
    if (size != 1)
    {
      createVolumes_d(leaves, nodes, blockades, idx);
    } else
    {
      LeafNode *leaf = &leaves[idx];
      leaf->boundingBox = createBoundingBox_d(leaf->point);
    }
  }
}


////////////// END BVH BUILD


////////////// TRAVERSAL

////////////// TRAVERSAL HELPER

__device__ float3 findSamplePointForIndex_d(const int idx, const int dimX, const int dimY, const int dimZ, const int dfDownscale)
{
  int dimY10 = dimY / dfDownscale;
  int dimZ10 = dimZ / dfDownscale;
  int z = idx % dimZ10;
  int y = (idx / dimZ10) % dimY10;
  int x = idx / (dimY10 * dimZ10);
  float px = float(x * dfDownscale + dfDownscale/2) / dimX;
  float py = float(y * dfDownscale + dfDownscale/2) / dimY;
  float pz = float(z * dfDownscale + dfDownscale/2) / dimZ;

  return make_float3(px, py, pz);

}

__device__ float float3SquaredDistance2_d(float3 v1, float3 v2)
{
  float x2 = (v2.x - v1.x) * (v2.x - v1.x);
  float y2 = (v2.y - v1.y) * (v2.y - v1.y);
  float z2 = (v2.z - v1.z) * (v2.z - v1.z);
  return (x2 + y2 + z2);
}

__device__ float getMinDistanceOfDimensions_d(float3 v1, float3 v2)
{
  return min(min(v1.x, v2.x), min(min(v1.y, v2.y), min(v1.z, v2.z)));
}

__device__ float distanceToBB_d(float3 v, BoundingBox bb)
{
  return min(getMinDistanceOfDimensions_d(v, bb.minCorner), getMinDistanceOfDimensions_d(v, bb.maxCorner));
}

__device__ bool pointInsideBB_d(float3 v, BoundingBox bb)
{
  return v < bb.maxCorner && v > bb.minCorner;
}


__device__ float distanceInDimension_d(float p, float minP, float maxP, const short dimension, int *mask)
{
  float dis = min(0.0f, min((float) p - minP, (float) maxP - p)) * (-1.0f);
  (*mask) |= ((dis > 0) << dimension);
  return dis;
}


__device__ float distanceToBox_d(float3 p, BoundingBox box)
{
  const int INSIDE = 0x0;
  const int X_ONLY = 1 << 0;
  const int Y_ONLY = 1 << 1;
  const int Z_ONLY = 1 << 2;
  const int XY = X_ONLY | Y_ONLY;
  const int XZ = X_ONLY | Z_ONLY;
  const int YZ = Y_ONLY | Z_ONLY;
  const int XYZ = X_ONLY | Y_ONLY | Z_ONLY;

  int mask = 0;
  float3 minCorner = box.minCorner;
  float3 maxCorner = box.maxCorner;
  float x = distanceInDimension_d(p.x, minCorner.x, maxCorner.x, 0, &mask);
  float y = distanceInDimension_d(p.y, minCorner.y, maxCorner.y, 1, &mask);
  float z = distanceInDimension_d(p.z, minCorner.z, maxCorner.z, 2, &mask);
  float distance = 0.0f;
  switch (mask)
  {
    case X_ONLY:
    {
      distance = x * x;
      break;
    }
    case Y_ONLY:
    {
      distance = y * y;
      break;
    }
    case Z_ONLY:
    {
      distance = z * z;
      break;
    }
      //only two dimension ==> distance = sqrt(d_d1^2 + d_d2^2)
    case XY:
    {
      distance = x * x + y * y;
      break;
    }
    case XZ:
    {
      distance = x * x + z * z;
      break;
    }
    case YZ:
    {
      distance = y * y + z * z;
      break;
    }
      //three d ==> full vector distance:
    case XYZ:
    {
      distance = x * x + y * y + z * z;
      break;
    }
      //inside => distance == 0.0f
    case INSIDE:
    default:
    {
      distance = 0.0f;
    }
  }
  return distance;

}
////////////// TRAVERSAL HELPER END


__device__ float
traverseIterative_d(InternalNode *nodes, LeafNode *leaves, float3 samplePoint, float upperBound/*, float3* closestPoint*/)
{
  // Allocate traversal stack from thread-local memory,
  // and push NULL to indicate that there are no postponed nodes.
  int stack[64];
  int *stackPtr = stack;
  *stackPtr++ = 0; // push

  // Traverse nodes starting from the root.
  int nodeId = 0;

  do
  {

    if (nodeId < 0)
    {
      //leaf
      LeafNode leaf = leaves[calcId_d(nodeId, false)];
      float distance = float3SquaredDistance2_d(leaf.point, samplePoint);
      upperBound = upperBound > distance ? distance : upperBound;
      nodeId = *--stackPtr; // pop
    } else
    {
      //node
      InternalNode node = nodes[nodeId];


      int leftChild = node.leftChildId;
      int rightChild = node.rightChildId;

      BoundingBox leftBB;
      BoundingBox rightBB;
      if (leftChild > 0)
      {
        leftBB = nodes[leftChild].boundingBox;

      } else
      {
        leftBB = leaves[calcId_d(leftChild, false)].boundingBox;
      }
      if (rightChild > 0)
      {
        rightBB = nodes[rightChild].boundingBox;
      } else
      {
        rightBB = leaves[calcId_d(rightChild, false)].boundingBox;
      }
      bool traverseL = distanceToBox_d(samplePoint, leftBB) < upperBound;
      bool traverseR = distanceToBox_d(samplePoint, rightBB) < upperBound;
      //traversal
      if (!traverseL && !traverseR)
      {
        nodeId = *--stackPtr; // pop
      } else
      {
        nodeId = (traverseL) ? leftChild : rightChild;
        if (traverseL && traverseR)
        {
          *stackPtr++ = rightChild;
        } // push
      }
    }
  } while (nodeId != 0);
  return upperBound;
}

__global__ void traverseTree_g(LeafNode *leaves, InternalNode *nodes, float *distances, const int size,
                               const int dimX, const int dimY, const int dimZ,
                               const int numSamples, const int numPoints, unsigned int* sampleIds, int dfDownscale, float3* samplePoints, const bool sampleGiven)

{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size)
  {

    float3 samplePoint;
    if(sampleGiven)
    {
      samplePoint = samplePoints[idx];
    } else
    {
      samplePoint = findSamplePointForIndex_d(idx, dimX, dimY, dimZ, dfDownscale);
    }

    float upperBound = 100000;
    for (int i = 0; i < numSamples; ++i)
    {
      float dis = float3SquaredDistance2_d(leaves[sampleIds[i]].point, samplePoint);
      if (dis < upperBound)
      {
        upperBound = dis;
      }
    }
    if (numPoints != 1)
    {
      distances[idx] = sqrt(traverseIterative_d(nodes, leaves, samplePoint, upperBound/*, &closestPoints[idx]*/));
    } else
    {

      distances[idx] = upperBound;//sqrt(upperBound);
    }
  }
}


////////////// TRAVERSAL END

__host__ void
traverse_h(LeafNode *d_leafNodes, InternalNode *d_internalNodes, const int dimX, const int dimY, const int dimZ,
           const int numPoints, float *h_distances, int numSamples, int dfDownscale, const bool sampleGiven, const unsigned long  sampleSize, vector<float3> samplePoints)
{
  cudaError_t e;
  Reporter r;
  unsigned long SIZE;
  if(sampleGiven)
  {
    SIZE = sampleSize;
  } else
  {
    SIZE = (unsigned long) ((dimX / dfDownscale) * (dimY / dfDownscale) * (dimZ / dfDownscale));
  }
  const unsigned long DISTANCES_SIZE = SIZE * sizeof(float);
  numSamples = numSamples > numPoints ? numPoints : numSamples;
  float *d_distances;
  printf("number of sample points: %lu, memsize: %lu\n", SIZE, DISTANCES_SIZE);

  checkCudaErrors(cudaMalloc((void **) &d_distances, DISTANCES_SIZE));
  printf("ERROR in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));
  r.reportStart((char *) "traversal");
  unsigned int h_sampleIds[numSamples];
  for (int i = 0; i < numSamples; ++i)
  {
    h_sampleIds[i] =  ((unsigned int)(i) * (unsigned int)(numPoints) / (unsigned int)(numSamples));
  }

  unsigned int* d_sampleIds;
  cudaMalloc((void **) &d_sampleIds, numSamples * sizeof(unsigned int));
  cudaMemcpy(d_sampleIds, h_sampleIds, numSamples * sizeof(unsigned int), cudaMemcpyHostToDevice);
  printf("ERROR in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));
  float3* d_samplePoints;
  if(sampleGiven)
  {
    cudaMalloc((void **) &d_samplePoints, sampleSize * sizeof(float3));
    cudaMemcpy(d_samplePoints, &samplePoints[0], sampleSize * sizeof(float3), cudaMemcpyHostToDevice);
     
    printf("size %lu vec length %lu\n", SIZE, samplePoints.size());

  }
  printf("ERROR in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));
  
  size_t numBlocks = calculateBlockNumber(SIZE, 256);
  printf("Number of blocks: %lu\n", numBlocks);
  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
  // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, traverseTree_g, 0, 0);
  // Round up according to array size
  gridSize = (SIZE + blockSize - 1) / blockSize;
  printf("Gridsize: %d, blocksize: %d \n", gridSize, blockSize);
  getLastCudaError("before traversal");
  printf("size: %d, numSamples %d\n", SIZE, numSamples);
  traverseTree_g << < gridSize , blockSize >> > (d_leafNodes, d_internalNodes, d_distances, SIZE,
      dimX, dimY, dimZ,
      numSamples, numPoints, d_sampleIds, dfDownscale, d_samplePoints, sampleGiven);
  getLastCudaError("after traversal");  
  printf("\n");
  printf("ERROR in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));
  fflush(stdout);
  printf("\n");

  r.reportEnd();
  cudaMemcpy(h_distances, d_distances, DISTANCES_SIZE, cudaMemcpyDeviceToHost);
  printf("Distance size: %d, %d\n", DISTANCES_SIZE, __LINE__);

  /*for (int j = 0; j < SIZE; ++j)
  {
    if (h_distances < 0)
    {
      printf ("\n\nBELOW ZERO %f, %d\n\n", h_distances[j], j);
    }
  }*/

  e = cudaGetLastError();
  if (e != cudaSuccess || numPoints == 1) {
    printf("CUDA ERROR: %s %d\n", __FILE__, __LINE__);
  }
  printf("ERROR in line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));

  if(sampleGiven)
  {
    cudaFree(d_samplePoints);
  }

  cudaFree(d_distances);
}


__host__ void setup(unsigned int*& d_sortedMortonCodes, float3*& d_points,
                    LeafNode*& d_leafNodes, InternalNode*& d_internalNodes, vector<float3> points)

{
  printf("noPoints: %d\n", points.empty());
  const int numPoints = points.size();
  getLastCudaError("Before Setup");
  printf("allocating memory: %d bytes\n", numPoints * sizeof(unsigned int));
  fflush(stdout);
  checkCudaErrors(cudaMalloc((void **) &d_sortedMortonCodes, numPoints * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **) &d_points, numPoints * sizeof(float3)));
  checkCudaErrors(cudaMalloc((void **) &d_leafNodes, numPoints * sizeof(LeafNode)));
  checkCudaErrors(cudaMalloc((void **) &d_internalNodes, (numPoints - 1) * sizeof(InternalNode)));
  checkCudaErrors(cudaMemcpy(&d_points[0], &points[0], numPoints * sizeof(float3), cudaMemcpyHostToDevice));
  getLastCudaError("After Setup");

  return;
}


__host__ void cleanup(int* d_blockades, float3* d_points, unsigned int* d_sortedMortonCodes,
                      InternalNode* d_internalNodes, LeafNode* d_leafNodes)
{
  //put into cleanup function
  cudaFree(d_blockades);
  cudaFree(d_points);
  cudaFree(d_sortedMortonCodes);
  cudaFree(d_internalNodes);
  cudaFree(d_leafNodes);
}



__host__ void init(unsigned int* d_sortedMortonCodes, float3* d_points,
                   LeafNode* d_leafNodes, InternalNode* d_internalNodes, const int numPoints, const int NUM_BLOCKS,
                   const int NUM_THREADS, vector<float3> pts)
{
  getLastCudaError("before morton");
  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
  // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateMortonCodes_g, 0, 0);
  // Round up according to array size
  int size = numPoints;

  gridSize = (size + blockSize - 1) / blockSize;


  calculateMortonCodes_g << < gridSize, blockSize >> > (d_points, numPoints, d_sortedMortonCodes);
  getLastCudaError("after morton codes");
  sortByKeys(d_sortedMortonCodes, d_points, numPoints); // thrust wrapper
  getLastCudaError("after sort");

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateMortonCodes_g, 0, 0);
  // Round up according to array size
  gridSize = (size + blockSize - 1) / blockSize;
  initLeaves_g << < gridSize, blockSize >> >(d_leafNodes, d_points, d_sortedMortonCodes, numPoints);
  getLastCudaError("after leaves");

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateMortonCodes_g, 0, 0);
  // Round up according to array size
  gridSize = (size + blockSize - 1) / blockSize;
  initNodes_g << < gridSize, blockSize >> > (d_internalNodes, numPoints - 1);
  getLastCudaError("after nodes");
}


__global__ void convertKernel_g(float3* d_points, const int maxDim, const int numPoints)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < numPoints)
  {
    float3 p = d_points[idx];
    d_points[idx] = make_float3(p.x,p.y,p.z);
  }
}

void printTree(InternalNode* nodes, LeafNode* leaves, int idx, string indent, int levels)//we need lambdas
{
  if (--levels == -1)
  {
    return; //quick n dirty :D
  }
  printf("%s", indent.c_str());
  if (idx < 0)
  {
    // leaf
    idx = -idx -1;
    LeafNode l = leaves[idx];
    printf(" leaf: idx: %d, id %d, parent %d, mc %u\n", idx, l.id, l.parentId, l.mortonCode);
  } else
  {
    InternalNode n = nodes[idx];
    printf(" node: idx: % d, id %d, p %d, lc %d, rc %d, first %d, last %d, split %d, \n", idx, n.id, n.parentId, n.leftChildId, n.rightChildId, n.first, n.last, n.split);

    printTree(nodes, leaves, n.leftChildId, indent + "L", levels);
    printTree(nodes, leaves, n.rightChildId, indent + "R", levels);

  }

}


//https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
extern "C"
__host__ void
generateHierarchy_h(vector<float3> points, const int dimX, const int dimY, const int dimZ, float *h_distances, const int numSamples, const int dfDownscale, const bool sampleGiven, const unsigned long sampleSize, vector<float3> samplePoints)
{
//  checkCudaErrors(cudaDeviceReset());
  getLastCudaError("start of generateHierarchy_h");
//  cudaError_t e;
  Reporter r;
//  r.reportStart((char *) "FULL BVH");
  const int NUM_THREADS = 256;
  const int numPoints = points.size();
  const int NUM_BLOCKS = calculateBlockNumber(numPoints, NUM_THREADS);

  printf("number of points: %d\n", numPoints);
  printf("max number in vector: %lu, capacity: %lu \n", points.max_size(), points.capacity());
  
  unsigned int *d_sortedMortonCodes;
  float3 *d_points;
  LeafNode *d_leafNodes;
  InternalNode *d_internalNodes;
  printf("before setup\n");
  setup(d_sortedMortonCodes, d_points, d_leafNodes, d_internalNodes, points);
  printf("after setup\n");

  init(d_sortedMortonCodes, d_points, d_leafNodes, d_internalNodes, numPoints, NUM_BLOCKS, NUM_THREADS, points);

  int blockSize;   // The launch configurator returned block size
  int minGridSize; // The minimum grid size needed to achieve the
  // maximum occupancy for a full device launch
  int gridSize;    // The actual grid size needed, based on input size
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, generateInternalNodes_g, 0, 0);
  // Round up according to array size
  gridSize = (numPoints-1 + blockSize - 1) / blockSize;
  // Construct internal nodes.
  r.reportStart((char *) "Building the tree");
  generateInternalNodes_g << < gridSize, blockSize >> >
                          (d_leafNodes, d_internalNodes, d_sortedMortonCodes, numPoints, numPoints - 1);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("ERROR`: %s\n", cudaGetErrorString(cudaGetLastError()));
  }
  // Node 0 is the root.
  int *d_blockades;
  cudaMalloc((void **) &d_blockades, numPoints * sizeof(int));
  cudaMemset(d_blockades, 0, numPoints * sizeof(int));

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, createBoundingVolumes_g, 0, 0);
  // Round up according to array size
  gridSize = (numPoints + blockSize - 1) / blockSize;
  createBoundingVolumes_g <<< gridSize, blockSize >>> (d_leafNodes, d_internalNodes, d_blockades, numPoints);
  r.reportEnd();
  e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("ERROR in line %d: %s\n", __LINE__,cudaGetErrorString(cudaGetLastError()));
  }

  traverse_h(d_leafNodes, d_internalNodes, dimX, dimY, dimZ, numPoints, h_distances, numSamples, dfDownscale, sampleGiven, sampleSize, samplePoints);
  cleanup(d_blockades, d_points, d_sortedMortonCodes, d_internalNodes, d_leafNodes);
//  r.reportEnd();
  return;
}

