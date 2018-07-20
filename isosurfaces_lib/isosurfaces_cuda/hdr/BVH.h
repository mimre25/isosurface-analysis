//
// Created by mimre on 8/3/16.
//
#ifndef __BVH__
#define __BVH__

struct BoundingBox
{
  float3 minCorner;
  float3 maxCorner;
};


struct Node
{
  Node* parent;
  bool internal;
  int id;
  int parentId;
  BoundingBox boundingBox;
};

struct LeafNode : Node
{
  float3 point;
  unsigned int mortonCode;
};

struct InternalNode : Node
{
  Node* childA;
  Node* childB;
  int leftChildId;
  int rightChildId;
  float3 center;
  int first;
  int last;
  int split;
};


struct BVH
{
  InternalNode* nodes;
  LeafNode* leaves;

  BVH(InternalNode* iNodes, LeafNode* lNodes)
  {
    nodes = iNodes;
    leaves = lNodes;
  }
};

#endif //__BVH__