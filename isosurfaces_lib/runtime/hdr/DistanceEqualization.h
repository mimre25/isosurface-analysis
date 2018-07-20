//
// Created by mimre on 1/13/17.
//

#ifndef ISOSURFACES_DISTANCE_EQUALIZATION_H
#define ISOSURFACES_DISTANCE_EQUALIZATION_H


#include <sys/stat.h>
#include "types/hdr/Isosurface.h"
#include "types/hdr/Settings.h"
#include "types/hdr/MarchingCubes.h"
#include "types/hdr/DistanceField.h"
#include "types/hdr/SimilarityMap.h"
#include "DAO/hdr/RepresentativesInfo.h"
#include "DAO/hdr/SingleConfig.h"
#include "Single.h"

//general parameters
#define NUM_DISTANCES 16
#define MIN_ITERATIONS 0
#define MAX_ITERATIONS 0
#define DAMPENING_FACTOR 0.66
#define MIN_FACTOR 0.05
#define BEST_IS_MAX false
#define BIN_SEARCH_MAX_ITR 180
#define BIN_SEARCH_STEP_ITR 5
#define KEEP_BEST_SOLUTION false
#define SURFACE_TO_SURFACE_DISTANCE false

#define TAKE_BEST_AT_FIX_SWITCH true

#define TIMESTEP_KEEPING false
#define NO_ESTIMATION_STAGE true


//jump related parameters
#define OUTPUT_DF_AT_JUMPS false
#define HANDLE_JUMPS false
#define RESTART false
#define DOUBLE_FIX true
#define JUMP_THRESHOLD_DISTANCE 0.2
#define JUMP_THRESHOLD_DIFF 0.5
#define RESAMPLE_AT_JUMP false
#define JUGGLE_THRESHOLD 0.30

#define SPIKE_TREATMENT false
#define SPIKE_THRESHOLD 0.05

#define TAKE_ALWAYS_THE_BEST false

//debug params
#define DENSE_SAMPLING false


class DistanceEqualization
{
public:
  static const std::string DISTANCEFIELD;
  static const std::string COMPUTE_DISTANCES;
  static const std::string REFINEMENT_STAGE;
  static const std::string ESTIMATION_STAGE;
  static const std::string APPROX;
  static const std::string SELECT_ISOVALUE;
  static const std::string BINARY_ISOVALUE_SELECTION;
  static const std::string FOUND_BEST;
  static const std::string DISTANCE_EVALUATION;
  static std::string currentStage;
  int bestSolutionIteration = -1;
  float errorAfterEstimation = -1;
  float errorAfterRefinement = -1;
  float errorAfterInit = -1;
  int iterationInEstimationStage = -1;
  int jumpCounter = 0;
  int lastJumpId = -1;
  bool refinementRunning = false;
  void fixInversions(float *isovalues, const int numValues);
  bool jumpHappened = false;

  void saveDistanceFields(const vector<DistanceField> &fields, const vector<pair<int, float>> &idValuePairs,
                          string prefix);
  void handleJump(const int distanceId, const vector<DistanceField> &fields, const float avgDistance, float *distances,
                    float *isovalues);

  void denseSample(float lower, float upper, int numSamples);
  void run(float *oldIsoValues, bool finalRun, bool skipEstimationStage);
  void clean();
  DistanceField* copyDistanceField(DistanceField& df);


  bool tightenInterval(float isovalue1, float isovalue2, const int id1, const int id2,
                         const float averageDistance, float distance, float *outvals);

  void deleteFolder();
  struct boundInformation
  {
    float boundValue; //isovalue
    float distanceLeft; //distance to left neighbor
    float distanceRight; //distance to right neighbor
  };

  struct iterationInfo
  {
    float isovalues[NUM_DISTANCES+1]; // isovalues of the iteration
    boundInformation lowerBounds[NUM_DISTANCES+1]; // lower bounds of the iteration
    boundInformation upperBounds[NUM_DISTANCES+1]; // upper bounds of the iteration

    float distances[NUM_DISTANCES]; // distances of the iteration
    bool fixedEven; // flag if even or odd values are fixed
    int iteration; // iteration number
  };

  /**
 * struct to save information about the currently moving value
   * */
  struct movingValueAndDistance
  {
    /**
     * the isovalue that is changing
     */
    float isovalue;
    /**
     * the distance to the left neighbor
     */
    float dRight;
  };
  movingValueAndDistance bestMovingValues[NUM_DISTANCES];



  bool fixedValues[NUM_DISTANCES+1] = { false };//flags for values that are fixed (occuring after a jump)
  bool pushedValues[NUM_DISTANCES+1] = { false };//flags for values that have been pushend and thus are not going to be pushed anymore
  bool jumpsIds[NUM_DISTANCES+1] = { false };//flags to show if a jump was found at that point



public:

  bool checkIfAlreadyProcessed();
  const float *getDistances() const;
  void restart(float *isovalues);

protected:
  Isosurface* isosurface;
public:
  void setIsosurface(Isosurface *isosurface);

protected:
  Settings*  settings;
  SingleConfig* config;
  MarchingCubes* marchingCubes;
  string varFolder;
  float distances[NUM_DISTANCES];
  float* compute_isovalues(Settings *settings, float* oldIsoValues);
  void get_uniform_histogram(int* hist, int UNIFORM_HISTOGRAM_BIN_COUNT, float* data, float dataSize, Settings settings, int histSize);




  void init();
  void createFolder();
  void printIsosurface(string fileName);
  SimilarityMap computeSimilarityMap(vector<DistanceField> fields);
  void printSimilarityMap(const int mapSize, SimilarityMap* similarityMap, vector<int>& possibleValues);

  void printResults(unordered_map<int, SimilarityMap::RepInfo> *recommendedVals, float *isovalues, string fname,
                      vector<RepresentativesInfo> &repInfo, vector<DistanceField> &fields);
  void calculateDistanceField(Isosurface *isosurface, Settings *settings, long length, int3 *points,
                              vector<DistanceField> *fields);
  void OutputIsosurfaceAndDistancefield(int index, float curIsovalue, PrintInformation printInfo,
                                          DistanceField *field);
  void saveRunInfo(vector<RepresentativesInfo> repInfo, int mapSize, double runtime);
  int iterations = 0;

  void binarySearchEqualization(bool fixedEven, bool start = false);

  void binaryIsoValueSearch(bool fixedEven, int numIsovalues, float *distances, float *isovalues,
                              iterationInfo *currentIterationInfo);

  void reportIsovalues(int iteration, bool finalRun, float *isovalues);
  void
  reportDistances(int iteration, bool finalRun, float avgDistance, float maxError, float avgError,
                    float *distances, float stdDev, float normalizedMaxError,
                    float normalizedAvgError, bool reportJumps, float *error, float stdDevNoJumps);

  void approximation(float *isovalues, int numDistanceFields, vector<int> &possibleSurfaces,
                       const PrintInformation &printInfo, vector<DistanceField> &fields);


  bool evaluateDistances(float *distances, float *isovalues, bool finalRun, bool fixEven, bool fixOdd, const vector<DistanceField>& fields);

  void computeDistances(vector<DistanceField> &fields, float *isovalues, vector<int> &possibleSurfaces,
                          int numDistances, float *distances);





  vector<iterationInfo> iterationInfos;


  void saveIteration(float* isovalues, int iteration, int numIsovalues = NUM_DISTANCES+1);


public:
  DistanceEqualization(Isosurface *isosurface, Settings *settings, SingleConfig* config);

  DistanceEqualization();


  string histogramFolder;
  string isoSurfaceFolder;
  string outputFolder;
  string simMapFolder;
  string distanceFieldFolder;
  string logFolder;
  string logFile;

  float bestAvgError = 10e20;
  float bestMaxError = 10e20;
  float bestIsovalues[NUM_DISTANCES+1] = {137.58087158203125,137.7071075439453125,138.2776336669921875,140.0638885498046875,144.370941162109375,151.1020965576171875,157.6292877197265625,163.8123931884765625,174.6049346923828125,214.08880615234375,324.660186767578125,604.75189208984375,1200.432861328125,2765.340087890625,6707.39599609375,14770.724609375,33156.9921875};


  float bestDistDifference[NUM_DISTANCES+1];
  float bestDistances[NUM_DISTANCES];

  void setDistances(float *distances, const int size);

  void juggle(float *isovalues, const int numIv);

  vector<int> getFixedIds();

  void resample(const vector<int> &fixedIds, float *isovalues);
};


#endif //ISOSURFACES_DISTANCE_EQUALIZATION_H
