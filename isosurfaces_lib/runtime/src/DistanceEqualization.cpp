//
// Created by mimre on 1/13/17.
//

#include <boost/filesystem/operations.hpp>
#include <malloc.h>
#include <visualization/hdr/HistogramVisualizer.h>

#include "runtime/hdr/DistanceEqualization.h"
#include "utils/hdr/StringUtils.h"
#include "utils/hdr/Report.h"
#include "isosurfaces_cuda/funcs.h"
#include "visualization/hdr/SimilarityMapVisualizer.h"
#include "DAO/hdr/RunInformation.h"
#include "json/hdr/JSONHandler.h"
#include "filehandler/hdr/FloatBinaryWriter.h"
#include "utils/hdr/MemUtils.h"
#include "runtime/hdr/SurfaceToSurface.h"
#include "runtime/hdr/globals.h"

using namespace std;
using namespace filehandler;


const std::string DistanceEqualization::DISTANCEFIELD = "DISTANCEFIELD";
const std::string DistanceEqualization::COMPUTE_DISTANCES = "COMPUTE_DISTANCES";
const std::string DistanceEqualization::REFINEMENT_STAGE = "REFINEMENT_STAGE";
const std::string DistanceEqualization::ESTIMATION_STAGE = "ESTIMATION_STAGE";
const std::string DistanceEqualization::APPROX = "APPROXIMATION";
const std::string DistanceEqualization::SELECT_ISOVALUE = "SELECT_ISOVALUE";
const std::string DistanceEqualization::BINARY_ISOVALUE_SELECTION = "BINARY_SELECT_ISOVALUE";
const std::string DistanceEqualization::FOUND_BEST = "FOUND_BEST";
const std::string DistanceEqualization::DISTANCE_EVALUATION = "DISTANCE_EVALUATION";
std::string DistanceEqualization::currentStage = "";

/**
 * Samples densely between the two values lower and upper with a specified number of samplse
 * @param lower isovalue for lower bound
 * @param upper isovalue for upper bound
 * @param numSamples number of samples between the bounds
 */
void DistanceEqualization::denseSample(float lower, float upper, int numSamples)
{
  float isovalues[numSamples];
  float step = (upper-lower)/numSamples;

  for (int i = 0; i < numSamples; ++i)
  {
    isovalues[i] = lower + step * i;
  }
  vector<int> possibleSurfaces;
  vector<DistanceField> fields;
  string fileName = settings->fileName;

  vector<string> fileparts = utils::StringUtils::split(fileName, '/');
  string fname = fileparts.back();
  PrintInformation printInfo { numSamples > 100 ? 3 : numSamples > 10 ? 2 : 1, fname};


  approximation(isovalues, numSamples, possibleSurfaces, printInfo, fields);

  std::vector<std::pair<int,float>> fieldsToSave;
  for (int j = 22; j < 26; ++j)
  {
    fieldsToSave.emplace_back(j, isovalues[j]);
    fields[j].print();
  }
  saveDistanceFields(fields, fieldsToSave, "dense-");

  DistanceField dfx = fields[23].difference(fields[24]);
  dfx.saveToFile(outputFolder + "difference" + ".dat");
  dfx.print();

  computeDistances(fields, isovalues, possibleSurfaces, 0, distances);

  bool done = evaluateDistances(distances, isovalues, true, false, false, fields);
  reportIsovalues(0, true, isovalues);

}




void DistanceEqualization::approximation(float *isovalues, int numDistanceFields, vector<int> &possibleSurfaces,
                                         const PrintInformation &printInfo, vector<DistanceField> &fields)
{

  float curIsovalue;    //Targetvalue

  int3 *points = NULL;
  int3 *lastPoints = NULL;
  long lastLength = 0;

  for (int d = 0; d < numDistanceFields; ++d)
  {
    long length = 0;

    curIsovalue = isovalues[d];

    cout << " nr " << to_string(d) << "/" << to_string(numDistanceFields-1) << endl;
    length = isosurface->calculateSurfacePoints(true, curIsovalue, &points);

    float val;
    if(d < numDistanceFields- 1)
    {
      val = isovalues[d+1];
    } else
    {
      val = curIsovalue + (isosurface->realMaxV - isosurface->realMinV)/NUM_DISTANCES;
    }

    float incr = (val - curIsovalue)/NUM_DISTANCES;
    while(length == 0)
    {

      curIsovalue += incr;
      if (curIsovalue > isosurface->realMaxV || curIsovalue != curIsovalue)
      {
        curIsovalue = isovalues[d-1];
        incr = (isovalues[d] - curIsovalue)/16;
        curIsovalue += incr;
      }
      if (d < (numDistanceFields-1) && curIsovalue > isovalues[d+1])
      {
        isovalues[d] = isovalues[d+1];
        isovalues[d+1] = curIsovalue;
        curIsovalue = isovalues[d];
      }

      isovalues[d] = curIsovalue;
      length = isosurface->calculateSurfacePoints(true, curIsovalue, &points);
    }

    //Distancefield compuation
    if (length > 0)
    {

      if(SURFACE_TO_SURFACE_DISTANCE)
      {
        if (d > 0)
        {
          //points is the current one
          //lastPoints is i-1 isosurface approx
          assert((length > 0 || lastLength > 0));

          SurfaceToSurface::calculateDistanceField(settings, config, lastLength, lastPoints, length, points, &fields, (d-1)*1000+d, (d-1)+d*1000);
        }
      } else
      {
        possibleSurfaces.push_back(d);

        calculateDistanceField(isosurface, settings, length, points, &fields);

        OutputIsosurfaceAndDistancefield(d, curIsovalue, printInfo, &fields.back());
        fields.back().id = d;
      }



    } else
    {
      cout << "\n" << d << ": empty\n";
    }
    isosurface->clear();
    if(SURFACE_TO_SURFACE_DISTANCE)
    {
      if(lastPoints != NULL)
      {
        free(lastPoints);
      }
      lastLength = length;
      lastPoints = points;
      points = NULL;
    }

    if(points != NULL)
    {
      free(points);
    }
  }
  if(lastPoints != NULL)
  {
    free(lastPoints);
  }

}

/**
 * Fixes inversion amongst isovalues and adjust the fixed and pushed values as well.
 * @param isovalues the set of isovalues
 */
void DistanceEqualization::fixInversions(float *isovalues, const int numValues)
{
  //sort isovalues to fix inversions
  for (int i = 0; i < numValues; ++i)
  {
    for (int j = i+1; j < numValues; ++j)
    {
      if(isovalues[i] > isovalues[j])
      {
        float tmp = isovalues[j];
        isovalues[j] = isovalues[i];
        isovalues[i] = tmp;
        bool tmpB = fixedValues[j];
        fixedValues[j] = fixedValues[i];
        fixedValues[i] = tmpB;
        tmpB = pushedValues[j];
        pushedValues[j] = pushedValues[i];
        pushedValues[i] = tmpB;
      }
    }

  }
}


/**
 * Handles jump situations by first verifying it (currently against extrema) and
 * then pushing a single isovalue over the jump position
 * @param distanceId the id of the distance that indicates a jump
 * @param fields the distance fields (for debug/out only)
 * @param distances the array with the distances values
 * @param isovalues the current isovalues that are subject to change
 */
void DistanceEqualization::handleJump(const int distanceId, const vector<DistanceField> &fields, const float avgDistance, float *distances,
                                      float *isovalues)
{
  if(refinementRunning && HANDLE_JUMPS)
  {
    const int idx = distanceId + 1;
    const float isovalue = isovalues[idx];

    cout << "possible jump bewteen " << distanceId << " and " << idx << " iv: " << isovalue << endl;
      if (DOUBLE_FIX)
      {
        //fix the values on both sides of the jump, continue with refinement
        if(!jumpsIds[distanceId])
        {
          float newVals[3];
          bool realJump = tightenInterval(isovalues[idx - 1], isovalues[idx], idx - 1, idx, avgDistance, distances[distanceId],
                                          newVals);
          if(realJump)
          {
            jumpHappened = true;
            ++jumpCounter;
            lastJumpId = distanceId;
            fixedValues[idx] = true;
            fixedValues[idx - 1] = true;
            jumpsIds[distanceId] = true;
            isovalues[idx-1] = newVals[0];
            isovalues[idx] = newVals[1];
            distances[distanceId] = newVals[2];
            this->juggle(isovalues, NUM_DISTANCES+1);
          }

        }

      } else
      {

        ++jumpCounter;

        int pushId = -1;
        //push isovalue over the jump, always from the smaller side of the jump
        if (distances[distanceId] < distances[idx])
        {
          //pushing a value from left to the right
          pushId = idx - 1;
          while (pushId > 0 && pushedValues[pushId])
          {
            pushId--;
            //finding the last "not fixed" value so that we push a new one over
          }

          isovalues[pushId] = isovalues[idx];
          isovalues[idx] = (isovalues[idx] + isovalues[idx + 1]) / 2;

          cout << "pushed " << pushId << endl;

        } else
        {
          //pushing a value from right to left

          pushId = idx + 1;
          while (pushId < NUM_DISTANCES && pushedValues[pushId])
          {
            pushId++;
            //finding the last "not fixed" value so that we push a new one over
          }

          isovalues[pushId] = isovalues[idx];
          isovalues[idx] = (isovalues[idx] + isovalues[idx - 1]) / 2;

          cout << "pushed " << pushId << endl;
        }

        //pushed value is fixed, the one causing the jump is moved and unfixed
        fixedValues[pushId] = true;
        pushedValues[pushId] = true;


        if (!RESTART)
        {
          fixedValues[idx] = false;
        }
      }

      if (OUTPUT_DF_AT_JUMPS)
      {
        std::vector<std::pair<int,float>> idValues;
        idValues.push_back(std::make_pair(distanceId+1, isovalue));
        saveDistanceFields(fields, idValues, "jump-");
      }
      if(RESTART)
      {
        restart(isovalues);
      }
    }
  //}
}

/**
 * Evaluates the distance values and computes statistical values
 * @param distances distance values to evaluate
 * @param isovalues set of isovalues according to @param distances
 *                  isovalues might change due to jumps
 * @param finalRun flag to check if it is the final run
 * @param fixEven flag to dictate which values need to change
 * @param computeJumps flag to see if jumps need to be checked
 * @param fields vector of distancefields needed for computation
 * @return returns whether the process is done or not
 */
bool DistanceEqualization::evaluateDistances(float *distances, float *isovalues, bool finalRun, bool fixEven, bool computeJumps, const vector<DistanceField>& fields)
{
  computeJumps = true;
  Report::begin(currentStage + DISTANCE_EVALUATION);
  float acc = 0;
  float maxD = -1e10;
  for (int j = 0; j < NUM_DISTANCES; ++j)
  {
    acc += distances[j];
    if (distances[j] > maxD)
    {
      maxD = distances[j];
    }
  }
  float avgDistance = acc/NUM_DISTANCES;


  float errors[NUM_DISTANCES];
  float accV = 0;
  acc = 0;
  float maxError = -1;

  if (refinementRunning && TAKE_ALWAYS_THE_BEST && !iterationInfos.empty())
  {
    float testSet[NUM_DISTANCES+1];
    float bestSet[NUM_DISTANCES+1];
    float testDistances[NUM_DISTANCES];
    float bestDistances[NUM_DISTANCES];
    for (int k = 0; k < NUM_DISTANCES + 1; ++k)
    {
      testSet[k] = isovalues[k];
      if (k < NUM_DISTANCES)
      {
        testDistances[k] = distances[k];
      }

    }
    int start = fixEven ? 1 : 2;
    float bestAvgError = 1e30;

    for (int i = 0; i < pow(2, NUM_DISTANCES / 2); ++i)
    {
      float acc = 0.0f;
      for (int j = 0; j < NUM_DISTANCES; ++j)
      {
        //test whether the j-th bit in i is 0 or 1,
        // if the j-th bit in i is 0 then we use the previous value for the j-th
        // isovalue, otherwise, we use the current value.
        if (((1<<j)&i))
        {
          //take new one
          testSet[j] = isovalues[j];
          testDistances[j] = distances[j];
        } else
        {
          //take old one
          testSet[j] = iterationInfos.back().isovalues[j];
          testDistances[j] = iterationInfos.back().distances[j];
        }
        acc += testDistances[j];
      }
      float avgD = acc/NUM_DISTANCES;
      float errAcc = 0.0f;
      for (int k = 0; k < NUM_DISTANCES; ++k)
      {
        errAcc = abs(testDistances[k] - avgD);
      }
      float avgError = errAcc/NUM_DISTANCES;
      if (avgError < bestAvgError)
      {
        bestAvgError = avgError;
        for (int j = 0; j < NUM_DISTANCES+1; ++j)
        {
          if (j < NUM_DISTANCES)
          {
            bestDistances[j] = testDistances[j];
          }
          bestSet[j] = testSet[j];

        }
      }
    }
    for (int l = 0; l < NUM_DISTANCES + 1; ++l)
    {
      isovalues[l] = bestSet[l];
      if(l < NUM_DISTANCES)
      {
        distances[l] = bestDistances[l];
      }
    }
  }

  for (int k = 0; k < NUM_DISTANCES; ++k)
  {
    errors[k] = abs(distances[k] - avgDistance);

    if (refinementRunning && SPIKE_TREATMENT && errors[k] > SPIKE_THRESHOLD && !iterationInfos.empty())
    {
      distances[k] = iterationInfos.back().distances[k];
      isovalues[k] = iterationInfos.back().isovalues[k];
      errors[k] = abs(distances[k] - avgDistance);
    }
    accV += errors[k] * errors[k];
    acc += errors[k];

    if (errors[k] > maxError)
    {
      maxError = errors[k];
    }

  }
  float avgError = acc/NUM_DISTANCES;
  float variance = accV/NUM_DISTANCES;
  float stdDev = sqrt(variance);

  float normalizedMaxError = maxError/avgDistance;
  float normalizedAvgError = avgError/avgDistance;
  reportDistances(iterations, finalRun, avgDistance, maxError, avgError, distances, stdDev, normalizedMaxError, normalizedAvgError, false, nullptr, 0);//FIXME report isovalues for final run as well (not necessarily here)

  bool done = false;
  if(avgError/avgDistance > bestAvgError && maxError > bestMaxError)
  {
    done = true;
  }

  if(avgError/avgDistance < bestAvgError)
  {
    bestAvgError = avgError/avgDistance;
    if(!BEST_IS_MAX)
    {
      Report::end(FOUND_BEST);
      Report::begin(FOUND_BEST);
      bestSolutionIteration = iterations;
      for (int i = 0; i < NUM_DISTANCES + 1; ++i)
      {
        bestIsovalues[i] = isovalues[i];
      }
    }
  }
  if (maxError < bestMaxError)
  {
    bestMaxError = maxError;
    if (BEST_IS_MAX)
    {
      for (int i = 0; i < NUM_DISTANCES + 1; ++i)
      {
        bestIsovalues[i] = isovalues[i];
      }
    }
  }

  if(computeJumps)
  {

    float accVNJ = 0;
    int dist_count = 0;
    float jumps[NUM_DISTANCES];
    for (int j = 0; j < NUM_DISTANCES; ++j)
    {
      jumps[j] = 0.0;
    }
    int start = fixEven ? 1 : 0;

    for (int i = start; i < NUM_DISTANCES - 1; i += 2)
    {
      if (TAKE_BEST_AT_FIX_SWITCH && i > 0)
      {
        float disLeft = distances[i-1];
        float disRight = distances[i];
        float curBestLeft = bestMovingValues[i-1].dRight;
        float curBestRight = bestMovingValues[i].dRight;
        float diff = abs(disLeft - disRight);
        float curBestDiff = abs(curBestLeft - curBestRight);
        if(diff < curBestDiff)
        {
          //the moving value got more equally spaced
          bestMovingValues[i-1].dRight = disLeft;
          bestMovingValues[i-1].isovalue = isovalues[i-1];
          bestMovingValues[i].dRight = disRight;
          bestMovingValues[i].isovalue = isovalues[i];
        }
      }


      float avgDisExceptThese = 0.0f;
      for (int j = 0; j < NUM_DISTANCES; ++j)
      {
        if(j != i && j != i+1)
        {
          avgDisExceptThese += distances[j];
        }
      }
      avgDisExceptThese /= (NUM_DISTANCES-2);
      float diff = abs(distances[i] - avgDisExceptThese);
      float error = diff / avgDisExceptThese;
      jumps[i+1] = error;
      if (error > JUMP_THRESHOLD_DIFF)
      {
        handleJump(i, fields, avgDisExceptThese, distances, isovalues);

      } else
      {
        float error1 = distances[i]-avgDistance;
        float error2 = distances[i+1]-avgDistance;
        accVNJ += error1*error1+error2*error2;
        dist_count += 2;
      }
    }

    accVNJ /= dist_count;
    float stdDevNJ = sqrt(accVNJ);
    reportDistances(iterations, finalRun, avgDistance, maxError, avgError, distances, stdDev, normalizedMaxError, normalizedAvgError, true, jumps,
                    stdDevNJ);

  }

  if(iterations == 0)
  {
    errorAfterInit = avgError/avgDistance;
  }

  Report::end(currentStage + DISTANCE_EVALUATION);

  return done;
}


void DistanceEqualization::computeDistances(vector<DistanceField> &fields, float *isovalues, vector<int> &possibleSurfaces,
                                            int numDistances, float *distances)
{
  int numDis = numDistances == 0 ? (DENSE_SAMPLING ? NUM_DISTANCES/2 : NUM_DISTANCES) : numDistances;

  if(SURFACE_TO_SURFACE_DISTANCE)
  {
    //mean distance
    for (int i = 0; i < numDis; ++i)
    {
      int idx = i*2;
      DistanceField& field1 = fields[idx];
      DistanceField& field2 = fields[idx+1];
      float m1;
      float m2;
      float acc = 0;
      for (int j = 0; j < field1.getNumDistance(); ++j)
      {
        acc += field1.getDistancePtr()[j];
      }
      m1 = acc/ field1.getNumberOfSamples();
      acc = 0;
      for (int j = 0; j < field2.getNumDistance(); ++j)
      {
        acc += field2.getDistancePtr()[j];
      }
      m2 = acc/ field2.getNumberOfSamples();


      distances[i] = (m1+m2)/2;
    }
  }
  else
  {

    MemUtils::checkmem("map start");



    int mapSize = (int) fields.size();

    SimilarityMap similarityMap = computeSimilarityMap(fields);
    similarityMap.setIsovalues(isovalues);


    vector<RepresentativesInfo> repInfo;


    for (int j = 0; j < numDis; ++j)
    {
      if(DENSE_SAMPLING)
      {
        distances[j] = 1 - similarityMap.getSimilarityMap()[0][j];
        distances[numDis+j] = 1 - similarityMap.getSimilarityMap()[numDis - 1][j];
      } else
      {
        distances[j] = 1 - similarityMap.getSimilarityMap()[j][j + 1];
      }

    }

  }

  Report::end(currentStage + COMPUTE_DISTANCES);
}

void DistanceEqualization::run(float *oldIsoValues, bool finalRun, bool skipEstimationStage)
{

  if(checkIfAlreadyProcessed())
  {
    cout << "timestep " << to_string(config->timestep) << " already processed... skipping" << endl;
    return;
  }

  Report::begin(ESTIMATION_STAGE);


  bool done = finalRun;
  ++iterations;
  printf("iterations: %d\n", iterations);
  MemUtils::checkmem("beginning");
  int mapSize = settings->similarityMapSize;
  int numIsoValues = settings->numberOfDistanceFields;
  string fileName = settings->fileName;

  vector<string> fileparts = utils::StringUtils::split(fileName, '/');
  string fname = fileparts.back();
  vector<DistanceField> fields;
  vector<int> possibleSurfaces;
  if(DENSE_SAMPLING)
  {
      float lower = 18270;
      float upper = 19000.9921875f;


    denseSample(lower, upper, NUM_DISTANCES/2);


    return;
  }


  //find isovalues

  float* isovalues;
  if(finalRun)
  {
    Report::begin(currentStage + SELECT_ISOVALUE);
    isovalues = (float *) malloc(numIsoValues * sizeof(*isovalues));
    for (int i = 0; i < numIsoValues; ++i)
    {
      isovalues[i] = bestIsovalues[i]; //todo fix all those to memcpy
    }
    Report::end(currentStage + SELECT_ISOVALUE);
  } else
  {
    isovalues = compute_isovalues(settings, oldIsoValues);
    for (int i = 0; i < NUM_DISTANCES+1; ++i)
    {
      isovalues[i] = bestIsovalues[i];
    }
  }

  PrintInformation printInfo { numIsoValues > 100 ? 3 : numIsoValues > 10 ? 2 : 1, fname};
  //Start Computation

  approximation(isovalues, numIsoValues, possibleSurfaces, printInfo, fields);


  computeDistances(fields, isovalues, possibleSurfaces, 0, distances);

  done = evaluateDistances(distances, isovalues, finalRun, false, false, fields);

  if(skipEstimationStage || MIN_ITERATIONS == 0 && MAX_ITERATIONS == 0 )
  {
    finalRun = true;
  }
  if(skipEstimationStage ||
      (iterations > MIN_ITERATIONS && (done || iterations >= MAX_ITERATIONS || finalRun)))//todo adapt this value
  {

    if(finalRun)
    {
      iterationInEstimationStage = iterations;
      errorAfterEstimation = bestAvgError;
      reportIsovalues(iterations, finalRun, isovalues);
      iterationInfo iF;
      iF.iteration = -1;
      iF.fixedEven = true;
      for (int i = 0; i < NUM_DISTANCES + 1; ++i)
      {
        iF.isovalues[i] = isovalues[i];
      }
      for (int i = 0; i < NUM_DISTANCES; ++i)
      {
        iF.distances[i] = distances[i];
      }


      if (BIN_SEARCH_MAX_ITR > 0 && BIN_SEARCH_STEP_ITR > 0)
      {
        Report::end(ESTIMATION_STAGE);
        DistanceEqualization::currentStage = REFINEMENT_STAGE;
        refinementRunning = true;
        Report::begin(REFINEMENT_STAGE);
        iterations = 0;
        binarySearchEqualization(true, true);
        Report::end(REFINEMENT_STAGE);
      }

      for (int i = 0; i < numIsoValues; ++i)
      {
        bestIsovalues[i] = isovalues[i];
      }




      cout << endl;
      cout << endl;
      Report::printTotals(true);


    }
    else {
      Report::end(ESTIMATION_STAGE);
      run(isovalues, done, skipEstimationStage);
    }

  } else
  {
    for(DistanceField& df : fields)
    {
      if(df.isDistancePtrSet())
      {
        free(df.getDistancePtr());
      }
    }
    Report::end(ESTIMATION_STAGE);
    run(isovalues, false, skipEstimationStage);
  }

  if(isovalues != NULL)
  {
    free(isovalues);
  }
  isosurface->freeImg();
  MemUtils::checkmem("map end");



}



void DistanceEqualization::saveIteration(float* isovalues, int iteration, int numIsovalues)
{
  vector<RepresentativesInfo> repInfo;
  for (int i = 0; i < numIsovalues; ++i)
  {
    RepresentativesInfo rI;
    rI.valueId = i;
    rI.isovalue = isovalues[i];
    rI.repId = i;
    rI.importance = i;
    rI.mapId = i;
    rI.filename = to_string(iteration) + "-surface-" + to_string(i) + ".dat";
    repInfo.push_back(rI);
  }


  saveRunInfo(repInfo, numIsovalues, iteration);
}

float* DistanceEqualization::compute_isovalues(Settings *settings, float* oldIsoValues)
{
  Report::begin(currentStage + SELECT_ISOVALUE);
  int numIsovalues = settings->numberOfDistanceFields;

  float min = isosurface->realMinV;
  float max = isosurface->realMaxV;



  float distanceSum = 0;
  float* isovalues = (float *) malloc(numIsovalues * sizeof(*isovalues));
  for (int i = 0; i < NUM_DISTANCES; ++i)
  {
    distanceSum += distances[i];
  }
  std::ofstream outfile;
  outfile.open(varFolder + "/" + "isovalues.csv", std::ios_base::app);

  if(distanceSum != 0)
  {//not first run
    for (int i = 0; i < NUM_DISTANCES; ++i)
    {
      distances[i] /= distanceSum;
      if(i!=0)
      {
        distances[i] += distances[i - 1];
      }
      printf("%d: %f\n", i, distances[i]);
    }

    float percentage = float(1)/float(NUM_DISTANCES);
    float curAcc = percentage;

    isovalues[0] = oldIsoValues[0];
    isovalues[NUM_DISTANCES] = oldIsoValues[NUM_DISTANCES];

    int cur_dist = 0;
    float stepD = float((DAMPENING_FACTOR-MIN_FACTOR))/float(MAX_ITERATIONS);
    float dampeningFactor = (float) (DAMPENING_FACTOR - stepD * iterations);

    outfile << iterations << ",";

    outfile << isovalues[0] <<",";

    for (int i = 1; i < numIsovalues-1; ++i)
    {
      while(curAcc>distances[cur_dist]){
        ++cur_dist;
      }
      float fac;
      if (cur_dist==0){
        fac = curAcc/distances[0];
      } else {
        fac = (curAcc-distances[cur_dist-1])/(distances[cur_dist]-distances[cur_dist-1]);
      }
      isovalues[i] = (1-fac)*oldIsoValues[cur_dist]+fac*oldIsoValues[cur_dist+1];
      isovalues[i] = (dampeningFactor * isovalues[i] + (1 - dampeningFactor) * oldIsoValues[cur_dist]);
      curAcc += percentage;
      outfile << isovalues[i] <<",";
    }
    outfile << isovalues[numIsovalues-1] <<",";

  } else
  {//first run
    outfile << iterations << ",";
    float step = (max - min) / (numIsovalues-1);
    float sum = 0.0;
    if(TIMESTEP_KEEPING)
    {
      for (int j = 0; j < numIsovalues; ++j)
      {
        sum += oldIsoValues[j];
      }
    }
    if (TIMESTEP_KEEPING && sum != 0.0)
    {
      float rangeFactor = (max-min)/(oldIsoValues[NUM_DISTANCES] - oldIsoValues[0]);
      float rangeAdder = min;
      isovalues[0] = min;
      for (int i = 1; i < numIsovalues - 1; ++i)
      {
        isovalues[i] = (oldIsoValues[i]-oldIsoValues[0]) * rangeFactor + rangeAdder;//range adaption
      }

      isovalues[numIsovalues-1] = max;
    } else
    {
      for (int i = 0; i < numIsovalues; ++i)
      {
        isovalues[i] = min + step * i;
        outfile << isovalues[i] << ",";
      }
    }
  }

  outfile << endl;
  outfile.close();
  Report::end(currentStage + SELECT_ISOVALUE);

  return isovalues;
}


void DistanceEqualization::get_uniform_histogram(int* hist, int UNIFORM_HISTOGRAM_BIN_COUNT, float* data, float dataSize, Settings settings, int histSize)
{


  int i, index;
  float gmin, gmax, oneovergrange;
  double sum, acc, threshold;


  gmin = isosurface->realMinV, gmax = isosurface->realMaxV;
  oneovergrange = 1.0 / (gmax - gmin);


  for (i = 0; i < dataSize; i++)
  {
    index = int(UNIFORM_HISTOGRAM_BIN_COUNT * (data[i] - gmin) * oneovergrange);

    if (index > UNIFORM_HISTOGRAM_BIN_COUNT - 1)
    { index = UNIFORM_HISTOGRAM_BIN_COUNT - 1; }
    if (index < 0)
    { index = 0; }
    hist[index] += 1.0;
  }


  sum = 0.0;
  // normalize
  for (i = 0; i < UNIFORM_HISTOGRAM_BIN_COUNT; i++)
  {
    sum += hist[i];
  }

  acc = 0;
  index = 0;
  threshold = sum / double(histSize);

  for (i = 0; i < histSize; i++)
  {
    while (((acc == 0.0) || ((acc + hist[index]) < threshold)) &&
           (index < UNIFORM_HISTOGRAM_BIN_COUNT - 1))
    {
      acc += hist[index];
      hist[index] = i;
      index++;
    }
    sum -= acc;
    acc = 0.0;
    if (i < histSize - 1)
    {
      threshold = sum / double(histSize - (i + 1));
    }
  }

}

void DistanceEqualization::init()
{
  outputFolder = settings->outputFolder;
  varFolder = outputFolder + config->variableName + "-" + to_string(config->timestep);
  createFolder();
  for (int i = 0; i < NUM_DISTANCES; ++i)
  {
    distances[i] = 0;
  }

  fixedValues[0] = true;
  fixedValues[NUM_DISTANCES] = true;

  DistanceEqualization::currentStage = ESTIMATION_STAGE;
}


bool DistanceEqualization::checkIfAlreadyProcessed()
{
  bool retVal = false;
  RunInformation runInformation;
  int curTS = config->timestep;

  string jsonFile = outputFolder + config->jsonFile;
  if (( access( jsonFile.c_str(), F_OK ) != -1 ))
  {
    cout << jsonFile << endl;
    runInformation = json::JSONHandler::loadJSON<RunInformation>(jsonFile);

    for (int i = 0; i < runInformation.volumes.size(); ++i)
    {
      if(runInformation.volumes[i].timestep == curTS)
      {
        retVal = true;
      }
    }

  }
  return retVal;



}

void DistanceEqualization::deleteFolder()
{
  boost::filesystem::remove_all(varFolder);
}

void DistanceEqualization::createFolder()
{
  string approx = settings->approximation ? "-approx" : "";
  string scale = settings->approximation ? "-scale_" + to_string(settings->scale) : "";
  string approxstr = approx + scale;
  string appendix = to_string(settings->similarityMapSize) + "-" +  approxstr;
  string jumps = HANDLE_JUMPS ? "-jumps" : "";
  string spikes = SPIKE_TREATMENT ? "-spikes-" + to_string(SPIKE_THRESHOLD) : "";

  varFolder += jumps;
  varFolder += spikes;
  config->jsonFile = jumps + spikes + config->jsonFile;



  cout << varFolder << endl;

  if(ISO_OUT)
  {
    isoSurfaceFolder = varFolder + "/isosurfaces-" +  appendix + "/";
    boost::filesystem::create_directories(isoSurfaceFolder);
  }
  if(SIM_PRINT)
  {
    simMapFolder = varFolder + "/similarityMap-" +  appendix + "/";
    boost::filesystem::create_directories(simMapFolder);
  }
  if (DF_OUT)
  {
    distanceFieldFolder = varFolder + DF_FOLDER;
    boost::filesystem::create_directories(distanceFieldFolder);
  }
  if (FILE_LOGGING) {
    logFolder = varFolder + "/log";
    boost::filesystem::create_directories(logFolder);
    logFile = logFolder + "/" + settings->fileName + "-" + appendix + ".log";
  }
  std::ofstream outfile;
  std::ofstream outfile2;
  std::ofstream outfile3;


  outfile.open(varFolder + "/" + "distances.csv", std::ios_base::app);
  outfile2.open(varFolder + "/" + "isovalues.csv", std::ios_base::app);
  outfile3.open(varFolder + "/" + "distances-split.csv", std::ios_base::app);
  outfile << "iterations" << ",";
  outfile2 << "iterations" << ",";
  for (int i = 0; i < NUM_DISTANCES+1; ++i)
  {
    if (i != NUM_DISTANCES)
    {
      outfile << "d" << i << ",";
    }
    outfile2 << "iv" << i << ",";
  }
  outfile << "avgD" << "," << "maxError" << "," << "averageError" << "," << "stdDev"
          << "," << "normMaxE" << "," << "normAvgE";
  outfile3 << "splits:" << endl;
  if(HANDLE_JUMPS)
  {
    outfile << "," << "jumps" << "," << "jumpId";
  }
  outfile << endl;
  outfile2 << endl;
  outfile.close();
  outfile2.close();
  outfile3.close();
}


void DistanceEqualization::printIsosurface(string fileName)
{
  FloatBinaryWriter floatBinaryWriter(fileName);

  unsigned char* data;
  long len = isosurface->printBinary(data);
  cout << "writing to file " << fileName << endl;
  floatBinaryWriter.writeFile(fileName, data, len);
  delete[] data;

}


/**
 * Copies a DistanceField, This should be replaced by a copy constructor but that made troubles
 * @param df DistanceField to copy
 * @return copied DistanceField
 */
DistanceField* DistanceEqualization::copyDistanceField(DistanceField& df)
{
  DistanceField* newDf = new DistanceField(df.getPoints(), df.getDfDownscale(),
                                      df.getDimX(),
                                      df.getDimY(),
                                      df.getDimZ(),
                                      df.getDimXOrig(),
                                      df.getDimYOrig(),
                                      df.getDimZOrig());

  newDf->id = df.id;




  float* source = df.getDistancesAsFloatPointer();
  float* destination = (float*) malloc(df.getNumberOfSamples()*sizeof(float));

  memcpy(destination, source, df.getNumberOfSamples()*sizeof(float));
  newDf->setDistancePtr(destination);

  return newDf;


}

SimilarityMap DistanceEqualization::computeSimilarityMap(vector<DistanceField> fields)
{
  vector<vector<JointHistogram> > histograms;
  int histogramSize = settings->histogramSize;

  SimilarityMap similarityMap = calculate_histogram_W(fields, (settings->dimX/config->dfDownscale) * (settings->dimY/config->dfDownscale) * (settings->dimZ/config->dfDownscale), histogramSize, fields.size(), false);

  return similarityMap;
}

void DistanceEqualization::printSimilarityMap(const int mapSize, SimilarityMap* similarityMap, vector<int>& possibleValues)
{
  SimilarityMapVisualizer similarityMapVisualizer(mapSize, similarityMap, COLOR, possibleValues);
  string app = settings->approximation ? "-approx" : "";
  string s = stringPrintf("%ssimilarityMap-%d%s-%f.bmp", simMapFolder.c_str(), mapSize, app.c_str(), DAMPENING_FACTOR);
  if(RESULTS_OUT)
  {
    similarityMapVisualizer.show(s);
  }
  if (SIM_PRINT)
  {
    printf("--Similarity map:\n");
    similarityMap->print();
    printf("--Similarity end\n");
    if(RESULTS_OUT)
    {
      similarityMap->save(simMapFolder+"map-" + to_string(DAMPENING_FACTOR) + ".dat");
    }

  }
}



void DistanceEqualization::printResults(unordered_map<int, SimilarityMap::RepInfo> *recommendedVals, float *isovalues, string fname,
                          vector<RepresentativesInfo> &repInfo, vector<DistanceField> &fields)
{
  if (ISO_OUT || DF_OUT)
  {
    int c = 0;
    int d = 0;
    unordered_map<int,int> order = unordered_map<int,int>();
    vector<int> vals = vector<int>();
    for (auto kv : (*recommendedVals))    {
      vals.push_back(kv.second.id);

    }

    sort(vals.begin(), vals.end());
    for (int i : vals)
    {
      order.insert(pair<int, int>(i, c++));
      printf("id %d, # %d\n", i, c);
    }
    c = 0;
    d = 0;


    for (auto kv : (*recommendedVals))
    {
      int fieldId = kv.second.id;

      float curIsovalue = isovalues[fieldId];
      if (ISO_OUT)
      {
        string calc = "calculating representative isosurface nr " + to_string(fieldId) + "/" + to_string(settings->numberOfDistanceFields-1);
        cout << calc << endl;

        if(OUTPUT)
        {
          isosurface->calculateSurfacePoints(false, curIsovalue, NULL);
        }
        if (!OUTPUT || !isosurface->isEmpty())
        {
          string sx = stringPrintf("%s%s-%04d-%0.2f-%0.2f-%f.dat", isoSurfaceFolder.c_str(), fname.c_str(), kv.first, curIsovalue, kv.second.isovalue, DAMPENING_FACTOR);
          if(OUTPUT)
          {
            printIsosurface(sx);
          }
          RepresentativesInfo representativesInfo = RepresentativesInfo(fieldId, curIsovalue, d++, kv.second.priority, sx, kv.second.mapId);
          repInfo.push_back(representativesInfo);
        }
        isosurface->clear();

      }
      if (DF_OUT)
      {
        cout << "saving distanceFields" << endl;


        int id = c++;
        if(INPUT_ORDER)
        {
          id = order.find(kv.second.id)->second;
        }
        DistanceField* field;
        for (DistanceField& df: fields)
        {
          if (df.id == fieldId)
          {
            field = &df;
          }
        }

        string s = stringPrintf("%sdistancefield-%f-%d.df", distanceFieldFolder.c_str(), DAMPENING_FACTOR, id);
        if(RESULTS_OUT)
        {
          field->writeToFile(s);
        }
      }
    }
  }
}


void DistanceEqualization::calculateDistanceField(Isosurface* isosurface, Settings* settings, long length, int3* points, vector<DistanceField>* fields)
{
  Report::begin(currentStage + DISTANCEFIELD);
  bool approx = settings->approximation;
  int dimX = settings->dimX;
  int dimY = settings->dimY;
  int dimZ = settings->dimZ;


  //calculate distance field
  string distanceFieldCalc = "distance field calcuation";
  if(!approx)
  {
    fields->push_back(
        DistanceField(isosurface->points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ));
  } else
  {
    fields->push_back(
        DistanceField(points, config->dfDownscale, dimX / config->dfDownscale, dimY / config->dfDownscale, dimZ / config->dfDownscale, dimX, dimY, dimZ,
                      length));


  }
  fields->back().calculateDistanceField(false, approx, settings->numberOfSamples);
  Report::end(currentStage + DISTANCEFIELD);
}

void DistanceEqualization::OutputIsosurfaceAndDistancefield(int index, float curIsovalue, PrintInformation printInfo,
                                                            DistanceField *field)
{
  string fname = printInfo.fileName;
  if (ISO_OUT && OUTPUT)
  {
    string s = stringPrintf("%s%s-%04d-%0.2f.obj", isoSurfaceFolder.c_str(),fname.c_str(), index, curIsovalue);
    printIsosurface(s);
  }

  if (DF_OUT && OUTPUT)
  {
    stringstream dfn;
    dfn << distanceFieldFolder << fname << "-df" << setfill('0') << setw(printInfo.leadingZeros) << index << "-" << curIsovalue
        << ".dst";
    field->writeToFile(dfn.str());
  }
}





void DistanceEqualization::saveRunInfo(vector<RepresentativesInfo> repInfo, int mapSize, double runtime)
{
  string jsonFile = outputFolder + config->jsonFile;

  RunInformation runInformation;

  if (( access( jsonFile.c_str(), F_OK ) != -1 ))
  {
    cout << jsonFile << endl;
	  runInformation = json::JSONHandler::loadJSON<RunInformation>(jsonFile);
  } else
  {
    std::vector<int> dimensions;
    dimensions.push_back(settings->dimX);
    dimensions.push_back(settings->dimY);
    dimensions.push_back(settings->dimZ);
    runInformation = RunInformation(dimensions);
    runInformation.dfDownscale = config->dfDownscale;
  }
  

  int ts = config->timestep;
  string name = config->variableName + "-" + to_string(DAMPENING_FACTOR);

  runInformation.addVolumeInformation(name, ts, mapSize, repInfo, runtime);

  runInformation.addSimilarityMap(name, name, ts, ts, simMapFolder+"map-" +to_string(DAMPENING_FACTOR) +".dat");

  json::JSONHandler::saveJSON(runInformation, jsonFile);

}

DistanceEqualization::DistanceEqualization(Isosurface *isosurface, Settings *settings, SingleConfig *config)
{
  this->settings = settings;
  this->config = config;
  this->isosurface = isosurface;
  marchingCubes = new MarchingCubes(isosurface);
  init();
}

void DistanceEqualization::clean()
{
  delete(marchingCubes);
}


/**
 * retrieves the information from the previous iteration and decides for every isovalue in the current "loose" set
 * whether to move it towards it's upper or lower bound
 * @param fixedEven flag to show if the even or odd values are fixed
 * @param numIsovalues number of Isovalues in total
 * @param distances array of distance values
 * @param isovalues set of isovalues, which is to be adjusted
 * @param currentIterationInfo OUTPUT creates the information struct for this iteration
 */
void DistanceEqualization::binaryIsoValueSearch(bool fixedEven, int numIsovalues, float *distances, float *isovalues,
                                                iterationInfo *currentIterationInfo)
{
  Report::begin(currentStage + BINARY_ISOVALUE_SELECTION);
  int start = fixedEven ? 1 : 2;

  int iterationInfoNumber = (int) ((iterations) % BIN_SEARCH_STEP_ITR);

  if(KEEP_BEST_SOLUTION && iterations == 0)
  {
    for (int i = 0; i < numIsovalues; ++i)
    {
      bestDistDifference[i] = 1e30;
      if(i < NUM_DISTANCES)
      {
        bestDistances[i] = distances[i];
      }
      bestIsovalues[i] = isovalues[i];
    }
  }


  if(KEEP_BEST_SOLUTION && iterationInfoNumber == 0)
  {
    for (int i = 0; i < numIsovalues; ++i)
    {
      if(i < NUM_DISTANCES)
      {
        distances[i] = bestDistances[i];
      }
      isovalues[i] = bestIsovalues[i];
      bestDistDifference[i] = 1e30;
    }
  }

  for (int i = start; i < numIsovalues-1; i+= 2)
  {
    int j = i-1;
    int k = i+1;
    float distanceJI = distances[j];
    float distanceIK = distances[i];
    float distanceSum = distanceJI + distanceIK;
    float goalDistance = distanceSum/2;

    if (KEEP_BEST_SOLUTION && abs(distanceJI-distanceIK) < bestDistDifference[i])
    {
      bestIsovalues[i] = isovalues[i];
      bestDistDifference[i] = abs(distanceJI - distanceIK);
      bestDistances[i] = distances[i];
      bestDistances[j] = distances[j];
    }

    //retrieve bound values
    boundInformation upperBound;
    boundInformation lowerBound;
    //resets the bound information whenever a a switch between fixed set happens
    if(iterationInfoNumber == 0)
    {
      upperBound.boundValue = isovalues[k];
      upperBound.distanceRight = 0.0;
      upperBound.distanceLeft = distanceSum;

      lowerBound.boundValue = isovalues[j];
      lowerBound.distanceLeft = 0.0;
      lowerBound.distanceRight = distanceSum;
    } else
    {
      upperBound = iterationInfos[iterations-1].upperBounds[i];
      lowerBound = iterationInfos[iterations-1].lowerBounds[i];
    }

    if (distanceJI > distanceIK)
    {
      upperBound.boundValue = isovalues[i];
      upperBound.distanceRight = distanceIK;
      upperBound.distanceLeft = distanceJI;
    }
    if (distanceJI < distanceIK)
    {
      lowerBound.boundValue = isovalues[i];
      lowerBound.distanceRight = distanceIK;
      lowerBound.distanceLeft = distanceJI;
    }



    //computing intersection between the two distance lines
    //k' = incr1, d' = fix1
    float incr1 = (lowerBound.distanceLeft - upperBound.distanceLeft)/(lowerBound.boundValue - upperBound.boundValue);
    float fix1 = lowerBound.distanceLeft - incr1 * lowerBound.boundValue;

    //k'' = incr2, d'' = fix2
    float incr2 = (lowerBound.distanceRight - upperBound.distanceRight)/(lowerBound.boundValue - upperBound.boundValue);
    float fix2 = lowerBound.distanceRight - incr2 * lowerBound.boundValue;

    //(d''-d')/(k' - k'')
    float newIsovalue = (fix2 - fix1)/(incr1 - incr2);


    if (!isnan(newIsovalue) && !fixedValues[i])
    {
      isovalues[i] = newIsovalue;
    }
    currentIterationInfo->upperBounds[i] = upperBound;
    currentIterationInfo->lowerBounds[i] = lowerBound;

  }
  Report::end(currentStage + BINARY_ISOVALUE_SELECTION);


}


void DistanceEqualization::binarySearchEqualization(bool fixedEven, bool start)
{
  //setup
  int numIsovalues = NUM_DISTANCES+1;
  PrintInformation printInfo { numIsovalues > 100 ? 3 : numIsovalues > 10 ? 2 : 1, ""};
  float isovalues[NUM_DISTANCES+1];

  Report::begin(currentStage + SELECT_ISOVALUE);
  vector<int> possibleSurfaces;
  vector<DistanceField> fields;
  //value retrieval
  if(iterations == 0)
  {

    fixedEven = true;
    if (start)
    {
      for (int i = 0; i < numIsovalues; ++i)
      {
        isovalues[i] = bestIsovalues[i];
      }
      if(TAKE_BEST_AT_FIX_SWITCH)
      {
        for (int j = 0; j < NUM_DISTANCES; ++j)
        {
          bestMovingValues[j].dRight = distances[j];
          bestMovingValues[j].isovalue = bestIsovalues[j];
        }
      }
    } else
    {
      //this happens when we encounter a restart
      fixInversions(isovalues, NUM_DISTANCES+1);
      approximation(isovalues, numIsovalues, possibleSurfaces, printInfo, fields);
      computeDistances(fields, isovalues, possibleSurfaces, 0, distances);
    }
  } else
  {
    for (int i = 0; i < numIsovalues; ++i)
    {
      isovalues[i] = iterationInfos.back().isovalues[i];
    }
  }
  Report::end(currentStage + SELECT_ISOVALUE);


  iterationInfo iF;
  //find new isovalues
  fixInversions(isovalues, NUM_DISTANCES+1);
  binaryIsoValueSearch(fixedEven, numIsovalues, distances, isovalues, &iF);


  //clear vectors in case there was a restart
  possibleSurfaces.clear();
  fields.clear();

  //create distancefields (~)
  approximation(isovalues, numIsovalues, possibleSurfaces, printInfo, fields);


  computeDistances(fields, isovalues, possibleSurfaces, 0, distances);



  bool finished = iterations == BIN_SEARCH_MAX_ITR;

  reportIsovalues(iterations, finished, isovalues);

  bool done = evaluateDistances(distances, isovalues, finished, fixedEven, (iterations + 1) % BIN_SEARCH_STEP_ITR == 0, fields);
  if(!done)
  {
    done = true;
    for (int j = 0; j < NUM_DISTANCES + 1; ++j)
    {
      done = done && pushedValues[j];
    }
  }

  for(DistanceField& df : fields)
  {
    if(df.isDistancePtrSet())
    {
      free(df.getDistancePtr());
    }
  }

  iF.iteration = iterations;
  iF.fixedEven = fixedEven;
  for (int i = 0; i < numIsovalues; ++i)
  {
    iF.isovalues[i] = isovalues[i];
  }
  for (int i = 0; i < NUM_DISTANCES; ++i)
  {
    iF.distances[i] = distances[i];
  }

  iterationInfos.push_back(iF);

  if (iterations % 10 == 0)
  {
    saveIteration(isovalues, iterations, numIsovalues);
  }
  if(!finished)
  {
    ++iterations;

    if (iterations % BIN_SEARCH_STEP_ITR == 0)
    {

      //retrieve best values for each single isovalue at the switch
      if(TAKE_BEST_AT_FIX_SWITCH)
      {
        for (int i = 0; i < NUM_DISTANCES; ++i)
        {
          isovalues[i] = bestMovingValues[i].isovalue;
          distances[i] = bestMovingValues[i].dRight;
        }
      }
      fixedEven = !fixedEven;
      if(jumpHappened && iterations % 10 == 0)
      {
        this->juggle(isovalues, NUM_DISTANCES+1);
        for (int i = 0; i < NUM_DISTANCES + 1; ++i)
        {
          iterationInfos.back().isovalues[i] = isovalues[i];
        }
      }

    }
    binarySearchEqualization(fixedEven, false);
  } else
  {
    if(KEEP_BEST_SOLUTION)
    {
      for (int i = 0; i < numIsovalues; ++i)
      {
        if(i < NUM_DISTANCES)
        {
          distances[i] = bestDistances[i];
        }
        isovalues[i] = bestIsovalues[i];
      }
    }
    errorAfterRefinement = bestAvgError;
    saveIteration(isovalues, iterations, numIsovalues);
  }
}


/**
 * writes array as line into the given file in csv style
 * @param outfile file to be written in
 * @param array array to write
 * @param arraySize size of the array
 */
void reportArray(ofstream &outfile, float *array, int arraySize)
{
  for (int i = 0; i < arraySize; ++i)
  {
    outfile << std::setprecision(32) << array[i] << ",";
  }
}

/**
 * prints isovalues into a file
 * @param iteration the current iteration number
 * @param finalRun flag indicating if it's the final run
 * @param isovalues the set of isovalues to print into the file
 */
void DistanceEqualization::reportIsovalues(int iteration, bool finalRun, float *isovalues)
{
  string fileName = varFolder + "/" + "isovalues.csv";

  std::ofstream outfile;
  outfile.open(fileName, std::ios_base::app);
  string itr = finalRun ? "final" : to_string(iteration);
  outfile << itr << "," << "";

  int numIV = DENSE_SAMPLING ? NUM_DISTANCES/2 : NUM_DISTANCES+1;
  reportArray(outfile, isovalues, numIV);
  outfile << endl;
  outfile.close();
}


void DistanceEqualization::reportDistances(int iteration, bool finalRun, float avgDistance, float maxError, float avgError,
                                           float *distances, float stdDev, float normalizedMaxError,
                                           float normalizedAvgError, bool reportJumps, float *error, float stdDevNoJumps)
{
  string fileName = varFolder + "/";
  string splitFile = varFolder + "/distances-split.csv";
  if(!reportJumps)
  {
    fileName += "distances.csv";
  } else
  {
    fileName += "jumps.csv";
  }
  std::ofstream outfile;
  outfile.open(fileName, std::ios_base::app);
  string itr = finalRun ? "final" : to_string(iteration);
  outfile << itr << ",";

  if(reportJumps && iteration == 0 )
  {
    outfile << "iterations,";
    for (int i = 0; i < NUM_DISTANCES; ++i)
    {
      outfile << "d" << to_string(i) << ",";
    }

    for (int i = 0; i < NUM_DISTANCES; ++i)
    {
      outfile << "j" << to_string(i) << ",";
    }
    outfile << "stdNJ" << "," << "avgD" << "," << "maxError"
            << "," << "averageError" << "," << "stdDev" << ","
            << "normMaxE" << "," << "normAvgE";
    if(HANDLE_JUMPS)
    {
      outfile << "," << "jumps";
    }
    outfile << endl;
  }
  reportArray(outfile, distances, NUM_DISTANCES);

  if(reportJumps)
  {
    reportArray(outfile, error, NUM_DISTANCES);
    outfile << stdDevNoJumps << ",";

  }
  outfile << avgDistance << "," << maxError << "," << avgError << "," << stdDev << "," << normalizedMaxError << "," << normalizedAvgError;

  if(HANDLE_JUMPS)
  {
    outfile << "," << jumpCounter << "," << lastJumpId;
  }
  outfile << endl;
  outfile.close();


  std::ofstream outfile2;
  outfile2.open(splitFile, std::ios_base::app);


  outfile2 << iteration << ",";
  vector<int> fixedIds = getFixedIds();
  for (int j = 0; j < fixedIds.size() - 1; ++j)
  {
    int start = fixedIds[j];
    int end = fixedIds[j + 1];
    int length = end-start;
    float distAcc = 0.0;
    for (int i = start; i < end; ++i)
    {
      distAcc += distances[i];
    }
    float avgDis = distAcc/length;
    outfile2 << avgDis << ",";

  }
  outfile2 << endl;

  outfile2.close();
}



/**
 * Resets the iterations to 0 and splits and resamples the isovalue range
 * This is based on some fixed/pushed values at jump points
 * The side effects of this method are not fully tested, and it currently fails
 * @param isovalues the current set of isovalues
 */
void DistanceEqualization::restart(float *isovalues)
{
  //restart the whole process, but with a value fixed
  //"split" range into two, for estimation.


  vector<int> fixedIds = this->getFixedIds();

  //resample the ranges between fixedIds
  int lastId = -1;
  this->resample(fixedIds, isovalues);


  //set all distances to 0
  for (int i = 0; i < NUM_DISTANCES; ++i)
  {
    distances[i] = 0;
  }
  iterations = 0;
}

const float *DistanceEqualization::getDistances() const
{
  return distances;
}

DistanceEqualization::DistanceEqualization()
{}

void DistanceEqualization::setIsosurface(Isosurface *isosurface)
{
  DistanceEqualization::isosurface = isosurface;
}


/**
 * This function takes the two isovalue at a jump point and tightens the interval between them
 * This is a binary search fashion approximation
 * @param isovalue1 isovalue left of the jump
 * @param isovalue2 isovalue right of the jump
 * @param id1 id of first isovalue
 * @param id2 id of second isovalue
 * @param distance initial distance between the isovalues
 * @param outvals a array of size 3 where the new isovalues and the new distance will be set: outvals = {iv1, iv2, d}
 * @return true if we actually have encountered a jump, false otherwise
 */
bool DistanceEqualization::tightenInterval(float isovalue1, float isovalue2, const int id1, const int id2,
                                           const float averageDistance, float distance, float *outvals)
{
  float delta = (isosurface->realMaxV-isosurface->realMinV)/1000;
  float left = isovalue1;
  float right = isovalue2;
  float curDistance = distance;
  if(fixedValues[id1] && fixedValues[id2])
  {
    outvals[0] = isovalue1;
    outvals[1] = isovalue2;
    outvals[2] = distance;
  } else
  {
    while (curDistance > 0 && abs(curDistance - averageDistance) > 0.2 && abs(left - right) > delta)
    {
      float middle = (left + right) / 2;
      //approximate the surface for middle
      vector<int> possibleSurfaces;
      vector<DistanceField> fields;
      float isovalues[] = {left, middle, right};
      approximation(isovalues, 3, possibleSurfaces, PrintInformation {1, ""}, fields);


      //measure distance to the other two
      float dis[3];
      computeDistances(fields, isovalues, possibleSurfaces, 3, dis);
      float disLeft = dis[0];
      float disRight = dis[1];

      //non of them is fixed
      if (disLeft > disRight)
      {
        if(fixedValues[id2])
        {
          left = (middle + left)/2;
        } else
        {
          right = middle;
        }
        curDistance = disLeft;
      } else if (disRight > disLeft)
      {
        if(fixedValues[id1])
        {
          right = (middle + right)/2;
        } else
        {
          left = middle;
        }
        curDistance = disRight;
      } else
      {
        //the distances are equal, so we can take either
        curDistance = disRight;
      }


    }
    printf("old values: %f, %f, distance %f\n", isovalue1, isovalue2, distance);
    printf("new values: %f, %f, distance %f\n", left, right, curDistance);

    //set outvalues
    outvals[0] = left;
    outvals[1] = right;
    outvals[2] = curDistance;
  }
  //see if we still encounter a jump
  return abs(curDistance - averageDistance) > 0.2;


}

void DistanceEqualization::setDistances(float *distances, const int size)
{
  for (int i = 0; i < size; ++i)
  {
    this->distances[i] = distances[i];
  }
}

/**
 * Moves a single isovalue from one side of a jump to the other side to equalize the distance.
 * Does so for all jumps
 * @param isovalues the set of isovalues to consider
 * @param numIv the number of isovalues
 */
void DistanceEqualization::juggle(float *isovalues, const int numIv)
{
  float juggleThreshold = JUGGLE_THRESHOLD;

  //get fixed ids
  vector<int> fixedIds = getFixedIds();


  for (int l = 0; l < NUM_DISTANCES + 1; ++l)
  {
    printf("isovalue %d: %f ", l, isovalues[l]);
    if(fixedValues[l])
    {
      printf("fixed %d", l);
    }
    printf ("\n");
  }
  //check distance for every interval between fixed ones
  float distAcc = 0.0f;
  for (int leftIntervalStart = 0; leftIntervalStart < fixedIds.size()-3; leftIntervalStart+= 2)
  {

    int leftOfJump = leftIntervalStart + 1;
    int rightOfJump = leftIntervalStart + 2;
    int rightIntervalEnd = leftIntervalStart + 3;

    if(leftOfJump-leftIntervalStart < 3 || rightIntervalEnd-rightOfJump < 3)
    {
      continue;
    }

    for (int i = fixedIds[leftIntervalStart]; i < fixedIds[leftOfJump]-1; ++i)
    {
      distAcc += distances[i];
    }
    float avgDistanceLeft = distAcc/fixedIds[leftOfJump];

    distAcc = 0.0f;

    for (int i = fixedIds[rightOfJump]; i < fixedIds[rightIntervalEnd]; ++i)
    {
      distAcc += distances[i];
    }
    float avgDistanceRight = distAcc/(fixedIds[rightIntervalEnd] - fixedIds[rightOfJump]);


    float disPct = avgDistanceRight > avgDistanceLeft ? avgDistanceLeft/avgDistanceRight : avgDistanceRight/avgDistanceLeft;
    disPct = 1-disPct;

    if (disPct > juggleThreshold)
    {
      int juggleIdx = avgDistanceLeft > avgDistanceRight ? rightOfJump : leftOfJump;
      int direction = avgDistanceLeft > avgDistanceRight ? 1 : -1;
      int changeVal = fixedIds[juggleIdx] + direction;
      bool found = false;
      while(!found && changeVal < NUM_DISTANCES+1 && changeVal > -1)
      {
        if(fixedValues[changeVal])
        {
          changeVal += direction;
        } else
        {
          found = true;
        }
      }
      if(changeVal < NUM_DISTANCES + 1 && changeVal > -1)
      {
        isovalues[changeVal] = (isovalues[fixedIds[juggleIdx-direction]] +
                                isovalues[fixedIds[juggleIdx-direction]-direction])/2;
        this->fixInversions(isovalues, numIv); //changes fixedIds as well


        //recalculate fixed Ids
        fixedIds = getFixedIds();

      } else
      {
        printf("no suitable value to juggle \n");
      }

    }
  }
  if(RESAMPLE_AT_JUMP)
  {
    this->resample(getFixedIds(), isovalues);
  }



}

/**
 * goes through the fixed Values and reports which ids are fixed
 * @return a vector with all the ids of fixed values
 */
vector<int> DistanceEqualization::getFixedIds()
{
  //fixed ids
  vector<int> fixedIds;
  for (int j = 0; j < NUM_DISTANCES + 1; ++j)
  {
    if(fixedValues[j])
    {
      fixedIds.push_back(j);
    }
  }
  return fixedIds;
}

/**
 * resamples the ranges between fixed ids starting with the interval [0,f_1]
 * with f_1 being the first fixed id.
 * continues with the interval [f_1,f_2], and so on.
 * @param fixedIds ids of the fixed values
 * @param isovalues set of isovalues that need to be resampled
 */
void DistanceEqualization::resample(const vector<int> &fixedIds, float *isovalues)
{
  int lastId = -1;
  for (int id : fixedIds)
  {
    if(lastId > -1)
    {
      int diff = id-lastId;
      float startValue = isovalues[lastId];
      float endValue = isovalues[id];
      float step = (endValue - startValue)/diff;
      for (int i = 0; i < diff; ++i)
      {
        isovalues[lastId+i] = startValue + step*i;
      }
    }
    lastId = id;
  }
}




/**
 * Saves distances fields specified by the @param idValuePairs to a file in the @member outputFolder
 * The filename will be @param prefix<id>-<isovalue>.dat
 * @param fields the vector with the distancefields
 * @param idValuePairs information about the fields saved (id, isovalue)
 * @param prefix filename prefix
 */
void DistanceEqualization::saveDistanceFields(const vector<DistanceField> &fields,
                                              const vector<pair<int, float>> &idValuePairs,
                                              string prefix)
{
  for (int i = 0; i < idValuePairs.size(); ++i)
  {
    int distanceId = idValuePairs[i].first;
    float isovalue = idValuePairs[i].second;
    auto xx = fields[distanceId];
    DistanceField *df = copyDistanceField(const_cast<DistanceField &>(fields[distanceId]));



    string fileName = outputFolder + prefix + to_string(distanceId) + "-" + to_string(isovalue) + ".dat";
    df->saveToFile(fileName);

    //cleanup
    free(df->getDistancePtr());
    delete df;
  }

}