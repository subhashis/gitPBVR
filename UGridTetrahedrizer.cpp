#include "UGridTetrahedrizer.h"

#include <float.h>

#include <iostream>
#include <fstream>

UGridTetrahedrizer::~UGridTetrahedrizer(){
  m_convertedGrid->Delete();
}

UGridTetrahedrizer::UGridTetrahedrizer(){
  m_className = "UGridTetrahedrizer ";
  m_convertedGrid = NULL;
  m_gridVolume = 0;
}

//method to load the given file and convert all polyhedrons to tetrahedrons
void UGridTetrahedrizer::convert(std::string filename){

  //first, read the grid
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(filename.c_str());
  reader->Update();
  std::cout << m_className << "(" << filename << ")" << "read error " << reader->GetErrorCode() << std::endl;

  vtkSmartPointer<vtkUnstructuredGrid> readGrid = reader->GetOutput();

  vtkSmartPointer<vtkUnstructuredGrid> inputGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

#if TUMOR
  inputGrid = extractDeadOnly(readGrid);
#endif

#if DRAGON
  //vtkSmartPointer<vtkUnstructuredGrid> inputGrid = extractDragon(readGrid);
  inputGrid = readGrid;
#endif

#if HEART
  inputGrid = readGrid;
#endif
  

  if(inputGrid == NULL){
    std::cout << m_className << "Could not read given file. Aborting" << std::endl;
    return;
    
  }

  vtkCellArray* inputCells = inputGrid->GetCells();

  //create a new grid:
  vtkSmartPointer<vtkUnstructuredGrid> outputGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
  outputGrid->SetPoints(inputGrid->GetPoints());
  
  //creating a debug file
  ofstream mydebugFile;
  //mydebugFile.open ("debugFile.txt");
  //mydebugFile << "Writing this to a file.\n";
  //mydebugFile.close();
  
  //set new array to store variance
  vtkDoubleArray* varArray = vtkDoubleArray::New();
  varArray->SetName("variance");  // ... fill the colors array
  
  std::cout << m_className << "no. of arrays=" << inputGrid->GetPointData()->GetNumberOfArrays() << std::endl;
  //copy scalars:
  for(int i = 0 ; i < inputGrid->GetPointData()->GetNumberOfArrays() ; i++)
  {
    outputGrid->GetPointData()->AddArray(inputGrid->GetPointData()->GetArray(i));
    std::cout << m_className << inputGrid->GetPointData()->GetArray(i)->GetMaxId() << "\n";
    
  }
  std::cout << m_className << "no. of arrays=" << outputGrid->GetPointData()->GetNumberOfArrays() << std::endl;
  std::cout << m_className << "array name = "<<varArray->GetName() << "\n";
  
  for(int i = 0 ; i <= inputGrid->GetPointData()->GetArray(0)->GetMaxId() ; i++)
  {
    //outputGrid->GetPointData()->AddArray(inputGrid->GetPointData()->GetArray(i));
    //mydebugFile << m_className << "value " << i << "=" << inputGrid->GetPointData()->GetArray(0)->GetTuple1(i) << "\n";
    varArray->InsertTuple1(i,5);
  }
  
  
  // add this new array to output grid
  outputGrid->GetPointData()->AddArray(varArray);
  
  std::cout << m_className << outputGrid->GetPointData()->GetArray(1)->GetTuple1(0) << "\n";
  std::cout << m_className << outputGrid->GetPointData()->GetArray(1)->GetTuple1(23) << "\n";
  std::cout << m_className << outputGrid->GetPointData()->GetArray(1)->GetTuple1(678999) << "\n";
  
  std::cout << m_className << "new no. of arrays=" << outputGrid->GetPointData()->GetNumberOfArrays() << std::endl;
  
  
#if TUMOR
  outputGrid->GetPointData()->SetActiveScalars("dead");
#endif

#if DRAGON
  outputGrid->GetPointData()->SetActiveScalars("SplatterValues");
#endif
    
#if HEART
  outputGrid->GetPointData()->SetActiveScalars("scalars");
#endif
  #if DEBUG_UGRIDTETRAHEDRIZER
    std::cout << m_className << "---- Grid Conversion Log --- " << std::endl;
    vtkSmartPointer<vtkDataArray> deadData = inputGrid->GetPointData()->GetArray("scalars");
    std::cout << m_className << "  Data type = " << deadData->GetDataTypeAsString() << std::endl;
  #endif

  //now check the cells:
  unsigned int numCells = inputGrid->GetNumberOfCells();
  unsigned int numNonTetras = 0;
  unsigned int numDegenerated = 0;

  for(unsigned int i = 0 ; i < numCells ; i++){
    //if we got a tetra here, everythings fine.
    if(inputGrid->GetCell(i)->GetCellType() == VTK_TETRA){
      vtkCell* currentCell = inputGrid->GetCell(i);
      m_gridVolume += calculateVolume(currentCell);
      outputGrid->InsertNextCell(currentCell->GetCellType(),currentCell->GetPointIds());
    }

    //otherwise, ignore the current cell and replace it with the new tetrahedrons
    else{
      std::cout << "cell type = " << inputGrid->GetCell(i)->GetCellType() << std::endl;
      numNonTetras++;
      vtkSmartPointer<vtkUnstructuredGrid> tetraSet = convertCell(inputGrid->GetCell(i));
      if(tetraSet){
        for(int i = 0 ; i < tetraSet->GetNumberOfCells() ; i++){
          outputGrid->InsertNextCell(tetraSet->GetCell(i)->GetCellType(),tetraSet->GetCell(i)->GetPointIds());
        }
      }
      else
        numDegenerated++;
    }
  }
  std::cout << "number of cells in input grid = " << numCells << std::endl;
  std::cout << "number of non-tetra cells in input grid = " << numNonTetras << std::endl;
  std::cout << "number of degenerated cells in input grid = " << numDegenerated << std::endl;
  std::cout << "number of cells in output grid = " << outputGrid->GetNumberOfCells() << std::endl;

  std::cout << "whole volume = " << m_gridVolume << std::endl;

  double max[3] = {DBL_MIN,DBL_MIN,DBL_MIN};
  double min[3] = {DBL_MAX,DBL_MAX,DBL_MAX};
  vtkPoints* pts = outputGrid->GetPoints();
  for(int i = 0 ; i < outputGrid->GetNumberOfPoints() ; i++){
    double currentPoint[3];
    outputGrid->GetPoint(i,currentPoint);
    for(int i = 0 ; i < 3 ; i++){
      if(currentPoint[i] > max[i])
        max[i] = currentPoint[i];
      if(currentPoint[i] < min[i])
        min[i] = currentPoint[i];
    }
  }

  double centerOfMass[3];
  for(int i = 0 ; i < 3 ; i++){
    centerOfMass[i] = min[i] + 0.5*(max[i] - min[i]);
  }


  std::cout << m_className << " Center of mass = " << centerOfMass[0] << " , " << centerOfMass[1] << " , " << centerOfMass[2] << std::endl;

#if TUMOR
  outputGrid->GetPointData()->SetActiveScalars("dead");
#endif

#if DRAGON
  outputGrid->GetPointData()->SetActiveScalars("SplatterValues");
#endif

#if HEART
  outputGrid->GetPointData()->SetActiveScalars("scalars");
#endif

  m_convertedGrid = outputGrid;

  #if DEBUG_UGRIDTETRAHEDRIZER
    std::cout << m_className << "  Num cells input = " << numCells << std::endl;
    std::cout << m_className << "  Num non tetras  = " << numNonTetras << std::endl;
    std::cout << m_className << "  Num degenerated = " << numDegenerated << std::endl;
    std::cout << m_className << "  Num cells output= " << outputGrid->GetNumberOfCells() << std::endl;
    std::cout << m_className << "  Should be:        " << inputGrid->GetNumberOfCells() - numDegenerated << std::endl;
    std::cout << m_className << "---------------------------- " << std::endl;
  #endif
}

//calculate volume of a single cell
double UGridTetrahedrizer::calculateVolume(vtkCell *cell){
  
  vtkPoints* pts = cell->GetPoints();  
  
  double a[3];
  pts->GetPoint(0,a);
  double b[3];
  pts->GetPoint(1,b);
  double c[3];
  pts->GetPoint(2,c);
  double d[3];
  pts->GetPoint(3,d);


  
  double tmp1[3];
  double tmp2[3];
  double tmp3[3];

  for(int i = 0 ; i < 3 ; i++){
    tmp1[i] = a[i] - d[i];
    tmp2[i] = b[i] - d[i];
    tmp3[i] = c[i] - d[i];
  }

  //tmp2 x tmp3
  double tmp4[3];
  tmp4[0] = tmp2[1] * tmp3[2] - tmp2[2] * tmp3[1];
  tmp4[1] = tmp2[2] * tmp3[0] - tmp2[0] * tmp3[2];
  tmp4[2] = tmp2[0] * tmp3[1] - tmp2[1] * tmp3[0];

  //tmp1 . tmp4
  double dot = tmp1[0] * tmp4[0] + tmp1[1] * tmp4[1] + tmp1[2] * tmp4[2];
  if( dot < 0)
    dot *= -1;
  double volume = dot / 6;


#if DEBUG_UGRIDTETRAHEDRIZER
  std::cout << "a = " << a[0] << " , " << a[1] << " , " << a[2] << std::endl;
  std::cout << "b = " << b[0] << " , " << b[1] << " , " << b[2] << std::endl;
  std::cout << "c = " << c[0] << " , " << c[1] << " , " << c[2] << std::endl;
  std::cout << "d = " << d[0] << " , " << d[1] << " , " << d[2] << std::endl;
  std::cout << "Volume = " << volume << std::endl;
#endif
  return volume;
}


//convert a single cell to a set of tetrahedrons
vtkSmartPointer<vtkUnstructuredGrid> UGridTetrahedrizer::convertCell(vtkCell* cell){
  
  int numPoints = cell->GetNumberOfPoints();
  if(numPoints < 4){
    //std::cout << "Num Points = " << numPoints << std::endl;
    return NULL;
  }
  //perform 3D delaunay on the points of the cell:
  vtkSmartPointer<vtkDelaunay3D> delaunay = vtkSmartPointer<vtkDelaunay3D>::New();
  vtkSmartPointer<vtkPolyData> poly = vtkSmartPointer<vtkPolyData>::New();
  poly->SetPoints(cell->GetPoints());
  delaunay->SetInput(poly);
  delaunay->Update();
  
  return delaunay->GetOutput();
}


vtkSmartPointer<vtkUnstructuredGrid> UGridTetrahedrizer::extractDeadOnly(vtkSmartPointer<vtkUnstructuredGrid> input){

  //split up the input into NUM_THRESHOLD_STEPS datasets according to the scalar range:

  vtkSmartPointer<vtkUnstructuredGrid> output;

  vtkPointData* pointData = input->GetPointData();
  pointData->SetActiveScalars("dead");
  vtkDataArray* scalars = pointData->GetScalars("dead");

  double* scalarRange = scalars->GetRange();
  std::cout << m_className << "Scalar range = " << scalarRange[0] << " - " << scalarRange[1] << std::endl;

  vtkThreshold* thresholdFilter = vtkThreshold::New();
  double lowerBound = /*scalarRange[0] + 0.5 * (scalarRange[1] - scalarRange[0]);*/0.2f;
  double upperBound = scalarRange[1];
  std::cout<< m_className << "Extracting from " << lowerBound << " to " << upperBound << std::endl;
  thresholdFilter->SetInput(input);
  thresholdFilter->ThresholdBetween(lowerBound, upperBound);
  thresholdFilter->Update();
  vtkUnstructuredGrid* currentGrid = thresholdFilter->GetOutput();
  output = currentGrid;
  
  return output;

}


vtkSmartPointer<vtkUnstructuredGrid> UGridTetrahedrizer::extractDragon(vtkSmartPointer<vtkUnstructuredGrid> input){

  //split up the input into NUM_THRESHOLD_STEPS datasets according to the scalar range:

  vtkSmartPointer<vtkUnstructuredGrid> output;

  vtkPointData* pointData = input->GetPointData();
  pointData->SetActiveScalars("dead");
  vtkDataArray* scalars = pointData->GetScalars("SplatterValues");

  double* scalarRange = scalars->GetRange();
  std::cout << m_className << "Scalar range = " << scalarRange[0] << " - " << scalarRange[1] << std::endl;

  vtkThreshold* thresholdFilter = vtkThreshold::New();
  double lowerBound = /*scalarRange[0] + 0.5 * (scalarRange[1] - scalarRange[0]);*/0.001f;
  double upperBound = scalarRange[1];
  std::cout<< m_className << "Extracting from " << lowerBound << " to " << upperBound << std::endl;
  thresholdFilter->SetInput(input);
  thresholdFilter->ThresholdBetween(lowerBound, upperBound);
  thresholdFilter->Update();
  vtkUnstructuredGrid* currentGrid = thresholdFilter->GetOutput();
  output = currentGrid;

  return output;

}
