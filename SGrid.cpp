#include "SGrid.h"

#include <float.h>

#include <iostream>
#include <fstream>

SGrid::~SGrid(){
  m_convertedGrid->Delete();
}

SGrid::SGrid(){
  m_className = "SGrid ";
  m_convertedGrid = NULL;
  m_gridVolume = 0;
}

//method to load the .vti file and store as unstructured grid
void SGrid::convert(std::string filename){

    // First read the file
    vtkSmartPointer<vtkXMLImageDataReader> reader =  vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();
    std::cout << m_className << "(" << filename << ")" << "read error " << reader->GetErrorCode() << std::endl;

    //Data analysis

    vtkImageData *imgData = reader->GetOutput();
    vtkDataSet *readData = reader->GetOutputAsDataSet();

    double origin[3];
    double spacing[3];

    imgData->GetOrigin(origin);
    imgData->GetSpacing(spacing);

    int extent[6];
    imgData->GetExtent(extent);

    //outData->SetExtent(extent);

    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();
    points->SetNumberOfPoints(imgData->GetNumberOfPoints());

    vtkIdType pointId = 0;
    int ijk[3];
      for (ijk[2] = extent[4]; ijk[2] <= extent[5]; ijk[2]++)
        {
        for (ijk[1] = extent[2]; ijk[1] <= extent[3]; ijk[1]++)
          {
          for (ijk[0] = extent[0]; ijk[0] <= extent[1]; ijk[0]++)
            {
            double coord[3];

            for (int axis = 0; axis < 3; axis++)
              {
              coord[axis] = origin[axis] + spacing[axis]*ijk[axis];
              }

            points->SetPoint(pointId, coord);
            pointId++;
            }
          }
        }



    //create a new grid:
    vtkSmartPointer<vtkUnstructuredGrid> outputGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    //outputGrid->SetExtent
    outputGrid->SetPoints(points.GetPointer());


    //copy scalars:
    for(int i = 0 ; i < readData->GetPointData()->GetNumberOfArrays() ; i++)
    {
      outputGrid->GetPointData()->AddArray(readData->GetPointData()->GetArray(i));
    }


    //assign cells to outputgrid
    //double gridVolume;
    //calculate volume of each cell
    //calculateVolume(readData->GetCell(34651100));
    unsigned int noc = readData->GetNumberOfCells();
    // int ipCount = 0;
    // int opCount = 0;
    for(unsigned int i = 0; i < noc; i++){
        if(readData->GetCell(i)->GetCellType() == VTK_VOXEL){
          vtkCell* currentCell = readData->GetCell(i);
          m_gridVolume += 1.0f;
          outputGrid->InsertNextCell(currentCell->GetCellType(),currentCell->GetPointIds());
        }
    }

     outputGrid->GetPointData()->SetActiveScalars("ImageFile");

     m_convertedGrid = outputGrid;


     /*std::cout << "[og]no. of cells=" << outputGrid->GetNumberOfCells() << std::endl;
     std::cout << "[og]no. of points=" << outputGrid->GetNumberOfPoints() << std::endl;
     std::cout << "[og]no. of arrays=" << outputGrid->GetPointData()->GetNumberOfArrays() << std::endl;
     std::cout << "[og]name=" << outputGrid->GetPointData()->GetArray(0)->GetName() << std::endl;
     std::cout << "[og]maxId=" << outputGrid->GetPointData()->GetArray(0)->GetMaxId() << std::endl;
     std::cout << "[og]tuple0=" << outputGrid->GetPointData()->GetArray(0)->GetTuple1(0) << "\n";
     std::cout << "[og]tuple23=" << outputGrid->GetPointData()->GetArray(0)->GetTuple1(234567) << "\n";
     //std::cout << "[og]tuple678999=" << outputGrid->GetPointData()->GetArray(0)->GetTuple1(24999999) << "\n";
     std::cout << "[og]gridVolume=" << m_gridVolume << std::endl;*/



}



//method to load the given file and convert all polyhedrons to tetrahedrons
void SGrid::testFunc(std::string str){

 std::cout << str << std::endl;
}


