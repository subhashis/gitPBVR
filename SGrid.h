/*
A class to read an structured grid data.
*/

#ifndef SGRID_H_
#define SGRID_H_

#include "Singleton.h"

#include "defs.h"

#include <string>
#include <iostream>

#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkProperty.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFieldData.h>
#include <vtkCellTypes.h>
#include <vtkDelaunay3D.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkXMLImageDataReader.h>


class SGrid : public Singleton<SGrid>{

  friend class Singleton<SGrid>;

public:
  ~SGrid();
  void convert(std::string filename);
  vtkSmartPointer<vtkUnstructuredGrid> getOutput(){return m_convertedGrid;};
  double getGridVolume(){return m_gridVolume;};
  void testFunc(std::string testString);

protected:
  SGrid();
  SGrid(SGrid& dontcopy){};


  vtkSmartPointer<vtkUnstructuredGrid> m_convertedGrid;
  //double calculateVolume(vtkCell* cell);*/

  char* m_className;
  double m_gridVolume;
};


#endif
