/*
A class to read an unstructured grid and convert all its cells to tetrahedrons
*/


#ifndef UGRIDTETRAHEDRIZER_H_
#define UGRIDTETRAHEDRIZER_H_

#include "Singleton.h"

#include "defs.h"

#include <string>
#include <iostream>

#include <vtkUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkGenericCell.h>
#include <vtkSmartPointer.h>
#include <vtkDelaunay3D.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkThreshold.h>

class UGridTetrahedrizer : public Singleton<UGridTetrahedrizer>{

  friend class Singleton<UGridTetrahedrizer>;

public:
  ~UGridTetrahedrizer();
  void convert(std::string filename);
  vtkSmartPointer<vtkUnstructuredGrid> getOutput(){return m_convertedGrid;};
  double getGridVolume(){return m_gridVolume;};

protected:
  UGridTetrahedrizer();
  UGridTetrahedrizer(UGridTetrahedrizer& dontcopy){};

  vtkSmartPointer<vtkUnstructuredGrid> convertCell(vtkCell* cell);
  vtkSmartPointer<vtkUnstructuredGrid> m_convertedGrid;
  vtkSmartPointer<vtkUnstructuredGrid> extractDeadOnly(vtkSmartPointer<vtkUnstructuredGrid> input);
  vtkSmartPointer<vtkUnstructuredGrid> extractDragon(vtkSmartPointer<vtkUnstructuredGrid> input);
  double calculateVolume(vtkCell* cell);

  char* m_className;
  double m_gridVolume;
};


#endif
