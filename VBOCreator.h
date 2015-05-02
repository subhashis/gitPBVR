#ifndef VBOCREATOR_H_
#define VBOCREATOR_H_

#include "defs.h"
#include "Singleton.h"
#include "UGridTetrahedrizer.h"
#include "SGrid.h"

#include <vtkPointData.h>

#include <vector>


class VBOCreator : public Singleton<VBOCreator>{

  friend class Singleton<VBOCreator>;

public:
  ~VBOCreator();
  void convertUGridToVBO();
  void convertSGridToVBO();
  
  void initVBOTetras(unsigned int numTetras);
  void initVBOVoxels(unsigned int numVoxels);
  void initPBOSubPixels(unsigned int windowWidth, unsigned int windowHeight);
  void initPBOSubPixelsDepth(unsigned int windowWidth, unsigned int windowHeight);
  void initPBOPixels(unsigned int windowWidth, unsigned int windowHeight);
  void initTexture(unsigned int windowWidth, unsigned int windowHeight);

  GLuint getVBOTetra(){return m_vboTetraID;};
  GLuint getVBOVoxel(){return m_vboVoxelID;};
  GLuint getPBOSubPixels(){return m_pboSubPixelID;};
  GLuint getPBOSubPixelsDepth(){return m_pboSubPixelDepthID;};
  GLuint getPBOPixels(){return m_pboPixelID;};
  GLuint getScreenMapTextureID(){return m_screenMapTextureID;};

  unsigned int getNumTetras(){return m_numTetras;};
  unsigned int getNumVoxels(){return m_numVoxels;};
  tetraFloat* getTetras(){return m_tetras;};
  voxelFloat* getVoxels(){return m_voxels;};
  float* getTetraVolumes(){return m_tetraVolumes;};
  float* getVoxelVolumes(){return m_voxelVolumes;};

protected:
  VBOCreator();
  VBOCreator(VBOCreator& dontcopy){};
  
  tetraFloat cellToTetra(unsigned int cellID, vtkUnstructuredGrid* grid);
  voxelFloat cellToVoxel(unsigned int cellID, vtkUnstructuredGrid* grid);
  float cellVolume(tetraFloat tetra);

  unsigned int m_numTetras;
  unsigned int m_numVoxels;
  GLuint m_vboTetraID;
  GLuint m_vboVoxelID;
  GLuint m_pboSubPixelID;
  GLuint m_pboSubPixelDepthID;
  GLuint m_pboPixelID;
  GLuint m_screenMapTextureID;
  char* m_className;
  tetraFloat* m_tetras;
  voxelFloat* m_voxels;
  float* m_tetraVolumes;
  float* m_voxelVolumes;

}; 

#endif
