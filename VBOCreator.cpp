#include "VBOCreator.h"

VBOCreator::VBOCreator(){
  m_className = "VBOCreator";
  m_vboTetraID = 0;

  m_pboSubPixelID = 0;
  m_numTetras = 0;
  m_screenMapTextureID = 0;
}

VBOCreator::~VBOCreator(){

}

//generates a VBO for the tetra data:
void VBOCreator::initVBOTetras(unsigned int numTetras){
  m_numTetras = numTetras;
  
  glGenBuffersARB(1, &m_vboTetraID);
    
  size_t sizeTetra = sizeof(m_tetras[0]);
  #if DEBUG_VBOCREATOR
    std::cout << m_className << " VBO ID tetras = " << m_vboTetraID << std::endl;
    std::cout << m_className << " size of tetra = " << sizeTetra << std::endl;
    std::cout << m_className << " size of array = " << static_cast<double>(numTetras*sizeTetra)/(1024*1024) << " mb"  << std::endl;
  #endif

  glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB,m_vboTetraID);
  glBufferDataARB(GL_ARRAY_BUFFER_ARB,numTetras*sizeTetra,m_tetras,GL_STATIC_DRAW_ARB);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
  
  //map to cuda:
  //deprecated: cutilSafeCall(cudaGLRegisterBufferObject(m_vboTetraID));

//checkCudaErrors(m_vboTetraID);
}

//generates a VBO for the voxel data:
void VBOCreator::initVBOVoxels(unsigned int numVoxels){
  m_numVoxels = numVoxels;

  glGenBuffersARB(1, &m_vboVoxelID);

  size_t sizeVoxel = sizeof(m_voxels[0]);
  #if DEBUG_VBOCREATOR
    std::cout << m_className << " VBO ID tetras = " << m_vboVoxelID << std::endl;
    std::cout << m_className << " size of tetra = " << sizeVoxel << std::endl;
    std::cout << m_className << " size of array = " << static_cast<double>(numVoxels*sizeVoxel)/(1024*1024) << " mb"  << std::endl;
  #endif

  glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB,m_vboVoxelID);
  glBufferDataARB(GL_ARRAY_BUFFER_ARB,numVoxels*sizeVoxel,m_voxels,GL_STATIC_DRAW_ARB);
  glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);

  //map to cuda:
  //deprecated: cutilSafeCall(cudaGLRegisterBufferObject(m_vboTetraID));

//checkCudaErrors(m_vboTetraID);
}


//generates an empty PBO for subpixels:
void VBOCreator::initPBOSubPixels(unsigned int windowWidth, unsigned int windowHeight){
  unsigned int windowSize = windowWidth * windowHeight;
  unsigned int numSubPixels = SUBPIXELLEVEL*SUBPIXELLEVEL;
  glGenBuffersARB(1, &m_pboSubPixelID);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, m_pboSubPixelID);
  glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, windowSize*numSubPixels*4,0,GL_STREAM_DRAW_ARB);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);


  #if DEBUG_VBOCREATOR
    std::cout << m_className << " PBO ID subPixels = " << m_pboSubPixelID << std::endl;
    std::cout << m_className << " num subPixels    = " << numSubPixels << std::endl;
    std::cout << m_className << " num pixels total = " << static_cast<double>(windowSize*numSubPixels)/(1000) << " k" << std::endl;
  #endif
}

//generates the depth buffer for the subpixels:
void VBOCreator::initPBOSubPixelsDepth(unsigned int windowWidth, unsigned int windowHeight){
  unsigned int windowSize = windowWidth * windowHeight;
  unsigned int numSubPixels = SUBPIXELLEVEL*SUBPIXELLEVEL;
  glGenBuffersARB(1, &m_pboSubPixelDepthID);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, m_pboSubPixelDepthID);
  glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, windowSize*numSubPixels*4,0,GL_STREAM_DRAW_ARB);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);


  #if DEBUG_VBOCREATOR
    std::cout << m_className << " PBO ID depth     = " << m_pboSubPixelID << std::endl;
    std::cout << m_className << " num subPixels    = " << numSubPixels << std::endl;
    std::cout << m_className << " num pixels total = " << static_cast<double>(windowSize*numSubPixels)/(1000) << " k" << std::endl;
  #endif
}

//pbo for final pixels to be displayed
void VBOCreator::initPBOPixels(unsigned int windowWidth, unsigned int windowHeight){
  unsigned int windowSize = windowWidth * windowHeight;
  glGenBuffersARB(1, &m_pboPixelID);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, m_pboPixelID);
  glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, windowSize*4,0,GL_STREAM_DRAW_ARB);
  glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);


#if DEBUG_VBOCREATOR
  std::cout << m_className << " PBO ID pixels     = " << m_pboSubPixelID << std::endl;
  std::cout << m_className << " num pixels total = " << static_cast<double>(windowSize)/(1000) << " k" << std::endl;
#endif
}

//generates a texture for displaying results:
void VBOCreator::initTexture(unsigned int windowWidth, unsigned int windowHeight){
  glActiveTexture(GL_TEXTURE0);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &m_screenMapTextureID);
  glBindTexture(GL_TEXTURE_2D, m_screenMapTextureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_ALPHA);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  

}

//reads the cells from the grid and converts then to our vbo structs
void VBOCreator::convertSGridToVBO(){

    SGrid* sg = SGrid::getInstance();
    if(!sg->getOutput()){
        std::cout << m_className << "No grid to convert to VBO struct. Aborting" << std::endl;
        return;
    }

    size_t voxelSize = sizeof(voxel<float>);

    vtkSmartPointer<vtkUnstructuredGrid> voxelGrid = sg->getOutput();
    unsigned int numVoxels = voxelGrid->GetNumberOfCells();

    //get the scalar value
    vtkPointData* pointData = voxelGrid->GetPointData();

#if ISABEL
    vtkDataArray* scalars= pointData->GetArray("ImageFile");
#endif
    std::cout << m_className << "----------------Debug Info---------------" << std::endl;
   // std::cout << m_className << " Number of scalars = " << scalars->GetNumberOfTuples()   << std::endl;
    std::cout << m_className << " Number of tuples = " << pointData->GetNumberOfTuples()   << std::endl;
    std::cout << m_className << " Number of points = " << voxelGrid->GetNumberOfPoints()   << std::endl;


    m_voxels = new voxelFloat[numVoxels];
    m_voxelVolumes = new float[numVoxels];

    for(unsigned int i = 0 ; i < numVoxels ; i++){
      voxelFloat currentVoxel = cellToVoxel(i,voxelGrid);
      m_voxels[i] = currentVoxel;
      m_voxelVolumes[i] = 1.0f;
    }

    std::cout << m_className << " sample vertex: " << m_voxels[0].v[0].pos[0] << " " << m_voxels[0].v[0].pos[1] << " " << m_voxels[0].v[0].pos[2] << std::endl;


    m_numVoxels = numVoxels;


    std::cout << m_className << " num voxels = " << numVoxels << std::endl;






}

//reads the cells from the grid and converts them to our vbo structs:
void VBOCreator::convertUGridToVBO(){
  
  UGridTetrahedrizer* tetrahedrizer = UGridTetrahedrizer::getInstance();
  if(!tetrahedrizer->getOutput()){
    std::cout << m_className << " No grid to convertUGridToVBO. Aborting" << std::endl;
    return;
  }
  
  size_t tetraSize = sizeof(tetra<float>);

  //now convertUGridToVBO each cell into a struct:
  vtkSmartPointer<vtkUnstructuredGrid> tetraGrid = tetrahedrizer->getOutput();
  unsigned int numTetras = tetraGrid->GetNumberOfCells();

  //get the scalars:
  vtkPointData* pointData = tetraGrid->GetPointData();
#if TUMOR
  vtkDataArray* scalars = pointData->GetArray("dead");
#endif

#if DRAGON
  vtkDataArray* scalars = pointData->GetArray("SplatterValues");
#endif

#if HEART
  vtkDataArray* scalars = pointData->GetArray("scalars");
#endif
    
  #if DEBUG_VBOCREATOR
    std::cout << m_className << " Number of scalars = " << scalars->GetNumberOfTuples()   << std::endl;
    std::cout << m_className << " Number of tuples = " << pointData->GetNumberOfTuples()   << std::endl;
    std::cout << m_className << " Number of points = " << tetraGrid->GetNumberOfPoints()   << std::endl;
  #endif



  m_tetras = new tetraFloat[numTetras];
  m_tetraVolumes = new float[numTetras];

  for(unsigned int i = 0 ; i < numTetras ; i++){
    tetraFloat currentTetra = cellToTetra(i,tetraGrid);
    m_tetras[i] = currentTetra;
    m_tetraVolumes[i] = cellVolume(currentTetra);
  }
  
  std::cout << m_className << " sample vertex: " << m_tetras[0].v[0].pos[0] << " " << m_tetras[0].v[0].pos[1] << " " << m_tetras[0].v[0].pos[2] << std::endl;


  m_numTetras = numTetras;
  
  #if DEBUG_VBOCREATOR
    std::cout << m_className << " num tetras = " << numTetras << std::endl;
  #endif

  

}

//convertUGridToVBO datastructure from vtkCell to tetra for VBO
voxelFloat VBOCreator::cellToVoxel(unsigned int cellID, vtkUnstructuredGrid* grid){
  voxelFloat currentVoxel;

  vtkCell* cell = grid->GetCell(cellID);
  vtkIdList* pointIDs = cell->GetPointIds();


  for(int i = 0 ; i < 8 ; i++){
    vertex<float> currentPoint;
    unsigned int currentID = pointIDs->GetId(i);
    currentPoint.pos[0] = grid->GetPoint(currentID)[0];
    currentPoint.pos[1] = grid->GetPoint(currentID)[1];
    currentPoint.pos[2] = grid->GetPoint(currentID)[2];
    //currentPoint.pos[3] = 1.0f;
#if ISABEL
    currentPoint.scalar = grid->GetPointData()->GetScalars("ImageFile")->GetTuple1(currentID);
#endif

    currentVoxel.v[i] = currentPoint;
  }
  return currentVoxel;
}


//convertUGridToVBO datastructure from vtkCell to tetra for VBO
tetraFloat VBOCreator::cellToTetra(unsigned int cellID, vtkUnstructuredGrid* grid){
  tetraFloat currentTetra;

  vtkCell* cell = grid->GetCell(cellID);
  vtkIdList* pointIDs = cell->GetPointIds();
  
  
  for(int i = 0 ; i < 4 ; i++){
    vertex<float> currentPoint;
    unsigned int currentID = pointIDs->GetId(i);
    currentPoint.pos[0] = grid->GetPoint(currentID)[0];
    currentPoint.pos[1] = grid->GetPoint(currentID)[1];
    currentPoint.pos[2] = grid->GetPoint(currentID)[2];
    //currentPoint.pos[3] = 1.0f;
#if TUMOR
    currentPoint.scalar = grid->GetPointData()->GetScalars("dead")->GetTuple1(currentID);
#endif

#if DRAGON
    currentPoint.scalar = grid->GetPointData()->GetScalars("SplatterValues")->GetTuple1(currentID);
#endif

#if HEART
    currentPoint.scalar = grid->GetPointData()->GetScalars("scalars")->GetTuple1(currentID);
    currentPoint.var = grid->GetPointData()->GetScalars("variance") -> GetTuple1(currentID);
#endif
    currentTetra.v[i] = currentPoint;
  }
  return currentTetra;
}

//calculate volume of single cell
float VBOCreator::cellVolume(tetraFloat tetra){
  vertexFloat a = tetra.v[0];
  vertexFloat b = tetra.v[1];
  vertexFloat c = tetra.v[2];
  vertexFloat d = tetra.v[3];
  float tmp1[3];
  float tmp2[3];
  float tmp3[3];

  for(int i = 0 ; i < 3 ; i++){
    tmp1[i] = a.pos[i] - d.pos[i];
    tmp2[i] = b.pos[i] - d.pos[i];
    tmp3[i] = c.pos[i] - d.pos[i];
  }
  

  //tmp2 x tmp3
  double tmp4[3];
  tmp4[0] = tmp2[1] * tmp3[2] - tmp2[2] * tmp3[1];
  tmp4[1] = tmp2[2] * tmp3[0] - tmp2[0] * tmp3[2];
  tmp4[2] = tmp2[0] * tmp3[1] - tmp2[1] * tmp3[0];

  //tmp1 . tmp4
  float dot = tmp1[0] * tmp4[0] + tmp1[1] * tmp4[1] + tmp1[2] * tmp4[2];
  if( dot < 0.0f)
    dot *= -1;
  return (dot / 6.0f);
}
