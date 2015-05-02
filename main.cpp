#include "defs.h"
#include "UGridTetrahedrizer.h"
#include "SGrid.h"
#include "VBOCreator.h"
#include "CudaRenderer.h"

#include <QApplication>

const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
//unsigned int timer = 0;
StopWatchInterface *timer = 0;


bool leftButton = false;
float lastX = 0.0f;
float lastY = 0.0f;
float rotX = 180.0f;
float rotY = 0.0f;

void mouse(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON){
    if (state == GLUT_DOWN){
      leftButton = true;
      std::cout << "left mouse button pressed" << std::endl;
    }
    else if(state == GLUT_UP){
      leftButton = false;
      std::cout << "left mouse button released" << std::endl;
    }
  }
}

void mouseMotion(int x, int y){   

  if(leftButton){
    rotX += static_cast<float>(y - lastY);
    rotY += static_cast<float>(x - lastX);
    if(rotX >= 360.0f)
      rotX -= 360.0f;
    if(rotY >= 360.0f)
      rotY -= 360.0f;
    //std::cout << "rotation x = " << rotX << " , rotation y = " << rotY << std::endl;
    //CudaRenderer::getInstance()->updateCamRotation(rotX,rotY);
  }
  
  lastX = x;
  lastY = y;
}

void passiveMouseMotion(int x, int y){
  lastX = x;
  lastY = y;

}


void camera (void) {
  CudaRenderer::getInstance()->updateCamRotation(rotX,rotY);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit) {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer)  / 1000.f);
        sprintf(fps, "Dark Matter PBVR: %3.1f fps (Max 100Hz)", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0; 
  	    fpsLimit = (int)MAX(ifps, 1.f);

        sdkResetTimer(&timer);
    }
 }



void timerEvent(int value){
  glutPostRedisplay();
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}


void display(void)
{

  sdkStartTimer(&timer);
  CudaRenderer::getInstance()->launchKernel(WINDOWWIDTH, WINDOWHEIGHT);


  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT |GL_ACCUM_BUFFER_BIT);

  float ratio = 1.0f;
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, VBOCreator::getInstance()->getPBOPixels());
  glBindTexture(GL_TEXTURE_2D, VBOCreator::getInstance()->getScreenMapTextureID());
  //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOWWIDTH*SUBPIXELLEVEL, WINDOWHEIGHT*SUBPIXELLEVEL, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOWWIDTH, WINDOWHEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glPushAttrib(GL_ENABLE_BIT|GL_DEPTH_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  camera();

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glBegin(GL_QUADS);
  glTexCoord2f(0.0, 1.0);   glVertex3f(-ratio,ratio,0.0f);				// Top Right
  glTexCoord2f(0.0, 0.0);   glVertex3f(-ratio,-ratio,0.0f);				// Top Left
  glTexCoord2f(1.0, 0.0);   glVertex3f(ratio,-ratio,0.0f);				// Bottom Left
  glTexCoord2f(1.0, 1.0);   glVertex3f(ratio,ratio,0.0f);				// Bottom Right
  glEnd();

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, 0);    

  glDisable(GL_BLEND);
  glDisable(GL_TEXTURE_2D);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glPopAttrib();
/*
  //test depth tex:
  GLuint depth;
  glGenTextures(1, &depth);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, VBOCreator::getInstance()->getPBOSubPixelsDepth());
  glBindTexture(GL_TEXTURE_2D, depth);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WINDOWWIDTH*SUBPIXELLEVEL,WINDOWHEIGHT*SUBPIXELLEVEL, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

  GLuint subPixels;
  glGenTextures(1, &subPixels);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, VBOCreator::getInstance()->getPBOSubPixels());
  glBindTexture(GL_TEXTURE_2D, subPixels);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOWWIDTH*SUBPIXELLEVEL,WINDOWHEIGHT*SUBPIXELLEVEL, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  // Filter and clamp
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE);
  glBindTexture(GL_TEXTURE_2D, 0);  
  glBegin(GL_POINTS);
  glEnd();
*/
  glutSwapBuffers();
  sdkStopTimer(&timer);
  computeFPS();
  glutReportErrors();

}


void resize(int w, int h) {

  // Prevent a divide by zero, when window is too short
  // (you cant make a window of zero width).
  if(h == 0)
    h = 1;
  float ratio = static_cast<float>(w) / static_cast<float>(h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(0, 0, w, h);
  gluPerspective(60,ratio,1,1000);
  //glOrtho(0,w,h,0,-1.1,1.1);
  glMatrixMode(GL_MODELVIEW);

}

void initGL(){
  glDisable(GL_DEPTH_TEST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glMatrixMode(GL_MODELVIEW);
  glClearColor(0,0,0,0);
  //glTranslatef(-58,-62,-104);
}

int main(int argc, char* argv[]){
  
  //need a gl context, creating a glut window is the easiest way:
  glutInit(&argc,argv);
  glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize (WINDOWWIDTH,WINDOWHEIGHT);   
  glutCreateWindow ("Test");
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutMouseFunc(mouse);
  glutMotionFunc(mouseMotion);
  glutPassiveMotionFunc(passiveMouseMotion);
  glutKeyboardFunc(NULL);
  glutIdleFunc(display);
  glutTimerFunc(REFRESH_DELAY, timerEvent,0);
  
  GLenum error = glewInit();
  if (GLEW_OK != error){
    fprintf(stderr, "Error: %s\n", glewGetErrorString(error));
  }
  
  //cudaGLSetGLDevice(0);


    UGridTetrahedrizer* tetra = UGridTetrahedrizer::getInstance();
    SGrid* sgrid = SGrid::getInstance();
    VBOCreator* vboCreator = VBOCreator::getInstance();
    bool isUnstuctured = true;
    sgrid->testFunc("here is the testcase");
#if TUMOR
  tetra->convert("appended.vtu");
  vboCreator->convertUGridToVBO();
  isUnstuctured = true;
#endif

#if DRAGON
  tetra->convert("dragon.vtu");
  vboCreator->convertUGridToVBO();
  isUnstuctured = true;
#endif

#if HEART
  tetra->convert("heart.vtu");
  vboCreator->convertUGridToVBO();
  isUnstuctured = true;
#endif
#if ISABEL
  sgrid->convert("darkmatterParicleOLD_VTK.vti");
  vboCreator->convertSGridToVBO();
  isUnstuctured = false;
#endif




  initGL();

  if(isUnstuctured)
      vboCreator->initVBOTetras(tetra->getOutput()->GetNumberOfCells());
  else
      vboCreator->initVBOVoxels(sgrid->getOutput()->GetNumberOfCells());

  vboCreator->initPBOSubPixels(WINDOWWIDTH,WINDOWHEIGHT);
  vboCreator->initPBOSubPixelsDepth(WINDOWWIDTH,WINDOWHEIGHT);
  vboCreator->initPBOPixels(WINDOWWIDTH,WINDOWHEIGHT);
  vboCreator->initTexture(WINDOWWIDTH,WINDOWHEIGHT);

  //CudaRenderer::getInstance()->initCuda();

  
  sdkCreateTimer(&timer);

  glutMainLoop();
  
  return 1;
}


