#include <stdlib.h>
#include <GL/glut.h> 	// these two header files are needed to compile a program in OpenGL with GLUT
#include <stdio.h>      // Header file for standard file i/o.
#include "utilfuncs.h"
#include "texcode.h"

#define ESCAPE 27

#define OWIDTH 720
#define OHEIGHT 480

void keyPressed(unsigned char key, int x, int y);

int mainwindow;
int i=0;
Image out;
long long tvar;
long long tvararr[256];
unsigned char tvarindex=0;
void display ()
{
	long long tvar2;
	//glLoadMatrixd(getPjMat());
	
	//tvar=nanotime();
	//printf("hi");
	glClear (GL_COLOR_BUFFER_BIT);  
	NextFrame();                                           

	//glColor(.5,.5,.5);
	glBegin(GL_QUADS);
		glTexCoord2f(0.f,0.f); glVertex3f(0.f		,0.f	,0);
		glTexCoord2f(1.f,0.f); glVertex3f(OWIDTH	,0.f	,0);
		glTexCoord2f(1.f,1.f); glVertex3f(OWIDTH	,OHEIGHT,0);
		glTexCoord2f(0.f,1.f); glVertex3f(0			,OHEIGHT,0);
	glEnd();
  	glBegin(GL_POINTS);
		glVertex3f(0.5f		,0.5f	,0.5f);
	glEnd();
  	//printf("hi");
  	
  	glFlush (); // makes sure commands are executed immediately
  	
	/*
	drawRose();
  	glFlush ();
  	*/
  	
	glutSwapBuffers();
	   
	glFinish();
	
}

/* The function called whenever a key is pressed. */
void keyPressed(unsigned char key, int x, int y) 
{
	int xx=0;
	int yy=0;
	int c=0;
	double avg=0;
	FILE *outp;
	
    /* avoid thrashing this procedure */
   //debug
   /*
   for(xx=0;xx<256;xx++){
   		for(yy=0;yy<7;yy++,xx++)
   			printf("%lf : ",((double)tvararr[xx])/2000000000);
   		printf("\n");
   }
   for(xx=0;xx<256;xx++)
		avg+=((double)tvararr[xx])/2000000000;
	printf("\naverage fps of processing:%lf\n",1/(avg/256));
	*/
   //debug/
	
    		
    /* If escape is pressed, kill everything. */
    if (key == 'd') 
	    { 
			/* shut down our window */
			free(out.data);
			glutDestroyWindow(mainwindow); 
		      /* exit the program...normal termination. */
			exit(0);                   
	    }
	if (key == 's') //save raw data
	    { 
		    out.data=malloc(OWIDTH*OHEIGHT*sizeof(char)*3);
		    glReadPixels(	0				,		//GLint x,
			     			0				,		//GLint y,
			     			OWIDTH			,		//GLsizei width,
			     			OHEIGHT			,		//GLsizei height,
			     			GL_BGRA			,		//GLenum format,
			     			GL_UNSIGNED_BYTE,		//GLenum type,
			     			out.data	);			//GLvoid *pixels
		     					
		    outp=fopen("outraw", "wb");

		    for(xx=0;xx<OHEIGHT*OWIDTH*3;xx++){
		      fprintf(outp,"%c",*(out.data+xx));
		    }
		    /* for(xx=0;xx<OWIDTH;xx++){ */
/* 			    for(yy=0;yy<OHEIGHT;yy++){ */
/* 				    for(c=0;c<3;c++){ */
/* 					    fprintf(outp,"%c",*(out.data+xx*OWIDTH*3+yy*3+c)); */
/* 				    } */
/* 			    } */
/* 		    } */
		    fclose(outp);
			/* shut down our window */
			free(out.data);
			glutDestroyWindow(mainwindow); 
		      /* exit the program...normal termination. */
			exit(0);                   
	    }
	glutDestroyWindow(mainwindow); 
      /* exit the program...normal termination. */
	exit(0); 
}

void init ()
{
	texinit();
	//LoadGLTextures();					// Load The Texture(s) 
    glEnable(GL_TEXTURE_2D);			// Enable Texture Mapping
   	
   	//glClearDepth(1.0);				// Enables Clearing Of The Depth Buffer
   	
    glShadeModel(GL_FLAT);				// Enables Smooth Color Shading

	glClearColor (0.0, 0.0, 0.0, 0.0); 	// sets background color to black                      
	
	
	 
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity ();
	// sets up basic scale for input for you to draw on
	//glOrtho (-1.0, 1, -1, 1, -10.0, 10.0);
	//glPushMatrix();					//push down stack to avoid overriding when we load next matrix
	glMatrixMode(GL_PROJECTION);
	//glLoadMatrixd(getPjMat());		//apply planar perspective transform
	//glLoadIdentity ();					//make sure OpenGL doesn't f*ck with my results
	glOrtho (.0, 720., 0, 480., -10.0, 10.0);
	out.width = OWIDTH;
	out.height = OHEIGHT;
	out.data = malloc(OWIDTH * OHEIGHT * 4);
}

int main(int argc, char** argv)
{
	glutInit (&argc, argv); // initialization      
	glutInitDisplayMode (GLUT_DOUBLE| GLUT_RGB); // certain settings 
	glutInitWindowSize (OWIDTH, OHEIGHT); // sets wize in pixels (horizontal and vertical)
	glutInitWindowPosition (10, 10); // upper left corner position
	mainwindow=glutCreateWindow ("Transform"); // sets the window title        
	init (); // calls the init() function above
	glutDisplayFunc (display); // uses the function called 'display' for displaying 
	glutIdleFunc(display);
	glutKeyboardFunc(&keyPressed);
	glutMainLoop (); // will execute OpenGL functions continuously if window left up
	return 0;
}


