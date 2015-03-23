#-
# ==========================================================================
# Copyright (c) 2011 Autodesk, Inc.
# All rights reserved.
# 
# These coded instructions, statements, and computer programs contain
# unpublished proprietary information written by Autodesk, Inc., and are
# protected by Federal copyright law. They may not be disclosed to third
# parties or copied or duplicated in any form, in whole or in part, without
# the prior written consent of Autodesk, Inc.
# ==========================================================================
#+

ifndef INCL_BUILDRULES

#
# Include platform specific build settings
#
include buildrules


#
# Always build the local plug-in when make is invoked from the
# directory.
#
all : plugins

endif

#
# Variable definitions
#

SRCDIR := .
DSTDIR := .
PKGDIR := $(HOME)/Documents/Packages

nicp3d_SOURCES  := nicp3d.cpp Mesh.cpp Deformable.cpp
nicp3d_OBJECTS  := nicp3d.o Mesh.o Deformable.o
nicp3d_PLUGIN   := $(DSTDIR)/nicp3d.$(EXT)
nicp3d_MAKEFILE := $(DSTDIR)/Makefile
nicp3d_EXTRA_INCLUDES := -I"$(PKGDIR)/eigen-eigen-10219c95fe65"
nicp3d_EXTRA_C++FLAGS := -std=c++11

#
# Include the optional per-plugin Makefile.inc
#
#    The file can contain macro definitions such as:
#       {pluginName}_EXTRA_CFLAGS
#       {pluginName}_EXTRA_C++FLAGS
#       {pluginName}_EXTRA_INCLUDES
#       {pluginName}_EXTRA_LIBS
-include $(SRCDIR)/Makefile.inc


#
# Set target specific flags.
#

$(nicp3d_OBJECTS): CFLAGS   := $(CFLAGS)   $(nicp3d_EXTRA_CFLAGS)
$(nicp3d_OBJECTS): C++FLAGS := $(C++FLAGS) $(nicp3d_EXTRA_C++FLAGS)
$(nicp3d_OBJECTS): INCLUDES := $(INCLUDES) $(nicp3d_EXTRA_INCLUDES)

depend_nicp3d:     INCLUDES := $(INCLUDES) $(nicp3d_EXTRA_INCLUDES)

$(nicp3d_PLUGIN):  LFLAGS   := $(LFLAGS) $(nicp3d_EXTRA_LFLAGS) 
$(nicp3d_PLUGIN):  LIBS     := $(LIBS)   -lOpenMaya -lFoundation $(nicp3d_EXTRA_LIBS) 

#
# Rules definitions
#

.PHONY: depend_nicp3d clean_nicp3d Clean_nicp3d


$(nicp3d_PLUGIN): $(nicp3d_OBJECTS) 
	-rm -f $@
	$(LD) -o $@ $(LFLAGS) $^ $(LIBS)

depend_nicp3d :
	makedepend $(INCLUDES) $(MDFLAGS) -f$(DSTDIR)/Makefile $(nicp3d_SOURCES)

clean_nicp3d:
	-rm -f $(nicp3d_OBJECTS)

Clean_nicp3d:
	-rm -f $(nicp3d_MAKEFILE).bak $(nicp3d_OBJECTS) $(nicp3d_PLUGIN)


plugins: $(nicp3d_PLUGIN)
depend:	 depend_nicp3d
clean:	 clean_nicp3d
Clean:	 Clean_nicp3d

