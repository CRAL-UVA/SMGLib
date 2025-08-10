# Makefile for Social-ORCA

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -DFULL_LOG=1 -DFULL_OUTPUT=0 -DMAPF_LOG=0
INCLUDES = -Iinclude -Iexternal/tinyxml2

# Directories
SRCDIR = src
EXPDIR = src/experiments
BUILDDIR = build
MAPFDIR = src/mapf

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Source files
SOURCES = $(SRCDIR)/agent.cpp \
          $(SRCDIR)/agent_pnr.cpp \
          $(SRCDIR)/agent_pnr_ecbs.cpp \
          $(SRCDIR)/agent_returning.cpp \
          $(SRCDIR)/direct_planner.cpp \
          $(SRCDIR)/environment_options.cpp \
          $(SRCDIR)/geom.cpp \
          $(SRCDIR)/map.cpp \
          $(SRCDIR)/mapf_instances_logger.cpp \
          $(SRCDIR)/mission.cpp \
          $(SRCDIR)/orca_agent.cpp \
          $(SRCDIR)/orca_diff_drive_agent.cpp \
          $(SRCDIR)/sub_map.cpp \
          $(SRCDIR)/thetastar.cpp \
          $(SRCDIR)/xml_logger.cpp \
          $(SRCDIR)/xml_reader.cpp \
          external/tinyxml2/tinyxml2.cpp

# Find all MAPF source files
MAPF_SOURCES = $(wildcard $(MAPFDIR)/*.cpp)

# Combine all sources
ALL_SOURCES = $(SOURCES) $(MAPF_SOURCES)

# Object files
OBJECTS = $(ALL_SOURCES:%.cpp=$(BUILDDIR)/%.o)

# Main targets
all: $(BUILDDIR) $(BUILDDIR)/single_test $(BUILDDIR)/series_test

$(BUILDDIR)/single_test: $(OBJECTS) $(BUILDDIR)/$(EXPDIR)/single_test.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread

$(BUILDDIR)/series_test: $(OBJECTS) $(BUILDDIR)/$(EXPDIR)/series_test.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread

# Compile object files
$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILDDIR)

# Force rebuild
rebuild: clean all

.PHONY: all clean rebuild
