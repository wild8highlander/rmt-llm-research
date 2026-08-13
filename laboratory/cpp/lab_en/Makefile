# Makefile — RMT-LLM Laboratory C++ (English)
# Build: make
# Run:   ./rmt_llm_lab_en

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
TARGET = rmt_llm_lab_en

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
