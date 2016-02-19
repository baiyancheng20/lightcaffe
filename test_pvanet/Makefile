all: 
	g++ test_pvanet.cpp -MMD -MP -pthread -fPIC -DNDEBUG -O2 -Wall -Wno-sign-compare -c -o test_pvanet.o
	g++ test_pvanet.o -o test_pvanet -pthread -fPIC -DNDEBUG -O2 -Wall -Wno-sign-compare -L./ -lpvanet -lopencv_core -lopencv_highgui -lopencv_imgproc \
		-Wl,-rpath,\./

clean:
	rm test_pvanet.o
	rm test_pvanet.d
	rm test_pvanet
