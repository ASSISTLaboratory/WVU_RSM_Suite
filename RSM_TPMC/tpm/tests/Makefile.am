TESTS = tpm_test.sh
EXTRA_DIST = Cube.stl tpm_test.sh

tpm_test.sh:
	cp -r ${srcdir}/tempfiles .
	mkdir -p Outputs/Projected_Area
	echo "#!/bin/bash" > tpm_test.sh
	echo "mpirun -np 2 ${srcdir}/../src/tpm" >> tpm_test.sh
	cat ./tpm_test.sh

