#!/bin/bash
gpu=mi300a
vendor=rocsolver
outdir=batch_svd_sweep_${gpu}

mkdir -p ${outdir}

# batch sizes
# values used in the paper: smallsq = 10000, and 1000 otherwise
# change as needed
batch_smallsq=10
batch=10

# small square dimensions
mstart_smallsq=2
mstop_smallsq=32
mstep_smallsq=2

# square dimensions
mstart_sq=50
mstop_sq=1000
mstep_sq=50

# tall-skinny dimensions
mstart_ts=100
mstop_ts=2000
mstep_ts=100
n_ts=(16 50 100)

# test options for all runs
OPTS="--align 1 --jobu s --jobv s --niter 10 -c -l"

# main loop over precision
for p in s d c z 
do
	# magma -- small-square
	ofile=magma_${p}gesvj_smallsq_batch_${batch_smallsq}_${gpu}.txt
	out=${outdir}/${ofile}
        echo ${ofile}
	./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch_smallsq} -N ${mstart_smallsq}:${mstop_smallsq}:${mstep_smallsq} --version 1 > ${out}

	# cusolver -- small-square
	ofile=${vendor}_${p}gesvdj_smallsq_batch_${batch_smallsq}_${gpu}.txt
	out=${outdir}/${ofile}
        echo ${ofile}
	./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch_smallsq} -N ${mstart_smallsq}:${mstop_smallsq}:${mstep_smallsq} --version 2 > ${out}

	# magma -- square
	ofile=magma_${p}gesvj_sq_batch_${batch}_${gpu}.txt
        out=${outdir}/${ofile}
        echo ${ofile}
        ./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch} -N ${mstart_sq}:${mstop_sq}:${mstep_sq} --version 5 > ${out}

        # cusolver -- square
	ofile=${vendor}_${p}gesvdj_sq_batch_${batch}_${gpu}.txt
        out=${outdir}/${ofile}
        echo ${ofile}
        ./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch} -N ${mstart_sq}:${mstop_sq}:${mstep_sq} --version 2 > ${out}

	# tall-skinny
	for n in "${n_ts[@]}"; do
		ofile=magma_${p}gesvj_ts_n_${n}_batch_${batch}_${gpu}.txt
		out=${outdir}/${ofile}
		echo ${ofile}
		./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch} -N ${mstart_ts}:${mstop_ts}:${mstep_ts},${n} --version 5 > ${out}

		ofile=${vendor}_${p}gesvdj_ts_n_${n}_batch_${batch}_${gpu}.txt
		out=${outdir}/${ofile}
		echo ${ofile}
		./testing/testing_${p}gesvj_batched_v2 ${OPTS} --batch ${batch} -N ${mstart_ts}:${mstop_ts}:${mstep_ts},${n} --version 2 > ${out}
	done
done

