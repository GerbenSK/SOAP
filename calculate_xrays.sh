#!/bin/bash

num_cores=16
# snapnum=68
box=L100_m9_JETS
basename=/net/hypernova/data2/student_projects/stemker/swift/source/examples/FLAMINGO
xray_tables=/net/hypernova/data2/FLAMINGO/Tables/Xray/X_Ray_table_combined.hdf5

for snapnum in 48 58 68;
do
    for fraction in 0.1 0.2 0.5; 
    do
        file_dir=${basename}/${box}_frac_${fraction}/flamingo_00${snapnum}.hdf5
        out_dir=${basename}/${box}_frac_${fraction}/recalculated_xrays/xray_00${snapnum}.hdf5
        mpirun -np $num_cores python3 misc/recalculate_xrays.py $file_dir $out_dir $xray_tables --snap-nr $snapnum
    done
done

# file_dir=${basename}/${box}_frac_0.0/snapshots/flamingo_00${snapnum}.hdf5
# out_dir=${basename}/${box}_frac_0.0/recalculated_xrays/xray_00${snapnum}.hdf5
# mpirun -np $num_cores python3 misc/recalculate_xrays.py $file_dir $out_dir $xray_tables --snap-nr $snapnum