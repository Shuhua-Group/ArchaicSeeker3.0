# as3_exec="/share/home/linhuanyu/02_Software/ArchaicSeeker3_mem_update/AS3_dev/ArchaicSeeker3.1-mamba"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load ArchaicSeeker/3.0

seed=138991691
outdir="/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/${seed}"
cd ${outdir}
/share/home/linhuanyu/02_Software/ArchaicSeeker3_mem_update/AS3_dev/ArchaicSeeker3.1-mamba -t ${outdir}/target.vcf.gz -r ${outdir}/ref.vcf.gz -m ${outdir}/ref.map -o ${outdir} --merge 5000





