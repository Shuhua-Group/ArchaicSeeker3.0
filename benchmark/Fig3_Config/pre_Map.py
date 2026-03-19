import pandas as pd

# df = pd.read_csv("/home/linhuanyu/share1/0_PublicData/3_GeneticMap/Global/GeneticMap/genetic_map_chr19.txt", sep=" ", header=0)
# df['chr'] = 'chr19'
# df = df[['chr', 'position', 'rate', 'GeneticMap']]
# df.to_csv("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/401.Fig2/Config/WithMap/chr19.b38.gmap.txt", sep="\t", index=False, header=False)

df = pd.read_csv("/home/linhuanyu/share1/0_PublicData/3_GeneticMap/Global/plink_38_map/plink.chr19.GRCh38.map",sep = ' ',header = None)
df[0] = 1
df.to_csv("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/0_Config/chr19.b38.map.as3.txt",sep = '\t',index = False,header = False)
print(df.head)