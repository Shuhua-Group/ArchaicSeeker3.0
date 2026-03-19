for demo in AncientEurasia AS2_HumanNeanderthalDenisovan BonoboGhost ChimpBonoboGhost HumanArchaic HumanNeanderthal HumanNeanderthalDenisovan OOANeanderthal Skov_HumanDenisovan
do
    find /home/linhuanyu/share1/20_AS3/results/inference/ArchaicSeeker3.0/${demo}/nref_50/ntgt_10/ \
      -type f -name "AS3_1208.accuracy" \
      -exec cat {} + \
      > /home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/3_Summary/5_Demo/${demo}.accuracy
done





















