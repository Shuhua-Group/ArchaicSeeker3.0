from snakemake.utils import min_version
min_version("6.0")

configfile: "config/snakemake/config.yaml"

module simulation_1src_workflow:
    snakefile: "workflows/1src/simulation.snake"
    config: config

use rule * from simulation_1src_workflow as simulation_1src_*

module simulation_2src_workflow:
    snakefile: "workflows/2src/simulation.snake"
    config: config

use rule * from simulation_2src_workflow as simulation_2src_*

module skovhmm_workflow:
    snakefile: "workflows/1src/skovhmm.snake"
    config: config

use rule * from skovhmm_workflow as skovhmm_*

module sprime_1src_workflow:
    snakefile: "workflows/1src/sprime.snake"
    config: config

use rule * from sprime_1src_workflow as sprime_1src_*

module sstar_1src_workflow:
    snakefile: "workflows/1src/sstar.snake"
    config: config

use rule * from sstar_1src_workflow as sstar_1src_*

module archaicseeker2_workflow:
    snakefile: "workflows/2src/archaicseeker2.snake"
    config: config

use rule * from archaicseeker2_workflow as archaicseeker2_*

module sprime_2src_workflow:
    snakefile: "workflows/2src/sprime.snake"
    config: config

use rule * from sprime_2src_workflow as sprime_2src_*

module sstar_2src_workflow:
    snakefile: "workflows/2src/sstar.snake"
    config: config

use rule * from sstar_2src_workflow as sstar_2src_*

module plots_workflow:
    snakefile: "workflows/plots.snake"
    config: config

use rule * from plots_workflow as plots_*

rule all:
    input:
        rules.simulation_1src_all.input,
        rules.simulation_2src_all.input,
        rules.skovhmm_all.input,
        rules.sprime_1src_all.input,
        rules.sstar_1src_all.input,
        rules.archaicseeker2_all.input,
        rules.sprime_2src_all.input,
        rules.sstar_2src_all.input,
        rules.plots_all.input,
    default_target: True
