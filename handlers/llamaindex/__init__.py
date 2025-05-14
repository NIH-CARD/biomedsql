from llama_index.core.objects import SQLTableSchema

TABLE_SCHEMAS = [
    SQLTableSchema(
        table_name="AlzheimerDisease_CombinedGeneData_UUID",
        context_str=("Contains combined gene data for Alzheimer's Disease including columns such as UUID, SNP, "
                     "alleles (A1, A2), frequency, effect sizes (b, se), p-values, chromosome numbers and positions, "
                     "and nearestGene annotations.")
    ),
    SQLTableSchema(
        table_name="AlzheimerDisease_GeneAssoc_Pathways_UUID",
        context_str=("Contains Alzheimer\'s Disease-gene pathway association summary statistics including columns "
                     "such as UUID, genes, effect size, significance statistics, and p-values")
    ),
    SQLTableSchema(
        table_name="DrugGeneTargets_ComprehensiveAnnotations_updated",
        context_str=("Holds comprehensive annotations for drug gene targets. Columns include drug names, "
                     "chemblIdentifier, clinical trial phases, drug approval statuses, trade names, synonyms, "
                     "and linked disease information.")
    ),
    SQLTableSchema(
        table_name="DrugTargets_IndicationsAndTherapeuticUses",
        context_str=("Contains details on drug targets and their indications/therapeutic uses, including drug names, "
                     "trade names, drug types, action types, targets, trial phases, patent data, and related therapeutic "
                     "information.")
    ),
    SQLTableSchema(
        table_name="DrugTargets_LiscensingAndUses",
        context_str=("Contains details on drug targets and their different liscensing statuses and uses approved"
                     "by the FDA including drug names, route of administration, dosage, strength, and marketing status")
    ),
    SQLTableSchema(
        table_name="DrugTargets_UsesAndDosages",
        context_str=("Contains details on drug targets and their different FDA-approved dosages. Columns include"
                     "drug names, dosage forms, dosage routes, dosage strengths, and mechanism of action")
    ),
    SQLTableSchema(
        table_name="NeurodegenerativeDisease_AlleleFrequencies_UUID",
        context_str=("Contains allele frequency information from a neurodegenerative disease cohort inlcuding FTD, "
                     "LBD, and ALS patients. Columns include UUID, chromosome, SNP, A1, A2, and frequency")
    ),
    SQLTableSchema(
        table_name="NeurodegenerativeDiseases_SMR_Genes_Full",
        context_str=("Holds full gene data related to neurodegenerative diseases from SMR analyses. Includes gene "
                     "identifiers, probe IDs, GWAS/SMR statistics, allele information, and tissue-specific details.")
    ),
    SQLTableSchema(
        table_name="ParkinsonDisease_CompleteGeneData_No23andMe",
        context_str=("Contains complete gene data for Parkinson's Disease (excluding 23andMe data), featuring SNP details, "
                     "allele frequencies, effect sizes, and chromosome mapping information.")
    ),
    SQLTableSchema(
        table_name="ParkinsonDisease_GeneAssoc_Pathways_UUID",
        context_str=("Contains Parkinson's Disease-gene pathway association summary statistics including columns "
                     "such as UUID, genes, effect size, significance statistics, and p-values")
    )
]