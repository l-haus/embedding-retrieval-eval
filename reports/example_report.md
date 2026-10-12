# Retrieval Evaluation Report

## Config

| setting | value |
|---|---|
| embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| top_k | 10 |
| candidate_depth | 100 |
| similarity | cosine |
| documents | `data/scifact/documents.jsonl` |
| queries | `data/scifact/queries.jsonl` |
| labels | `data/scifact/labels.jsonl` |

## Inputs

| input | count | truncated |
|---|---|---|
| documents | 5183 | 3681 (71.0%) |
| queries | 300 | 0 (0.0%) |

Text past the model's input limit is dropped before embedding, so a truncated document is scored on its opening only.

## Metrics

| metric | value |
|---|---|
| k | 10 |
| num_queries | 300 |
| num_queries_evaluated | 300 |
| precision_at_k | 0.0883 |
| recall_at_k | 0.7833 |
| mrr | 0.6047 |
| ndcg_at_k | 0.6451 |

## Best and worst queries

Ranked by ndcg_at_k.

### Best

| query_id | query | precision@k | recall@k | RR | nDCG@k |
|---|---|---|---|---|---|
| `1012` | Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume. | 0.1000 | 1.0000 | 1.0000 | 1.0000 |
| `1014` | Rapamycin decreases the concentration of triacylglycerols in fruit flies. | 0.1000 | 1.0000 | 1.0000 | 1.0000 |
| `1086` | Sildenafil improves erectile function in men who experience sexual dysfunction as a result of the use of SSRI antidepressants. | 0.1000 | 1.0000 | 1.0000 | 1.0000 |
| `1088` | Silencing of Bcl2 is important for the maintenance and progression of tumors. | 0.1000 | 1.0000 | 1.0000 | 1.0000 |
| `1089` | Smc5/6 engagment drives the activation of SUMO E3 ligase Mms21 by ATP-dependent remolding. | 0.1000 | 1.0000 | 1.0000 | 1.0000 |

### Worst

| query_id | query | precision@k | recall@k | RR | nDCG@k |
|---|---|---|---|---|---|
| `975` | Primary pro-inflammatory cytokines induce secondary pro- and anti-inflammatory mediators. | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `887` | Only a minority of cells survive development after differentiation into stress-resistant spores. | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `870` | Obesity decreases life quality. | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `834` | NOX2-independent pathways can generate peroxynitrite by reacting with nitrogen intermediates. | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `831` | NF2 (Merlin) prevents phosphorylation and subsequent cytoplasmic sequestration of YAP in Drosophila. | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Failure cases

62 of 300 labelled queries did not retrieve a relevant document within top-k:

- 40 ranked below the top 10 but within the top 100 (`relevant_ranked_low`)
- 22 not within the top 100 at all (`no_relevant_retrieved`)

| query_id | query | category | retrieved | missed |
|---|---|---|---|---|
| `13` | 5% of perinatal mortality is due to low birth weight. | relevant_ranked_low | `7662395`, `4791384`, `1263446`, `11748341`, `8529693`, `23557241`, `8842332`, `17450673`, `27123743`, `26611834` | `1606628` |
| `48` | A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection. | no_relevant_retrieved | `18617259`, `11349166`, `24512064`, `5850219`, `44672703`, `29955650`, `21644993`, `43587663`, `13322804`, `27063470` | `13734012` |
| `70` | Activation of PPM1D suppresses p53 function. | relevant_ranked_low | `9483851`, `27949347`, `11903247`, `6896063`, `28107602`, `31624828`, `9169645`, `21557055`, `37204802`, `14460402` | `4414547`, `5956380` |
| `128` | Arterioles have a larger lumen diameter than venules. | relevant_ranked_low | `22867765`, `79447`, `87758`, `2593298`, `243694`, `12549585`, `2056197`, `7850867`, `24974080`, `23513718` | `8290953` |
| `132` | Aspirin inhibits the production of PGE2. | no_relevant_retrieved | `6945691`, `15648443`, `21181273`, `7613033`, `2443495`, `9159125`, `3866315`, `5704562`, `12513042`, `20732789` | `7975937` |
| `198` | CCL19 is absent within dLNs. | relevant_ranked_low | `29347970`, `17934603`, `3462075`, `15795880`, `8538916`, `16427454`, `26030079`, `29366489`, `6501747`, `2236768` | `2177022` |
| `239` | Cellular aging closely links to an older appearance. | relevant_ranked_low | `16693950`, `22264117`, `6040392`, `7514614`, `9291668`, `9559146`, `516867`, `6599693`, `2466614`, `3230557` | `14079881` |
| `294` | Crossover hot spots are not found within gene promoters in Saccharomyces cerevisiae. | relevant_ranked_low | `17240457`, `406733`, `35861290`, `2817000`, `43156471`, `11420613`, `5116145`, `23664875`, `27134931`, `3367829` | `10874408` |
| `303` | DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region. | no_relevant_retrieved | `5278233`, `4085204`, `1780819`, `9791313`, `39481265`, `2722988`, `14581009`, `15635366`, `13639330`, `3346812` | `4388470` |
| `312` | De novo assembly of sequence data has more specific contigs than unassembled sequence data. | no_relevant_retrieved | `14464451`, `32770503`, `3662132`, `13123189`, `2107238`, `17447653`, `5650232`, `13618987`, `15966318`, `7324039` | `6173523` |
| `314` | Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome. | relevant_ranked_low | `2638387`, `544971`, `20996244`, `25373397`, `37768883`, `28904104`, `13221399`, `15319019`, `29214508`, `25014337` | `4347374` |
| `384` | Epidemiological disease burden from noncommunicable diseases is more prevalent in low economic settings. | relevant_ranked_low | `51972698`, `25691878`, `8582337`, `15670968`, `12428497`, `22194407`, `13914633`, `3716075`, `39984099`, `8529693` | `13770184` |
| `415` | Female carriers of the Apolipoprotein E4 (APOE4) allele have increased risk for dementia. | relevant_ranked_low | `20937018`, `18852643`, `4709641`, `37205759`, `51386222`, `24530130`, `30351165`, `12443371`, `6917133`, `24906548` | `6309659` |
| `421` | Flexible molecules experience greater steric hindrance in the tumor microenviroment than rigid molecules. | no_relevant_retrieved | `1065627`, `12866641`, `25435456`, `17388232`, `6944800`, `927561`, `26231129`, `43224840`, `4346436`, `10812605` | `11172205` |
| `431` | FoxO3a activation in neuronal death is mediated by reactive oxygen species (ROS). | relevant_ranked_low | `14496749`, `2543135`, `16728949`, `19922508`, `29429111`, `13380011`, `23052989`, `25251625`, `4795303`, `23887844` | `28937856` |
| `437` | Functional consequences of genomic alterations due to Myelodysplastic syndrome (MDS) are poorly understood due to the lack of an animal model. | no_relevant_retrieved | `2359152`, `3391547`, `24466904`, `5782614`, `5765455`, `8083310`, `5836`, `39851630`, `8385277`, `6000423` | `18399038` |
| `475` | Glycolysis is one of the primary glycometabolic pathways in cells. | relevant_ranked_low | `23576165`, `195689757`, `18810195`, `3468459`, `20672596`, `7548577`, `3831884`, `8724666`, `13466622`, `29473081` | `18678095` |
| `502` | Healthcare delivery efficiency in crowded delivery centers is impaired by improving structural, logistical, and interpersonal elements. | no_relevant_retrieved | `38180456`, `3222122`, `40905302`, `17626822`, `6421734`, `143796742`, `21377587`, `16390264`, `21203899`, `39059143` | `13071728` |
| `508` | Hematopoietic Stem Cell purification reaches purity rate of up to 50%. | relevant_ranked_low | `25516011`, `25985964`, `14550841`, `34982259`, `92499`, `4381486`, `623486`, `4417177`, `18374364`, `7583161` | `13980338` |
| `517` | High levels of copeptin decrease risk of diabetes. | relevant_ranked_low | `19658917`, `25135304`, `23017040`, `20288322`, `11481946`, `38028419`, `42800527`, `2158500`, `25028913`, `4353857` | `15663829` |

Showing 20 of 62; the full set is in `failure_cases.jsonl`.

## Limitations

These numbers describe one configuration over one labelled dataset.
Before drawing conclusions from them, read `docs/limitations.md` —
in particular that unlabelled retrieved documents count as
non-relevant, that no significance testing is performed, and that
text is used exactly as provided with no normalization.
