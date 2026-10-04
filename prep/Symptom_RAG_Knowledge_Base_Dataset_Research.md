# Symptom RAG Analyser --- Knowledge-Base Dataset Research

**Research date:** 22 September 2026\
**Purpose:** Select reliable, useful, and technically practical
biomedical data/resources for enriching the Symptom RAG Analyser before
changing the ingestion or retrieval pipeline.

------------------------------------------------------------------------

## 1. Executive Summary

Our current knowledge base is centered on the public
`dux-tecblic/symptom-disease-dataset`. It is useful as a **symptom →
disease association layer**, but it is not sufficient as the only
biomedical knowledge source.

The main limitation is not simply the number of records. The dataset
does not provide the depth of evidence needed for a clinical RAG system:
research evidence, detailed disease descriptions, diagnostic
information, medication safety, clinical guidelines, and standardized
medical terminology are largely missing.

The recommended knowledge-base strategy is therefore **layered**:

1.  **Keep the existing symptom-disease dataset** as the symptom
    association layer.
2.  Add **PMC Open Access full-text biomedical literature** for detailed
    evidence.
3.  Use **PubMed and Europe PMC** as literature discovery/indexing
    infrastructure, with Europe PMC especially useful for programmatic
    retrieval and open-access full text.
4.  Add **MedlinePlus** for curated disease, symptom, medical-test, and
    patient-oriented health information.
5.  Add a **targeted WHO guideline layer** for authoritative
    clinical/public-health guidance.
6.  Add **DailyMed/openFDA** for medication indications,
    contraindications, warnings, adverse reactions, and drug-label
    information.
7.  Use **MeSH, HPO, and RxNorm as terminology/normalization layers**,
    rather than treating them as ordinary document corpora.
8.  Keep **BioASQ primarily for evaluation**, not as the main medical
    knowledge corpus.
9.  Do not make **MIMIC-IV/MIMIC-IV-Note** a first-stage dependency
    because access requires credentialing, training, and a data-use
    agreement.

### Recommended first-stage stack

``` text
                 SYMPTOM RAG KNOWLEDGE BASE
                              |
        +---------------------+----------------------+
        |                     |                      |
        v                     v                      v
 Symptom-Disease       Biomedical Literature    Medication
    Dataset              PMC / PubMed /         DailyMed /
                         Europe PMC              openFDA
        |                     |                      |
        +---------------------+----------------------+
                              |
                 Medical Terminology Layer
                  MeSH / HPO / RxNorm
                              |
                              v
                           Qdrant
                              |
                              v
                     Retrieval + RAG
```

The goal is not to claim that this will "predict diagnoses clearly." The
goal is to improve **evidence retrieval, grounding, coverage,
provenance, and answer quality**. A clinical decision-support system
should not present retrieved evidence as a definitive diagnosis.

------------------------------------------------------------------------

# 2. Current Baseline: dux-tecblic/symptom-disease-dataset

## 2.1 Source

**Dataset:** `dux-tecblic/symptom-disease-dataset`\
**Host:** Hugging Face\
**URL:**
https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset

The dataset is a public machine-learning dataset for text
classification. It contains symptom text and disease labels and provides
a `mapping.json` file that maps numerical labels to disease names.

### What it is good for

It gives the system a direct starting relationship:

``` text
Symptoms / symptom description
             |
             v
        Disease label
```

This matches the core use case of our Symptom RAG Analyser.

## 2.2 Our actual inspected copy

Our local downloaded training CSV contained:

  Property                                 Observed value
  -------------------------------------- ----------------
  Raw training rows                                 5,634
  Unique `(text, label)` pairs                      1,993
  Exact duplicate rows                              3,641
  Disease IDs used by training CSV                    866
  Disease IDs in `mapping.json`                     1,082
  Empty symptom-text rows                               0
  CSV labels missing from mapping                       0
  Documents produced by our loader                  1,993
  Chunks produced with current chunker              2,690

This distinction is important:

> **5,634 raw rows do not represent 5,634 unique pieces of knowledge.**

A large portion of the training CSV consists of exact duplicates.

## 2.3 Suitability

**Suitability: HIGH as a baseline symptom-disease layer; LOW as a
complete clinical knowledge base.**

### Strengths

-   Directly relevant to symptom-based retrieval.
-   Simple structure.
-   Disease labels are available.
-   Useful for testing the end-to-end RAG pipeline.
-   Easy to ingest and embed.
-   Publicly accessible.
-   Provides natural-language and structured symptom descriptions.

### Limitations

-   The repository does not provide enough documentation to establish
    clinical validation or expert annotation methodology.
-   Significant exact duplication exists in the training data.
-   It is primarily a classification dataset, not a clinical evidence
    repository.
-   It does not provide detailed diagnostic reasoning.
-   It does not provide comprehensive medical-test information.
-   It does not provide authoritative treatment guidance.
-   It does not adequately cover medication safety/interactions.
-   The dataset should not be described as clinical ground truth.

### Decision

**KEEP.**

Use it as:

> **Symptom--Disease Association Layer**

Do not use it as the sole source for clinical evidence.

------------------------------------------------------------------------

# 3. PubMed

**Official source:** https://pubmed.ncbi.nlm.nih.gov/

PubMed is NLM's major biomedical literature search resource. The PubMed
homepage currently reports more than 40 million citations covering
biomedical literature from MEDLINE, life-science journals, and online
books.

PubMed also provides NCBI E-utilities for programmatic access.

## 3.1 What PubMed provides

Typical records can contain:

-   PMID
-   title
-   abstract
-   authors
-   journal
-   publication date
-   MeSH terms
-   links to available full text
-   other bibliographic metadata

## 3.2 Suitability for our analyser

**Suitability: VERY HIGH for literature discovery and evidence
metadata.**

PubMed is particularly useful for finding evidence relevant to:

``` text
Disease
Symptoms
Diagnosis
Treatment
Risk factors
Drug effects
Clinical findings
```

## 3.3 Limitation

PubMed is primarily an indexing/search resource.

Not every PubMed record gives us reusable full-text content.

Therefore:

> We should not treat PubMed as a giant downloadable full-text corpus.

Instead, use it to **discover and filter relevant biomedical
literature**, then retrieve reusable full text through appropriate
open-access mechanisms.

## 3.4 Decision

**USE.**

Role:

> **Biomedical Literature Discovery + Metadata**

------------------------------------------------------------------------

# 4. PubMed Central (PMC) Open Access Subset

**Official source:** https://pmc.ncbi.nlm.nih.gov/

PMC is especially important because the Open Access Subset contains
millions of full-text articles available under Creative Commons or
similar terms that permit reuse, subject to each article's license.

PMC currently states that:

-   not all PMC articles are available for text mining/reuse;
-   license terms vary by article;
-   automated retrieval must use the approved PMC services;
-   the Open Access Subset can be accessed through the PMC Cloud
    Service, OAI-PMH, E-utilities, or BioC API.

## 4.1 Why this is highly valuable

Our current dataset might tell us:

``` text
Chest pain + sweating
        |
        v
Myocardial infarction
```

A biomedical article can provide:

``` text
Myocardial infarction
        |
        +-- clinical presentation
        +-- risk factors
        +-- diagnostic findings
        +-- biomarkers
        +-- imaging
        +-- treatment evidence
        +-- outcomes
        +-- discussion
```

That is much richer evidence.

## 4.2 Suitability

**Suitability: EXTREMELY HIGH.**

This is one of the best candidates for the actual knowledge corpus.

## 4.3 Limitations

-   Not every PMC article is reusable.
-   Licenses vary.
-   Full-text corpus can become very large.
-   Articles require careful filtering.
-   Retractions/corrections must be handled.
-   We should preserve article metadata and license information.
-   We should not scrape PMC through arbitrary automated methods.

## 4.4 Decision

**CORE KNOWLEDGE SOURCE.**

Use a targeted subset rather than blindly ingesting millions of
articles.

------------------------------------------------------------------------

# 5. Europe PMC

**Official source:** https://europepmc.org/

Europe PMC provides programmatic access to biomedical literature and
supports APIs and bulk data resources.

Its current developer documentation reports access to:

-   over 33 million publications through the REST service;
-   around 10.2 million full-text articles;
-   around 6.5 million open-access articles;
-   text-mined annotations for chemicals, diseases, genes, proteins and
    other entities.

Europe PMC also integrates content from PubMed, PMC and other sources.

## 5.1 Why it is useful

Europe PMC is especially attractive for our engineering workflow because
it gives us:

``` text
Search
  |
  v
Filter
  |
  v
Retrieve metadata
  |
  v
Retrieve open-access full text where permitted
  |
  v
Extract
  |
  v
Chunk
  |
  v
Embed
  |
  v
Qdrant
```

## 5.2 Suitability

**Suitability: EXTREMELY HIGH as our literature acquisition/integration
layer.**

It can complement PubMed/PMC rather than being treated as a completely
separate duplicate corpus.

## 5.3 Limitations

-   Full text is not universally available.
-   Copyright/license conditions still apply.
-   Some content includes preprints, which are not equivalent to
    peer-reviewed articles.
-   Europe PMC is partly an aggregation/indexing infrastructure rather
    than an independent clinical evidence authority.

## 5.4 Decision

**CORE ACQUISITION/SEARCH SOURCE.**

Potentially our main programmatic route for selecting an open-access
biomedical literature subset.

------------------------------------------------------------------------

# 6. MedlinePlus

**Official source:** https://medlineplus.gov/

MedlinePlus is operated by the U.S. National Library of Medicine.

It provides health information covering:

-   symptoms
-   causes
-   treatment
-   prevention
-   diseases and health conditions
-   medical tests
-   genetics
-   medications

The current MedlinePlus overview reports health-topic information for
more than 1,000 diseases/illnesses/health conditions and descriptions of
nearly 300 medical tests.

## 6.1 Why it complements our system

Biomedical literature is often technical.

MedlinePlus provides more structured, curated explanatory information.

This can improve retrieval for queries such as:

``` text
What is asthma?
What are symptoms of asthma?
What test is used?
What does the test mean?
What treatments are commonly discussed?
```

## 6.2 Suitability

**Suitability: VERY HIGH.**

Especially useful for:

-   disease summaries
-   symptom descriptions
-   medical tests
-   patient-oriented explanations

## 6.3 Limitations

-   It is not a replacement for primary research.
-   It is not intended to provide the depth of a complete research
    literature corpus.
-   It is U.S.-oriented in some aspects.
-   It should not be mixed with research articles without retaining
    source type.

## 6.4 Decision

**USE.**

Role:

> **Curated Disease / Symptom / Medical-Test Knowledge Layer**

------------------------------------------------------------------------

# 7. WHO Guidelines

**Official source:** https://www.who.int/publications/who-guidelines

WHO provides guidelines and normative health recommendations.

## 7.1 Why it matters

A research article might report an observed result.

A guideline can represent a formally developed recommendation based on
an evidence-review process.

For a clinical decision-support RAG system, this is an important
distinction.

Potential knowledge:

``` text
Disease
  |
  +-- recommended management
  +-- prevention
  +-- public-health guidance
  +-- treatment recommendations
  +-- diagnostic guidance
```

## 7.2 Suitability

**Suitability: VERY HIGH, but targeted rather than massive ingestion.**

## 7.3 Limitations

-   Guidelines cover particular diseases/conditions and topics.
-   They can become outdated.
-   Individual guideline licensing must be respected.
-   WHO guidelines are not intended to replace clinician judgment.
-   Not every clinical question has a relevant WHO guideline.

## 7.4 Decision

**USE SELECTIVELY.**

We should ingest only guidelines relevant to the diseases/topics our
analyser supports.

------------------------------------------------------------------------

# 8. DailyMed

**Official source:** https://dailymed.nlm.nih.gov/

DailyMed provides drug labeling information.

The current DailyMed download page provides large periodic and
full-release collections of drug labels, including human prescription
and OTC label archives.

Drug labels can contain clinically important sections such as:

-   indications
-   contraindications
-   warnings
-   adverse reactions
-   drug interactions
-   dosage
-   use in specific populations

## 8.1 Why this is critical for our analyser

Our clinical input explicitly includes:

``` json
{
  "diseases": [],
  "symptoms": [],
  "medications": [],
  "tests": []
}
```

Our current symptom-disease dataset cannot adequately answer
medication-related questions.

DailyMed fills that gap.

## 8.2 Suitability

**Suitability: EXTREMELY HIGH for the medication component.**

## 8.3 Limitations

-   The complete archive is enormous.
-   Drug labels are not the same as clinical guidelines.
-   Label information can change over time.
-   U.S. labeling may differ from regulatory information in other
    countries.
-   We should not ingest the entire archive for a student project.

## 8.4 Decision

**CORE SOURCE, BUT TARGETED INGESTION ONLY.**

Instead of downloading every label:

``` text
User medication
      |
      v
Normalize medication
      |
      v
Retrieve relevant label
      |
      v
Extract useful sections
      |
      v
Qdrant
```

------------------------------------------------------------------------

# 9. openFDA Drug Label Data

**Official source:** https://open.fda.gov/apis/drug/label/

openFDA provides machine-readable FDA data and a Drug Labeling API.

The current download page provides zipped JSON datasets, and the API
supports field-based searching.

## Suitability

**Suitability: HIGH.**

It is particularly useful when we want programmatic, structured access
to drug-label information.

## Strengths

-   JSON
-   API
-   searchable fields
-   downloadable data
-   machine-friendly

## Limitations

-   Data updates can change old records.
-   Large downloads can be substantial.
-   It represents FDA data, not all global medication knowledge.
-   It overlaps substantially with the DailyMed/FDA labeling domain.

## Decision

**USE AS A TECHNICAL/API SOURCE OR ALTERNATIVE TO BULK DAILYMED
INGESTION.**

We do not need to blindly ingest both full DailyMed and full openFDA
drug-label corpora.

------------------------------------------------------------------------

# 10. RxNorm

**Official source:** https://www.nlm.nih.gov/research/umls/rxnorm/

RxNorm is produced by NLM and provides normalized names for clinical
drugs and links between drug vocabularies.

Example:

``` text
Tylenol
Acetaminophen
Paracetamol
```

can be mapped toward a standardized drug concept.

## Suitability

**Suitability: VERY HIGH as retrieval infrastructure.**

## Important point

RxNorm should NOT primarily be treated as a giant document collection.

Instead:

``` text
User medication
      |
      v
RxNorm normalization
      |
      v
Canonical drug concept
      |
      v
DailyMed / FDA evidence retrieval
```

## Limitations

-   It is terminology/normalization data, not a clinical evidence
    corpus.
-   It does not replace drug labels or clinical guidelines.

## Decision

**USE AS A NORMALIZATION LAYER.**

------------------------------------------------------------------------

# 11. MeSH

**Official source:** https://www.nlm.nih.gov/mesh/

Medical Subject Headings (MeSH) is a controlled and hierarchically
organized biomedical vocabulary produced by NLM.

It is used for indexing and searching biomedical information.

MeSH provides:

-   controlled terminology
-   hierarchical relationships
-   entry terms/synonyms
-   descriptors
-   qualifiers
-   relationships between concepts

## Suitability

**Suitability: EXTREMELY HIGH for retrieval improvement.**

It can help our system understand that different expressions may refer
to related biomedical concepts.

Example:

``` text
heart attack
myocardial infarction
acute myocardial infarction
```

can be handled through standardized terminology.

## Limitations

-   It is not a medical evidence corpus.
-   It should not be used alone to answer clinical questions.
-   It changes over time, so versioning matters.

## Decision

**USE AS A RETRIEVAL/QUERY NORMALIZATION LAYER.**

------------------------------------------------------------------------

# 12. HPO --- Human Phenotype Ontology

**Official source:** https://hpo.jax.org/

HPO is a standardized vocabulary for phenotypic abnormalities and
clinical findings.

## Why it is useful

Our input is heavily symptom/phenotype based.

A normalization layer can help connect expressions such as:

``` text
shortness of breath
dyspnea
breathlessness
```

to a standardized clinical concept where appropriate.

## Suitability

**Suitability: HIGH for symptom/query normalization.**

## Limitations

-   It is terminology/ontology data rather than a replacement for
    medical evidence.
-   It should be used to improve retrieval rather than to generate
    unsupported diagnoses.

## Decision

**USE LATER AS PART OF RETRIEVAL ENHANCEMENT.**

------------------------------------------------------------------------

# 13. ICD-11

**Official source:** https://icd.who.int/

ICD-11 is WHO's international disease classification system.

## Suitability

**Suitability: MEDIUM-HIGH for disease normalization/classification.**

It can help map disease concepts to standardized classifications.

## Limitations

-   It is primarily a classification system.
-   It is not a disease-treatment knowledge base.
-   It should not be used as the main textual corpus.

## Decision

**OPTIONAL / LATER.**

------------------------------------------------------------------------

# 14. BioASQ

**Official source:** https://participants-area.bioasq.org/datasets/

BioASQ provides biomedical question-answering and information-retrieval
datasets.

The current BioASQ page lists thousands of biomedical questions for
recent editions and large PubMed/MeSH-based article datasets for Task A.

## Why it is interesting

BioASQ is particularly useful for:

> **evaluating whether our biomedical retrieval system actually works.**

It provides questions and gold/relevant evidence structures.

## Suitability

**Suitability for knowledge enrichment: MEDIUM.**

**Suitability for retrieval evaluation: EXTREMELY HIGH.**

## Limitations

-   It is primarily an evaluation/benchmark resource.
-   Its large article datasets can be enormous.
-   It should not become our main clinical knowledge corpus.

## Decision

**USE LATER FOR RETRIEVAL EVALUATION.**

------------------------------------------------------------------------

# 15. MIMIC-IV / MIMIC-IV-Note

**Official source:** https://physionet.org/

MIMIC contains deidentified clinical data, including clinical notes in
MIMIC-IV-Note.

## Why it is attractive

It contains real-world clinical data and can provide a very different
type of evidence from curated disease datasets.

## Why we should NOT start with it

Access to restricted MIMIC resources requires credentialing and a Data
Use Agreement, along with required training.

This introduces unnecessary complexity for our current project stage.

## Suitability

**Suitability: HIGH research value.**

**Suitability for immediate project implementation: LOW-MEDIUM.**

## Decision

**DEFER.**

It can be considered later if the project requires real-world
clinical-note research.

------------------------------------------------------------------------

# 16. Comparative Evaluation

  -----------------------------------------------------------------------------------------------------------------------------
  Resource          Main knowledge               Authority        RAG          Access/API     Main limitation   Decision
                                                                  usefulness
  ----------------- ---------------------------- ---------------- ------------ -------------- ----------------- ---------------
  Current           Symptoms → diseases          Medium /         High         Easy           Duplication,      KEEP
  symptom-disease                                insufficiently                               limited clinical
  dataset                                        documented                                   depth

  PubMed            Biomedical literature        Very high        Very high    Excellent      Full text not     USE
                    metadata/abstracts                                                        universal

  PMC OA            Full-text biomedical         Very high        Extremely    Excellent      License varies    CORE
                    evidence                                      high

  Europe PMC        Literature + OA full text +  Very high        Extremely    Excellent      Not all full text CORE
                    annotations                                   high                        reusable

  MedlinePlus       Diseases, symptoms, tests    Very high        Very high    Good           Less research     USE
                                                                                              depth

  WHO guidelines    Guidelines/recommendations   Very high        Very high    Good           Topic-specific,   USE SELECTIVELY
                                                                                              versioning

  DailyMed          Drug labels                  Very high        Extremely    Excellent      Huge corpus       CORE, TARGETED
                                                                  high for
                                                                  meds

  openFDA           Machine-readable FDA drug    Very high        High         Excellent      Overlap with      USE AS API/DATA
                    data                                                                      labels            OPTION

  RxNorm            Drug normalization           Very high        Extremely    Excellent      Not evidence      NORMALIZATION
                                                                  high for
                                                                  retrieval

  MeSH              Biomedical terminology       Very high        Extremely    Excellent      Not evidence      NORMALIZATION
                                                                  high for
                                                                  retrieval

  HPO               Phenotype terminology        High             High         Good           Not evidence      LATER
                                                                                                                NORMALIZATION

  ICD-11            Disease classification       Very high        Medium       Good           Not evidence      OPTIONAL

  BioASQ            Biomedical QA/retrieval      High             High for     Good           Benchmark, not    EVALUATION
                    benchmark                                     evaluation                  main corpus

  MIMIC             Clinical notes/data          Very high        High         Credentialed   Access            DEFER
                                                 research value                               requirements
  -----------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 17. What We Should Actually Use

The goal is not to maximize the number of datasets.

The goal is to maximize **useful, complementary knowledge**.

## Tier 1 --- Core knowledge sources

### 1. Existing symptom-disease dataset

Keep it.

Purpose:

> Symptom → disease associations.

### 2. PMC Open Access / Europe PMC

Add a targeted biomedical literature collection.

Purpose:

> Detailed biomedical evidence.

### 3. MedlinePlus

Add relevant disease/test information.

Purpose:

> Curated disease, symptom and medical-test knowledge.

### 4. DailyMed / openFDA

Add targeted medication information.

Purpose:

> Drug indications, contraindications, warnings, adverse reactions and
> label information.

### 5. Selected WHO guidelines

Add only guidelines relevant to the conditions and use cases supported
by the analyser.

Purpose:

> Guideline-level evidence.

------------------------------------------------------------------------

# 18. Tier 2 --- Retrieval Intelligence

These should not simply become another document pile.

### MeSH

Purpose:

> Biomedical terminology + query expansion + metadata.

### RxNorm

Purpose:

> Medication normalization.

### HPO

Purpose:

> Symptom/phenotype normalization.

### ICD-11

Purpose:

> Disease classification/normalization.

This is where our future **unique retriever** can become substantially
better.

------------------------------------------------------------------------

# 19. Tier 3 --- Evaluation

### BioASQ

Purpose:

> Test whether our retriever retrieves appropriate biomedical evidence.

This is important because adding more documents does not automatically
mean better retrieval.

We need to eventually measure:

``` text
Before enrichment
       ↓
Current retriever
       ↓
Evaluation
       ↓
After enrichment
       ↓
Improved retriever
       ↓
Evaluation
```

------------------------------------------------------------------------

# 20. What We Should NOT Do

## Do not ingest everything from PubMed

Too large and unnecessary.

## Do not ingest the entire DailyMed archive

The current full releases are multi-gigabyte archives. We only need
relevant medication information.

## Do not treat MeSH/RxNorm/HPO as ordinary documents

They are more valuable as structured terminology and normalization
resources.

## Do not randomly download medical PDFs

Source quality and licensing matter.

## Do not mix all sources without metadata

Every chunk should preserve provenance.

Example:

``` json
{
  "source_type": "research_article",
  "source": "PMC",
  "pmid": "...",
  "pmcid": "...",
  "title": "...",
  "publication_date": "...",
  "license": "...",
  "disease": "...",
  "section": "Results"
}
```

For a drug label:

``` json
{
  "source_type": "drug_label",
  "source": "DailyMed",
  "drug_name": "...",
  "section": "Contraindications",
  "label_date": "..."
}
```

This metadata will later become extremely important for retrieval
ranking.

------------------------------------------------------------------------

# 21. Final Recommended Architecture

``` text
                         USER CLINICAL INPUT
                                  |
             +--------------------+--------------------+
             |                    |                    |
          Symptoms              Disease           Medication
             |                    |                    |
             v                    v                    v
            HPO                 MeSH               RxNorm
             |                    |                    |
             +--------------------+--------------------+
                                  |
                           Query Enrichment
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
 Symptom-Disease           Biomedical Literature       Drug Knowledge
    Dataset                  PMC / Europe PMC        DailyMed / FDA
        |                         |                         |
        |                         +-----------+-------------+
        |                                     |
        |                              MedlinePlus
        |                                     |
        |                              WHO Guidelines
        |                                     |
        +------------------+------------------+
                           |
                      Qdrant / Retrieval
                           |
                  Evidence Reranking
                           |
                  Evidence Context
                           |
                       RAG Answer
```

------------------------------------------------------------------------

# 22. Final Decision

For the **actual knowledge enrichment phase**, the recommended stack is:

### CORE

1.  **Existing `dux-tecblic/symptom-disease-dataset`**
2.  **PMC Open Access / Europe PMC biomedical literature**
3.  **MedlinePlus**
4.  **Targeted DailyMed or openFDA drug-label information**
5.  **Selected WHO guidelines**

### RETRIEVAL SUPPORT

6.  **MeSH**
7.  **RxNorm**
8.  **HPO**

### EVALUATION

9.  **BioASQ**

### DEFER

10. **MIMIC-IV/MIMIC-IV-Note**

The strongest immediate combination is therefore:

> **Symptom-Disease Dataset + Open-Access Biomedical Literature +
> MedlinePlus + Targeted Drug Labels + Selected Guidelines**

This gives the Symptom RAG Analyser five complementary evidence types:

``` text
SYMPTOM ASSOCIATION
        +
RESEARCH EVIDENCE
        +
CURATED MEDICAL INFORMATION
        +
MEDICATION SAFETY
        +
GUIDELINE KNOWLEDGE
```

That is much more valuable than simply increasing the number of symptom
records.

------------------------------------------------------------------------

# 23. Important Clinical-System Limitation

Even with all these sources, the system should be described as an
**evidence-retrieval and clinical decision-support prototype**, not as a
system that independently diagnoses patients.

More knowledge can improve retrieval coverage and evidence grounding,
but it does not guarantee diagnostic correctness.

The project should preserve:

-   source provenance
-   publication/update dates
-   source type
-   licensing information
-   evidence confidence
-   retrieval scores
-   explicit limitations
-   uncertainty when evidence is insufficient

------------------------------------------------------------------------

# 24. Official References

1.  PubMed --- https://pubmed.ncbi.nlm.nih.gov/
2.  PMC Article Datasets ---
    https://pmc.ncbi.nlm.nih.gov/tools/textmining/
3.  PMC Open Access Subset ---
    https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
4.  Europe PMC --- https://europepmc.org/
5.  Europe PMC REST API ---
    https://www.ebi.ac.uk/europepmc/webservices/rest/
6.  MedlinePlus --- https://medlineplus.gov/
7.  WHO Guidelines --- https://www.who.int/publications/who-guidelines
8.  WHO Open Access Policy ---
    https://www.who.int/about/policies/publishing/open-access
9.  DailyMed --- https://dailymed.nlm.nih.gov/
10. DailyMed bulk drug labels ---
    https://www.dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
11. openFDA Drug Labels --- https://open.fda.gov/apis/drug/label/
12. RxNorm --- https://www.nlm.nih.gov/research/umls/rxnorm/
13. MeSH --- https://www.nlm.nih.gov/mesh/
14. MeSH RDF/API --- https://id.nlm.nih.gov/mesh/
15. BioASQ datasets --- https://participants-area.bioasq.org/datasets/
16. PhysioNet / MIMIC --- https://physionet.org/
17. Existing symptom-disease dataset ---
    https://huggingface.co/datasets/dux-tecblic/symptom-disease-dataset

------------------------------------------------------------------------

## Bottom Line

**Do not start coding the enrichment yet.**

The resource selection is now sufficiently clear.

The next engineering task should be to design the **exact
subset-selection strategy**:

``` text
Which diseases?
Which articles?
Which sections?
How many articles?
Which medications?
Which guidelines?
What metadata?
What licenses?
What chunk types?
What source priorities?
```

Only after that should we modify the ingestion pipeline and Qdrant
schema.
